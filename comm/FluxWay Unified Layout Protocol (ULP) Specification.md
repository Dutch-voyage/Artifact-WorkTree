# ULP 分布式传输协议系统设计文档
## **1. 概述 (Overview)**

ULP (Unified Layout Protocol) 是一种高性能、架构无关（Framework-Agnostic）的分布式数据传输范式。它旨在解决大规模**模型训练**中 **Dense（静态稠密）** 与 **Sparse（动态稀疏）** 数据混合并存的复杂通信问题。

在复杂的混合模态模型中，**不同模型部分的计算访存比不一样导致每个部分的最佳并行方案和并行度不一样**，对于SPMD范式来说，很难实现不同部分的硬件设备分配最优化。如果采用多SPMD（MPMD）范式，则需要解决不同并行策略的**SPMD-to-SPMD**产生的Tensor **N-to-M问题**。

系统的核心设计哲学是**“静态规划，动态执行”**：

- **统一视图**：将规则切分（Sharding）、Hash/MoE 路由、以及全量广播（Replication）统一视为数据在不同 Layout 之间的变换。
- **向量化路由**：拒绝 Python层面的循环，所有路由计算下沉至底层框架（JAX/Torch/NumPy）的向量化指令中。
- **票据栅栏 (Ticket Barrier)**：接收端采用无锁的信号量计数机制，实现零同步开销的数据重组。

## **2. 核心概念与抽象 (Core Abstractions)**

为了屏蔽底层物理设备的复杂性，系统向用户和编译器提供了两层抽象。其中Mesh和ShardingSpec相互配合的概念来自Jax和GSPMD

### **2.1 Mesh与ShardingSpec**

Mesh 描述的是设备的逻辑布局，而 sharding spec 描述的是张量各个维度如何映射到这个布局上。给定一个设备 mesh（例如 2×4 的二维网格）和一个张量（例如 shape 为 [8, 16]），sharding spec 可以指定第 0 维沿 mesh 的第 0 维切分、第 1 维沿 mesh 的第 1 维切分。两者组合后，张量会被切成多个子块，每个子块对应 mesh 中的一个逻辑坐标，并最终放置到该坐标对应的物理设备上。Mesh 决定“设备怎么排”，sharding spec 决定“数据怎么切并放到这些设备上”。

```python
# 一个 2D 设备 mesh：2 × 4 = 8 个设备
mesh_shape = {
    "data": 2,
    "model": 4,
}

# 一个张量
tensor_shape = (8, 16)  # [batch, hidden]

# sharding spec：张量维度 -> mesh 维度
sharding_spec = {
    0: "data",   # batch 维度沿 data 轴切分
    1: "model", # hidden 维度沿 model 轴切分
}

# 由此推导出的每个 shard 的形状
per_shard_shape = (
    tensor_shape[0] // mesh_shape["data"],   # 8 / 2 = 4
    tensor_shape[1] // mesh_shape["model"],  # 16 / 4 = 4
)

# 直观理解：
# - 张量被切成 2 × 4 = 8 个 (4, 4) 的子块
# - mesh 中每个逻辑坐标 (data_i, model_j)
#   持有其中一个子块

```

### **2.2 分区算子体系 (Partition Operators)**

所有数据传输本质上都是由源 Layout 到目标 Layout 的变换。系统定义了一组原子算子来处理这些变换，这些算子在发送端以 **Vectorized Kernel** 形式执行。

|**算子类型**|**逻辑描述**|**典型场景**|
|---|---|---|
|**BLOCK (切块)**|连续几何切分。基于 Rank ID 计算全局切片范围。|数据并行 (DP)、张量并行 (TP)|
|**HASH / CYCLIC**|PID = Hash(Value) % Size 或 Index % Size。|Embedding, 负载均衡|
|**LOOKUP (查表)**|PID = Table[Index]。依赖上游生成的 Routing Table。|MoE Router 分发|
|**REPLICATE**|PID = All Ranks。数据被复制到所有目标节点。|参数广播、全局配置同步|

```python
class LayoutOp(IntEnum):
    # 静态切分 (Static / Dense)
    BLOCK = auto()       # 连续切分: rank = index // size

    # 动态切分 (Dynamic / Sparse)
    CYCLIC = auto()      # 轮询: rank = index % size
    HASH = auto()        # 哈希: rank = hash(val) % size
    LOOKUP = auto()      # 查表: rank = table[val]

    # 特殊
    REPLICATE = auto()   # 广播: All ranks

```

### **2.3 系统视图：Internal Layout**

编译器将用户 Spec 下降（Lowering）后的物理描述，是单节点（Rank）视角的真值。

- **Dense和Sparse类型**

```python
class LayoutType(Enum):
    DENSE = 1   # 连续几何切块 (涵盖 Replicated, Sharded, Partial)
    SPARSE = 2  # 离散集合 (Hash/Mod/Lookup)

```

- **物理网格 (Mesh)**：当前节点在全局设备阵列中的多维坐标。

```python
class LayoutMesh:
    device_mesh: np.ndarray
    axis_names: Tuple[str, ...]

    def __init__(self, axis_shapes: Tuple[int, ...], axis_names: Tuple[str, ...]):
        device_num = np.prod(axis_shapes)
        device_mesh = np.reshape(np.arange(0, device_num), axis_shapes)
        self.axis_names = axis_names

    def __post_init__(self):
        assert len(self.device_mesh.shape) == len(self.axis_names)
        self.axis_map = dict(zip(self.axis_names, self.device_mesh.shape))

    def get_axis_size(self, axis_name: str) -> int:
        return self.axis_map.get(axis_name, 1)

    def get_coordinate(self, rank_id: int) -> Tuple[int, ...]:
        """获取 rank 在 mesh 中的多维坐标"""
        # np.argwhere returns an array of coordinates, we take the first (and only) match
        coords = np.argwhere(self.device_mesh == rank_id)
        if len(coords) == 0:
            raise ValueError(f"Rank {rank_id} not found in mesh")
        return tuple(coords[0])

    def to_engine_mesh(engine: Any):
        """转换成对应框架的mesh"""
        pass

# 可以通过int或name str来表示哪个轴
@dataclass
class Axis:
    axis: Tuple[Union[int, str], ...]

```

- **几何切片 (Dense Slice)**：对于稠密数据，计算出当前节点持有的全局数据范围 [Start, End)。

```python
@dataclass
class DenseLayoutSlice:
    start: int
    end: int

    @property
    def size(self): return self.end - self.start

```

- **路由规则 (Sparse Rule)**：对于稀疏数据，持有用于计算数据归属权的算法句柄（如 Routing Table 或 Hash Function）。

```python
@dataclass
class SparseLayoutRule:
    """
    描述 Tensor 在单节点视角的稀疏/动态分布规则。
    不包含物理目的地信息，只包含逻辑归属算法。
    """

    # 1. 切分维度 (Normalized)
    # 指示当前 Tensor 的哪些维度是动态分布的。
    # 即使是单轴，也统一存储为 Tuple (e.g., (1,)) 以支持多轴联合 Hash。
    axis: Axis

    # --- 互斥逻辑 A: 自计算 (Self-Calculated) ---
    # 如果设置了 method，说明该 Tensor 自带路由属性 (如 Key 本身)。
    method: Optional[LayoutOp] = None

    # 额外元数据，用于存储算子所需的静态参数
    # e.g., Hash 的 seed, Modulus 的大小 (如果不是 World Size)
    meta: Dict[str, Any] = field(default_factory=dict)

    # --- 互斥逻辑 B: 依赖跟随 (Dependent) ---
    # 如果设置了 sharding_dependency，说明路由逻辑由其他 Tensor 决定。

    # 依赖某个目标Tensor进行sharding时的行为设置（定义见用户视图）
    sharding_dependency: Optional[ShardingDependency] = None

    def is_dependent(self) -> bool:
        """Helper: 是否是跟随型规则"""
        return self.dependency_ptr is not None

    def __post_init__(self):
        # 简单校验：不能既没有 method 也没有 dependency，也不能两者都有
        has_method = self.method is not None
        has_dep = self.sharding_dependency is not None

        # 注意：某些特殊情况可能允许 None (表示完全动态未知)，但一般应当互斥
        if has_method and has_dep:
            raise ValueError("SparseLayoutRule cannot have both 'method' and 'dependency'.")

```

- **总类：**

```python
@dataclass
class InternalLayout:
    type: LayoutType
    mesh: LayoutMesh
    global_shape: Tuple[int, ...]

    # Dense 专用
    # 核心：用对 Slice 表达一切。
    # - 完全复制: slice(0, global_len)
    # - 切分: slice(start, end)
    # - Partial Replication: 混合上述两种
    # Tuple长度等于 global_shape 维度。
    dense_slices: Optional[Tuple[DenseLayoutSlice, ...]] = None

    # Sparse 专用
    sparse_rule: Optional[SparseLayoutRule] = None

    def get_local_shape(self) -> Tuple[int, ...]:
        if self.type == LayoutType.DENSE and self.dense_slices:
            return tuple(s.size for s in self.dense_slices)
        # Sparse shape is dynamic, unknown at compile time without data
        return (-1,)

```

### **2.4 用户视图：Sharding Spec**

描述“数据**应该**如何分布”，不关注物理拓扑细节。

```python
class ShardingSpec:
    """
    用户可见的 Sharding 描述基类。
    """
    layout_mesh: LayoutMesh

    def lower(self, global_shape: Tuple[int], rank_id: int) -> InternalLayout:
        """
        将高级 Spec 编译为底层的 InternalLayout。
        """
        raise NotImplementedError

    def to_engine_spec(engine: Any):
        """转换成对应框架的sharding spec"""
        raise NotImplementedError

class DependencyMode(Enum):
    # 模式 A: 影子跟随 (KV Cache)
    # 依赖项也是数据，它怎么切，我就怎么切。
    # 场景：Key 去 Rank 1，Value 也要去 Rank 1。
    # 隐含逻辑：compute_route(dependency_tensor) -> dest_ranks
    SHADOW = auto()

    # 模式 B: 显式指引 (MoE)
    # 依赖项是路由结果 (Indices)，它的值就是我的目的地。
    # 场景：TopK Indices 说第 i 个 token 去 Rank 5，那我就去 Rank 5。
    # 隐含逻辑：dependency_tensor IS dest_ranks
    DIRECT_INDICES = auto()

@dataclass
class ShardingDependency:
    """
    描述 Tensor 之间的切分依赖关系。
    """
    # 依赖的目标 Tensor 标识
    target_tensor_name: str

    # 依赖模式
    mode: DependencyMode

    # --- 支持多轴 ---
    # 本 Tensor 的哪个轴受ShardingDependency的控制
    # (通常与 ShardingSpec.sparse_axis 一致，可省略)
    # E.g., (0, 1) 表示前两维是动态路由的
    # source_axis: Axis = -1

    # 描述依赖 Tensor 的哪些维度提供指导
    # E.g., 如果 target 是 [B, S] 的 indices，这里就是 (0, 1)
    target_axis: Axis = 0

```

- **Static Sharding**：类似 JAX 的设计。定义数据维度与逻辑网格轴（Mesh Axis）的映射关系（如 ("data", "model")）。用于描述 Weights、Images 等静态形状张量。

```python
# --- 1. 兼容 JAX 的 Static Sharding ---
@dataclass
class StaticShardingSpec(ShardingSpec):
    """
    类似 JAX PartitionSpec。
    sharding_axis: ('data', 'model') 对应 global_shape 的每一维映射到 Mesh 的哪个轴。
    None 表示该维度不切分 (Replicated)。
    """
    layout_mesh: LayoutMesh
    sharding_axis: Tuple[Optional[Axis], ...]

    def lower(self, global_shape: Tuple[int, ...], rank_id: int) -> InternalLayout:
        assert len(global_shape) == len(self.sharding_axes)

        # 1. 获取当前 Rank 在 Mesh 中的坐标
        mesh_coords = self.layout_mesh.get_coordinate(rank_id) # e.g. (0, 1)
        mesh_axis_name_to_idx = {name: i for i, name in enumerate(mesh.axis_names)}

        slices = []

        for i, (dim_len, axis_name) in enumerate(zip(global_shape, self.sharding_axes)):
            if axis_name is None:
                # Replicate: 拥有完整维度
                slices.append(DenseLayoutSlice(0, dim_len))
            else:
                # Shard: 计算切片
                if axis_name not in mesh_axis_name_to_idx:
                    raise ValueError(f"Unknown mesh axis: {axis_name}")

                mesh_dim_idx = mesh_axis_name_to_idx[axis_name]
                num_devices_along_axis = mesh.device_mesh.shape[mesh_dim_idx]
                device_pos = mesh_coords[mesh_dim_idx]

                # 简单的 Block 切分逻辑 (可扩展处理不能整除的情况)
                chunk_size = math.ceil(dim_len / num_devices_along_axis)
                start = device_pos * chunk_size
                end = min(start + chunk_size, dim_len)

                slices.append(DenseLayoutSlice(start, end))

        return InternalLayout(
            type=LayoutType.DENSE,
            mesh=self.layout_mesh,
            global_shape=global_shape,
            dense_slices=tuple(slices)
        )

```

- **Dynamic Sharding**：描述动态数据的分布规则。指定在哪个轴上进行离散化，以及使用何种算法（如 Hash, Mod）将数据打散。用于描述 Embedding Lookup 或 MoE Token Dispatch。

```python
# --- 2. Fluxway 原生 Dynamic Sharding ---
@dataclass
class DynamicShardingSpec(ShardingSpec):
    """
    sparse_axis: 在哪个维度是动态的 (通常是 Batch 维或 Token 维)
    method: 切分方法 (HASH, CYCLIC, LOOKUP)
    """
    layout_mesh: LayoutMesh

    # 指定本 Tensor 在哪些维度上变得“稀疏/动态”。
    # 对于 MoE，通常是 batch 或 token 维。
    # 类型为 Axis，支持联合切分。
    sparse_axis: Axis

    # 自计算方法 (HASH/CYCLIC)
    method: LayoutOp

    # 依赖关系 (互斥：有 dependency 则忽略 method)
    sharding_dependency: Optional[ShardingDependency] = None

    # 对于 Lookup 或 Hash，可能需要额外的元数据
    meta: Optional[Dict] = None

    def lower(self, global_shape: Tuple[int, ...], rank_id: int) -> InternalLayout:
        # Sparse Layout 在 lower 阶段主要记录规则，具体数据归属依赖运行时数据
        rule = SparseLayoutRule(
            axis=self.sparse_axis, # 支持处理多轴
            method=self.method,
            sharding_dependency=self.sharding_dependency,
            meta=self.meta
        )

        return InternalLayout(
            type=LayoutType.SPARSE,
            mesh=self.layout_mesh,
            global_shape=global_shape,
            sparse_rule=rule
        )

```

### **2.2 用户使用举例：**

- **kernel装饰器**

```python
def spmd_kernel(
    input_sharding:  Union[ShardingSpec, Sequence[ShardingSpec], Dict[str, ShardingSpec]],
    output_sharding: Union[ShardingSpec, Sequence[ShardingSpec]],
    state_argnames:  Sequence[str] = None,
    static_argnames: Sequence[str] = None
):
    """
    input_sharding: 定义输入 Tensor 的分布。支持 jax.sharding 或 ulp.DynamicSharding。
    output_sharding: 定义输出 Tensor 的期望分布。
    state_argnames:  动态元数据 (int/float/str)，通过 SPMD driver to SPMD driver 传输，但不参与 Sharding 计算。
    static_argnames: 编译期常量，不参与 RPC 传输，触发重新编译 (Re-trace)。
    """

```

- **kernel装饰器使用举例**

```python
from typing import TypeAlias
Engine: TypeAlias = jax
Compute: TypeAlias = jax.numpy
Tensor: TypeAlias = jax.numpy.array

emb_mesh = LayoutMesh((5,), ("key_axis"))
emb_key_in_sharding = DynamicShardingSpec(emb_mesh, "key_axis", LayoutOp.HASH)
emb_quant_dependency = ShardingDependency(target_tensor_name="keys", mode=DependencyMode.SHADOW, target_axis="key_axis")
emb_quant_in_sharding = DynamicShardingSpec(emb_mesh, "key_axis", LayoutOp.HASH, sharding_dependency=emb_quant_dependency)
emb_value_out_sharding = DynamicShardingSpec(emb_mesh, "key_axis", LayoutOp.HASH)

@flux.spmd_kernel(
    input_sharding = {"keys": emb_key_in_sharding, "Quants": emb_quant_dependency},
    output_sharding = (emb_value_out_sharding),
    state_argnames = ["if_quant"],
    static_argnames = ["rate"],
)
def QuantEmbeddingLookup(keys: Tensor, quants: Tensor, if_quant: bool, rate: Tensor) -> values: Tensor :
    emb = rate * MagicEmb.lookup(keys)
    if if_quant:
        emb = emb * quants
    return emb

@flux.spmd_kernel(
    input_sharding = (emb_key_in_sharding),
    output_sharding = (emb_value_out_sharding),
)
def NormalEmbeddingLookup(keys: Tensor) -> values: Tensor :
    return rate * NonMagicEmb.lookup(keys)

# Dense Sharding是对jax sharding兼容的
dense_mesh = LayoutMesh((2,), ("dp", "mp"))
dense_in_sharding = StaticSharding(dense_mesh, ("dp", "mp"))
dense_out_sharding = StaticSharding(dense_mesh, ("dp", None))

emgine_mesh = dense_mesh.to_engine_mesh(Engine)
emgine_dense_in_sharding = dense_in_sharding.to_engine_spec(Engine)
emgine_dense_out_sharding = dense_out_sharding.to_engine_spec(Engine)

@Engine.jit(
    in_shardings = (dense_in_sharding, dense_in_sharding),
    out_shardings = dense_out_sharding
)
def engine_ffn(x_0: Tensor, x_1: Tensor):
    x = Compute.concat(x_0, x_1)
    x = FFNLayer(x)
    return x

@flux.spmd_kernel(
    input_sharding = (dense_in_sharding, dense_in_sharding),
    output_sharding = (dense_out_sharding),
)
def FFN(x_0: Tensor, x_1: Tensor):
    return ENGINE_FFN(x_0, x_1)

# 编写模型（理想状态用函数式，但底层其实是编辑计算图，目前先做计算图模式）
# fluxway组装图的时候的逻辑算子有(cond、while_loop、control_dependency)
input_data = SomeWhereWithOutSharding()
quant_values = QuantEmbeddingLookup(input_data)
normal_values = QuantEmbeddingLookup(input_data)
result = FFN(quant_values, normal_values)

```

## **3. 通信规划与执行 (Communication Workflow)**

系统将通信分为 **Compiler 规划阶段** 和 **Runtime 执行阶段**。

### **3.1 发送端：向量化路由 (Vectorized Sender)**

发送端不仅仅是搬运数据，它负责计算“数据去哪儿”。这是系统的核心。我们需要一个 Planner 来决定数据如何从 Source Layout 移动到 Dest Layout。

**设计思路：**

- **RpcTask**: 定义基本的传输单元 (src_rank, dst_rank, data_slice_or_indices)。
- **Dense <-> Dense**: 纯几何计算（求交集）。
- **Dense -> Sparse**: 相当于 Scatter/Dispatch。Dense 端需要计算 Hash/Mod，算出数据该去哪里。
- **Sparse -> Dense**: 相当于 Gather/Combine。Sparse 端持有数据，但需要知道这些数据在 Global View 的位置，从而发送给对应的 Dense Owner。
- **Sparse -> Sparse**: 相当于 Reshuffle (e.g., Hash -> Cyclic)。

```python
@dataclass
class RpcTask:
    src_rank: int
    dst_rank: int
    # 传输描述
    # 若是 Dense: 传递 Slice (start, end)
    # 若是 Sparse: 传递 Indices 列表或 Filter Mask
    meta: Any

class CommunicationPlanner:

    @staticmethod
    def plan(src_layout: InternalLayout, dst_layout: InternalLayout,
             rank_id: int, world_size: int) -> List[RpcTask]:
        """
        生成当前 rank 需要执行的发送/接收任务。
        注意：这里为了演示逻辑，是在上帝视角计算（或者假设每个rank都知道全局拓扑）。
        实际部署时，通常每个Rank只计算自己相关的 "My Send Tasks"。
        """

        # 1. Dense to Dense (Static Geometric Intersection)
        if src_layout.type == LayoutType.DENSE and dst_layout.type == LayoutType.DENSE:
            return CommunicationPlanner._plan_dense_to_dense(src_layout, dst_layout, rank_id, world_size)

        # 2. Dense to Sparse (Dynamic Dispatch / Scatter)
        elif src_layout.type == LayoutType.DENSE and dst_layout.type == LayoutType.SPARSE:
            return CommunicationPlanner._plan_dense_to_sparse(src_layout, dst_layout, rank_id)

        # 3. Sparse to Dense (Restore / Gather)
        elif src_layout.type == LayoutType.SPARSE and dst_layout.type == LayoutType.DENSE:
            return CommunicationPlanner._plan_sparse_to_dense(src_layout, dst_layout, rank_id)

        # 4. Sparse to Sparse (Reshuffle)
        elif src_layout.type == LayoutType.SPARSE and dst_layout.type == LayoutType.SPARSE:
            # 逻辑类似：先 Sparse->Global->Sparse 或直接重映射
            # 这里简化处理，通常涉及 AllToAll
            pass

        return []

    @staticmethod
    def _plan_dense_to_dense(src: InternalLayout, dst: InternalLayout,
                             my_rank: int, world_size: int) -> List[RpcTask]:
        tasks = []
        # 这里需要遍历所有可能的 dst_rank 来检查是否有重叠
        # (实际优化：只遍历 mesh 里的 neighbor 或者根据 axis 推导)

        # 假设我们只计算 "我是发送方 (src_rank == my_rank)" 的任务
        src_slices = src.dense_slices

        # 模拟遍历所有其他 rank 的 layout (实际系统中可以通过 Mesh 公式直接算)
        # 这里简化代码，假设有一个 helper 能拿到 layout
        # 实际上应该根据 dst.mesh 的切分逻辑反推谁拥有哪块数据

        # ...省略 Mesh 遍历代码，直接展示核心交集逻辑...

        # 伪代码逻辑：
        # for target_rank in all_ranks:
        #     target_slices = get_layout(dst, target_rank).dense_slices
        #     overlap = calc_intersection(src_slices, target_slices)
        #     if overlap:
        #         tasks.append(RpcTask(my_rank, target_rank, overlap))

        return tasks

    @staticmethod
    def _plan_dense_to_sparse(src: InternalLayout, dst: InternalLayout, my_rank: int) -> List[RpcTask]:
        """
        Dense -> Sparse (Forward):
        我是 Dense 持有者。我需要查看 dst 的 sparse_rule (例如 Hash)。
        如果不运行实际数据，我无法知道具体发给谁。

        **关键设计**: 这是一个 Runtime Plan。
        Compiler 生成的是 "Dispatch Kernel" 的指令。
        """
        rule = dst.sparse_rule
        # 返回一个特殊的 Task，指示 Runtime 执行 Dispatch
        # Runtime 行为：
        # 1. 取出 src data (local slice)
        # 2. 对 rule.axis 维度执行 rule.method (e.g. hash(data[axis]))
        # 3. 计算 target_rank = hash_val % mesh_size
        # 4. Pack data 并执行 AllToAll
        return [RpcTask(my_rank, -1, meta={"op": "dispatch", "rule": rule})]

    @staticmethod
    def _plan_sparse_to_dense(src: InternalLayout, dst: InternalLayout, my_rank: int) -> List[RpcTask]:
        """
        Sparse -> Dense (Forward / Restore):
        我是 Sparse 持有者（手头有一堆离散数据）。
        我要把它们送回原本属于它们的几何位置 (Dense Block)。
        """
        # Runtime 行为：
        # 1. 每一条 Sparse 数据必须携带其 "Original Global Index" (Metadata)。
        # 2. 根据 Global Index 和 dst.mesh 的 Block 逻辑，计算 target_rank。
        #    target_rank = (global_index // block_size) % device_count
        # 3. 发送数据
        return [RpcTask(my_rank, -1, meta={"op": "combine", "dst_layout": dst})]

```

1. **静态求交 (Dense-to-Dense)**：

- 如果源和目标都是静态切分，Planner 直接计算源切片与目标切片的几何交集。无需扫描实际数据，开销极低。

1. **动态分发 (Dense-to-Sparse/Sparse-to-Dense/Sparse-to-Sparse/)**：

- **计算 (Compute)**：对本地数据执行向量化算子（如 Hash），生成目标 Rank 索引数组。
- **重排 (Sort & Group)**：使用 argsort 或 bucketize 对数据进行分组，将发往同一 Rank 的数据聚合。
- **空包保活 (Keep-Alive)**：**关键协议**。如果通过计算发现某目标 Rank 本次不需要接收数据，发送端**必须**发送一个长度为 0 的空包（Empty Payload）。这是为了防止接收端的栅栏机制死锁。

1. **异步发射 (Fire Async)**：

- 通过底层的 RPC/NCCL 接口，对所有目标 Rank 发起异步推送。

### **3.2 接收端：票据栅栏 (Ticket Barrier Receiver)**

接收端被设计为一个被动的、无锁的状态机，它不关心数据来自哪个具体的源节点，只关心“是否收齐了所有碎片”。

1. **依赖注册 (Dependency Registration)**：

- 在 Batch 开始前，系统根据拓扑图计算出当前节点应当接收到的数据包数量（Expected Shards），并初始化一个计数器（Ticket）。

1. **原子接收 (Atomic Receive)**：

- 每收到一个 RPC 请求（无论是否为空包），计数器原子加一。
- 非空数据通过 Zero-Copy 存入临时缓冲区。

1. **栅栏触发 (Barrier Trigger)**：

- 当 Count == Expected Shards 时，触发栅栏。
- **数据恢复**：系统执行 Concat 操作将碎片拼接。
- **执行下游**：唤醒计算图的下一个节点。

## **4. 数据恢复与一致性 (Restoration & Consistency)**

针对 Sparse 数据流（如 MoE 处理完毕后的数据），系统必须保证能将其精确还原回原本的 Dense 形状，以便进行后续的 Transformer 层计算。

### **4.1 SparseTensor 设计**

Sparse Layout 的中间产物不仅仅是 Tensor Data，它必须是一个 Tuple：(Data, Metadata)。

```python
@dataclass
class SparseTensor:
    data: Tensor          # 实际数据 (Compact Buffer)
    original_indices: Tensor # 对应的全局坐标 (用于 Sparse->Dense 恢复)
    source_rank_info: Optional[Tensor] = None # 用于 Backward 路由

```

- 在 Sparse 传输过程中，RPC 数据包不仅携带数据本身（Data Tensor），还强制携带 元数据（Metadata）：
    - **Global Indices**：数据在原始全局视图中的索引位置。
    - **Source Rank Info**：(可选) 用于反向传播（Backward）时的梯度路由。**如果数据来自于StaticSharding可以从original_indices中通过Dense Partition OP重算得到。如果来自于DynamicSharding则使用传输换计算。**
- 还原流程 (Sparse-to-Dense为例)：

1. **Dense -> Sparse (Forward)**:

Input: Dense Tensor X.

Op: Shuffle/Dispatch.

Action: 根据 X 的内容计算 Hash。将 X[i] 发送给 Target Rank。

**关键**: 发送时，必须附带 i (Global Index) 和 src_rank。

Output (在 Target Rank): SparseTensorPayload(data=received_data, original_indices=received_indices).

1. **Sparse -> Dense (Forward / Restore)**:

Input: SparseTensorPayload.

Op: Combine/Sort.

Action: 读取 payload.original_indices。

计算该 Index 属于哪个 Dense Layout 的 Rank (通过 index // dense_block_size)。

发送数据给该 Rank。接收端收到离散的数据包后，不只是简单的拼接，而是根据携带的 Global Indices 将数据 Scatter回本地内存的正确偏移位置，从而实现无损的形状恢复。

1. **Backward Pass (Auto-diff)**:

因为我们保留了 Layout 变换的逻辑，Backward 只是 Layout 变换的**逆运算**。

Dense -> Sparse 的 Backward 是 Sparse -> Dense (梯度聚合)。

Sparse -> Dense 的 Backward 是 Dense -> Sparse (梯度分发)。

**路由**: 前向传播时记录的 source_rank_info 在反向传播时变为 dest_rank。

## **5. 性能优化原则 (Optimization Checklist)**

- **全链路零拷贝 (Zero-Copy)**：发送端使用 View/Slice 选取数据；接收端仅在 Barrier 触发时刻进行一次必要的内存整合。
- **计算下沉**：严禁 Python 参与逐元素的路由判断，完全依赖底层的 jnp.argsort, torch.gather 等原语。
- **死锁预防**：严格执行“空包保活”策略。在动态路由场景下，即使没有数据要发，也要发送信号告知接收端“我处理完了，你不需要等我了”。
- **静态拓扑缓存**：InternalLayout 和依赖关系表在编译期生成并缓存，Runtime 仅需查表执行。