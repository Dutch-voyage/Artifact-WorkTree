# Analysis: Triton-distributed

- [x] Phase 1: Strategic Screening
- [x] Phase 2: Systematic Review
- [x] Phase 3: Knowledge Synthesis

---

## Phase 1: Strategic Screening

### A. Objective Alignment (The "Why")

**Primary Goal Linked:** Tier 1, Objective 2 - **Collective Optimization: All-to-All, All-Reduce**

**The Delta:**
Triton-distributed provides **kernel-fused communication primitives** that fundamentally differ from standard NCCL approaches:

1. **NCCL Approach:** Two-sided communication requiring CPU synchronization, separate communication kernels, and explicit kernel launches for computation
2. **Triton-Distributed Approach:** One-sided GPU-initiated communication (via OpenSHMEM/NVSHMEM) that can be fused directly into compute kernels

**Key Differentiators:**
- Kernel-level communication-computation overlap (not just operation-level)
- GPU-driven synchronization without CPU intervention
- Symmetric heap allocation for direct remote memory access
- Multiple AllReduce algorithms: DoubleTree, OneShot, TwoShot, OneShot_Multimem, TwoShot_Multimem

### B. The "Must-Know" Bridge (Prerequisites)

**PGAS (Partitioned Global Address Space):** You must understand how symmetric heap allocation enables direct remote memory access.

In traditional MPI/NCCL, each GPU has its own private address space. To send data, you explicitly call send/recv functions that copy data between address spaces.

In PGAS (NVSHMEM/ROC_SHMEM), all GPUs allocate memory from a "symmetric heap" where the same virtual address on each GPU maps to different physical memory. Using `nvshmem_ptr(local_addr, remote_pe)`, any GPU can translate a local virtual address to a remote physical address and perform one-sided PUT/GET operations without the remote GPU's active participation.

This is fundamental because Triton-distributed's kernel-fused communication relies entirely on one-sided operations where a GPU kernel writes directly to remote memory while continuing computation.

### C. The Target Map (Where to Look)

**The Engine Folders:**

**Primary - Distributed Primitives:**
- `python/triton_dist/language/extra/cuda/libnvshmem_device.py` - NVSHMEM device function bindings (992 lines)
- `python/triton_dist/language/distributed_ops.py` - High-level distributed language ops
- `python/triton_dist/kernels/nvidia/allreduce.py` - AllReduce implementations (6 algorithms)
- `python/triton_dist/kernels/nvidia/low_latency_all_to_all.py` - All-to-All for MoE routing

**Fused Kernels:**
- `python/triton_dist/kernels/nvidia/gemm_allreduce.py` - Fused GEMM+AllReduce
- `python/triton_dist/kernels/nvidia/allgather_gemm.py` - Fused AllGather+GEMM
- `python/triton_dist/kernels/nvidia/gemm_reduce_scatter.py` - Fused GEMM+ReduceScatter

**Compiler Infrastructure:**
- `lib/Dialect/Distributed/IR/` - MLIR dialect definitions
- `lib/Conversion/TritonDistributedToLLVM/` - LLVM conversion passes

**Keywords for Grep:**
1. `shmem` or `nvshmem` - SHMEM backend integration (249 files)
2. `all_reduce` - Collective implementations (40+ files)
3. `all_to_all` - MoE routing kernels (15+ files)
4. `symm_at` or `symmetric` - Symmetric heap operations
5. `signal_wait` or `barrier` - GPU-side synchronization

### D. The "Skip" List (Noise Suppression)

**Tutorials:** `tutorials/*.py` - Reference examples only

**Tests:** `python/triton_dist/test/` - Validation code

**Bindings Infrastructure:** `shmem/*/` - Plumbing code for SHMEM libraries

**Build System:** `CMakeLists.txt`, build scripts - Not algorithmic

**Documentation:** `docs/` - Usage documentation

---

## Phase 2: Systematic Review

### A. The "Should-Do" Logic (The Levers)

#### Module 1: NVSHMEM Integration & Initialization
**File Path:** `python/triton_dist/utils.py`

**Key Lever Locations:**
- **Lines 316-360:** `initialize_distributed` - NVSHMEM initialization with PyTorch process group
- **Lines 246-256:** `nvshmem_create_tensor` - Symmetric heap allocation

**Mechanism:**
NVSHMEM is initialized with a PyTorch process group (`nvshmem.core.init_pg(pg)`), then symmetric tensors are allocated via `nvshmem.core.tensor()`. This creates a symmetric heap where the same virtual address maps to physical memory on each GPU, enabling direct pointer-based remote access via `nvshmem_ptr()`.

---

#### Module 2: AllReduce Kernel Implementations
**File Path:** `python/triton_dist/kernels/nvidia/allreduce.py`

**Key Lever Locations:**
- **Lines 1-60:** Available AllReduce methods (DoubleTree, OneShot, TwoShot, OneShot_Multimem, TwoShot_Multimem)
- **Lines 216-332:** `allreduce_double_tree_intra_node_kernel` - DoubleTree reduction implementation

**Mechanism:**
Multiple AllReduce algorithms optimized for different scenarios:
- **DoubleTree:** Two complementary binary trees for log2(N) reduction depth
- **OneShot:** Single-pass aggregation using symmetric buffer (requires N * workspace)
- **TwoShot:** Two-pass with scatter + gather phases
- **OneShot_Multimem:** Uses H100 multicast load for simultaneous reads

**Critical Code (DoubleTree Kernel - Lines 280-295):**
```python
# Reduce phase: aggregate from children
if has_child0:
    libshmem_device.putmem_nbi_block(child0_buf, local_buf, size, child0_rank)

# Propagate phase: send result to parent
if has_parent:
    libshmem_device.putmem_signal_nbi_block(parent_buf, local_buf, size, signal_ptr, ...)
    libshmem_device.signal_wait_until(signal_ptr, NVSHMEM_CMP_EQ, expected_value)
```

---

#### Module 3: All-to-All for MoE Routing
**File Path:** `python/triton_dist/kernels/nvidia/low_latency_all_to_all.py`

**Key Lever Locations:**
- **Lines 35-119:** `all_to_all_kernel` - MoE token routing implementation

**Mechanism:**
The kernel calculates token ranges for each expert block, performs non-blocking PUT to destination ranks, and signals completion. Performance: **137us on 32xH800** (128 tokens, topk=8, hidden=7168, fp8) vs DeepEP 182us.

**Critical Code (Lines 95-105):**
```python
# Non-blocking put to destination rank
libshmem_device.putmem_nbi_block(
    data_dst_ptr, data_src + src_off * HIDDEN,
    num_rows_cur_block * HIDDEN * ELEMENT_SIZE, pid
)

# Signal completion
libshmem_device.fence()
if threadidx == 0:
    libshmem_device.signal_wait_until(signal_ptr, NVSHMEM_CMP_EQ, call_count)
```

---

#### Module 4: One-Sided Communication Primitives
**File Path:** `python/triton_dist/language/extra/cuda/libnvshmem_device.py`

**Key Lever Locations:**
- **Lines 463-464:** `putmem_nbi_block` - Non-blocking block-strided PUT
- **Lines 543-545:** `putmem_signal_nbi_block` - PUT with automatic signal operation

**File Path:** `python/triton_dist/language/distributed_ops.py`

**Key Lever Locations:**
- **Lines 1-112:** Symmetric address operations (`symm_at`), notification (`notify`), waiting (`wait`)

**Mechanism:**
One-sided PUT operations initiate transfer from the current thread block to remote PE and return immediately. The `putmem_signal_nbi_block` variant atomically updates a signal after data transfer completes, eliminating separate signaling round-trips. Symmetric addressing (`symm_at(ptr, rank)`) converts local pointers to remote addresses.

---

#### Module 5: GPU-Side Barrier Synchronization
**File Path:** `python/triton_dist/kernels/nvidia/common_ops.py`

**Key Lever Locations:**
- **Lines 52-57:** `_is_gpu_master` - Master CTA identification
- **Lines 60-86:** `unsafe_barrier_on_this_grid` - Block-level atomic barrier
- **Lines 109-135:** GPU-level synchronization using CUDA Cooperative Groups ABI

**Mechanism:**
Multi-level barriers enable fine-grained synchronization:
- **Warp-level:** Thread vote operations
- **Block-level:** Atomic add on shared memory with high-bit signaling
- **Grid-level:** Uses CUDA Cooperative Groups ABI via environment registers (`%envreg1/2`)

**Critical Code (Lines 72-82):**
```python
# Master CTA sets high bit
if _is_cta_master():
    nb = tl.where(_is_gpu_master(),
                 tl.cast(0x80000000, tl.uint32) - (expected - 1), 1)
    old_arrive = atomic_add(ptr, nb, scope="gpu", semantic="release")

# Spin-wait for high bit to be set
if _is_cta_master():
    current_arrive = ld_acquire(ptr)
    while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
        current_arrive = ld_acquire(ptr, scope="gpu")
```

---

### B. Objective-Specific Observations

**Observation 1:** [Tier 1, Obj 2 - Collective Optimization]
Triton-distributed implements **six different AllReduce algorithms** (allreduce.py:1-60), each optimized for different message sizes and topologies. The DoubleTree algorithm achieves log2(N) reduction depth using two complementary binary trees, while OneShot_Multimem leverages H100's multicast load for simultaneous reads from multiple PEs.

**Observation 2:** [Tier 1, Obj 2 - All-to-All for MoE]
The All-to-All kernel (low_latency_all_to_all.py:35-119) achieves **137us latency on 32xH800** for MoE token routing, outperforming DeepEP's 182us. This is accomplished by fusing the token-to-expert assignment calculation directly with the non-blocking PUT operations, eliminating intermediate staging buffers.

**Observation 3:** [Tier 1, Obj 2 - Kernel Fusion]
Fused kernels like `gemm_allreduce` and `allgather_gemm` demonstrate **1.3-1.5x speedup** by overlapping communication with computation. Instead of launching separate GEMM → AllReduce kernels, the fused kernel initiates AllReduce for the previous tile while computing the next tile (allreduce.py:280-295).

**Observation 4:** [Tier 2, Obj 2 - Modern C++ Concurrency]
The MLIR dialect (`lib/Dialect/Distributed/IR/`) and LLVM conversion passes (`lib/Conversion/TritonDistributedToLLVM/`) implement a **compiler-based approach** to distributed primitives. The `WaitOpConversion` (DistributedOpToLLVM.cpp:156-242) generates PTX spin-wait loops with configurable memory semantics, demonstrating instruction-level parallelism.

---

### C. The "Aha!" Moments (Grounded Insight)

**Aha! Moment 1: Symmetric Heap Magic**
**Look at:** `python/triton_dist/utils.py:246-256`

**The Logic:** When `nvshmem.core.tensor()` is called, NVSHMEM allocates memory from a "symmetric heap" where each GPU allocates at the **same virtual address** but points to **different physical memory**. Using `nvshmem_ptr(local_addr, remote_pe)`, any GPU can translate its local virtual address to a remote GPU's physical address. This enables direct pointer-based remote access - GPU A can write to `*(nvshmem_ptr(my_buf, rank_B))` and the data appears on GPU B without any explicit receive operation.

---

**Aha! Moment 2: GPU-Initiated Barrier Without CPU**
**Look at:** `python/triton_dist/kernels/nvidia/common_ops.py:60-86`

**The Logic:** This barrier implements the classic "tournament barrier" pattern entirely in GPU code. The master CTA atomically adds a value with the high bit set (0x80000000), then all CTAs spin-wait on that high bit. No CPU involvement, no `cudaDeviceSynchronize()`, no `torch.cuda.synchronize()`. The GPU manages its own synchronization, enabling kernels to launch collective operations independently.

---

**Aha! Moment 3: DoubleTree Reduction Topology**
**Look at:** `python/triton_dist/kernels/nvidia/allreduce.py:216-332`

**The Logic:** DoubleTree uses **two complementary binary trees** to reduce contention. In Tree 0, each node reduces from children at positions `(2*i, 2*i+1)`. In Tree 1, the pattern is reversed. This halves the reduction depth compared to a single tree and provides better load balancing for irregular topologies. The two-phase approach (reduce from children → propagate to parent) enables perfect pipelining.

---

**Aha! Moment 4: Signal-Op Fusion**
**Look at:** `python/triton_dist/language/extra/cuda/libnvshmem_device.py:543-545`

**The Logic:** The `putmem_signal_nbi_block` function fuses a PUT operation with a signal update in a single NVSHMEM call. After the data transfer completes, NVSHMEM atomically performs the signal operation (increment, set, or bitwise OR). This eliminates the separate "signal after put" round-trip that would otherwise be required, reducing latency by one network RTT.

---

### Summary Table: Lever Code Locations

| Module | File Path | Key Lines | Function | Objective |
|--------|-----------|-----------|----------|-----------|
| NVSHMEM Init | `python/triton_dist/utils.py` | 316-360 | `initialize_distributed` | PGAS setup |
| Symmetric Alloc | `python/triton_dist/utils.py` | 246-256 | `nvshmem_create_tensor` | Shared address space |
| DoubleTree AllReduce | `python/triton_dist/kernels/nvidia/allreduce.py` | 216-332 | `allreduce_double_tree_kernel` | Collective optimization |
| All-to-All MoE | `python/triton_dist/kernels/nvidia/low_latency_all_to_all.py` | 35-119 | `all_to_all_kernel` | MoE routing |
| One-Sided PUT | `python/triton_dist/language/extra/cuda/libnvshmem_device.py` | 463-464 | `putmem_nbi_block` | Remote write |
| Signal Fusion | `python/triton_dist/language/extra/cuda/libnvshmem_device.py` | 543-545 | `putmem_signal_nbi_block` | Op fusion |
| GPU Barrier | `python/triton_dist/kernels/nvidia/common_ops.py` | 60-86 | `unsafe_barrier_on_this_grid` | GPU sync |
| Symmetric Address | `python/triton_dist/language/distributed_ops.py` | 1-112 | `symm_at` | Remote addressing |

---

## Phase 3: Knowledge Synthesis

### Project Category
**Distributed GPU Computing Framework** - Compiler-based approach to kernel-fused communication primitives

### A. The Problem and The Fix

**The Problem:**

In distributed training and inference, communication operations (AllReduce, AllGather, All-to-All) incur significant overhead:

1. **Kernel Launch Latency:** Separate communication kernels require CPU scheduling (~10-50μs per launch)
2. **Synchronization Overhead:** NCCL requires CPU-side barriers between operations
3. **Memory Copies:** Data movement between compute and communication buffers
4. **Underutilized Hardware:** GPUs idle during communication phases

**Example - Traditional NCCL (GEMM + AllReduce):**
```
1. Launch GEMM kernel (10ms)
2. CPU sync + launch AllReduce (5ms)  # GPU idle
3. CPU sync + launch next layer
Total: 15ms (33% overhead)
```

**The Fix:**

Triton-distributed enables **kernel-fused communication** where NVSHMEM/OpenSHMEM operations are embedded directly in compute kernels:

```python
@triton_dist.jit
def gemm_allreduce_kernel(...):
    # Producer: Compute GEMM tiles
    for tile_id in range(num_tiles):
        C_tile = matmul(A_tile, B_tile)

        # Consumer: Communicate while computing next tile
        if tile_id > 0:
            libshmem_device.putmem_nbi_block(remote_buf, prev_tile, ...)

        prev_tile = C_tile

    # Finalize last tile
    libshmem_device.putmem_signal_nbi_block(remote_buf, last_tile, signal, ...)
```

**Benefits:**
- **Zero kernel launch overhead:** Communication initiated from GPU within compute kernel
- **Perfect overlap:** Compute next tile while communicating previous tile
- **No CPU synchronization:** GPU-managed barriers via `signal_wait_until`
- **Reduced memory movement:** Direct remote PUT eliminates staging buffers

**Performance Impact (from docs/e2e.md):**
- MLP (M=2048): 0.6587s → 0.4930s (**1.34x speedup**)
- Attn Prefill: 0.1274s → 0.0862s (**1.48x speedup**)
- Attn Decode: 0.1367s → 0.0981s (**1.39x speedup**)

---

### B. Can I Use It? (Portability)

**How hard is it to move?** **MEDIUM**

**Hardware Dependencies:**

| Platform | SHMEM Implementation | Supported Hardware |
|----------|---------------------|-------------------|
| NVIDIA | NVSHMEM | H100, H800, A100, L20, L40 |
| AMD | ROC_SHMEM | MI300X |
| MACA | MXSHMEM | MACA-compatible GPUs |

**Software Dependencies:**
- Triton language (>= 2.1)
- NVSHMEM/ROC_SHMEM libraries
- PyTorch (for process group management)
- LLVM/MLIR toolchain

**What can be extracted:**

| Concept | Portability | Effort |
|---------|-------------|--------|
| DoubleTree algorithm | High | Low - Tree topology is platform-agnostic |
| OneShot/TwoShot | High | Medium - Requires symmetric heap |
| All-to-All kernel | High | Medium - Token routing logic is reusable |
| MLIR dialect | Low | High - Requires MLIR expertise |
| NVSHMEM bindings | Low | High - NVIDIA-specific |

**Recommended Approach:**
1. **Use as Library:** Import `triton_dist` kernels directly in your Triton code
2. **Study Algorithms:** Implement DoubleTree/OneShot in other frameworks (CUDA, HIP)
3. **Adopt Patterns:** Use one-sided communication + GPU barriers pattern regardless of SHMEM library
4. **Compiler Integration:** If using Triton, leverage the MLIR dialect for custom distributed ops

---

### C. The Starter Bridge

**"Understanding PGAS (Partitioned Global Address Space) and how symmetric heap allocation enables direct remote memory access is the foundation for kernel-fused communication."**

**Key Prerequisite Concepts:**

1. **Symmetric Heap:** Each GPU allocates at the same virtual address → different physical memory
2. **Remote Pointer:** `nvshmem_ptr(local_addr, remote_pe)` translates to remote physical address
3. **One-Sided PUT:** GPU writes directly to remote address without receiver involvement
4. **GPU Barrier:** Atomic operations on shared memory for SM synchronization
5. **Kernel Fusion:** Communication calls embedded in compute kernel (not separate launches)

**Learning Path:**
1. Read `python/triton_dist/utils.py:246-256` to understand symmetric tensor allocation
2. Read `python/triton_dist/language/distributed_ops.py:1-112` to see high-level distributed ops
3. Read `python/triton_dist/kernels/nvidia/allreduce.py:216-332` to see DoubleTree algorithm
4. Experiment with NVSHMEM tutorial programs to understand PGAS programming model
5. Modify a simple Triton kernel to add one-sided PUT operations

**Key Insight:** Triton-distributed moves communication from "operation-level" (separate kernels) to "instruction-level" (fused in compute), enabled by the PGAS programming model where any GPU can directly access any other GPU's memory.

---

### Summary

**Triton-distributed** provides a compiler-based approach to distributed GPU computing that fuses communication directly into compute kernels via OpenSHMEM/NVSHMEM primitives.

**Key Differentiators vs NCCL:**
1. GPU-initiated one-sided communication (no CPU synchronization)
2. Kernel-level compute-communication overlap
3. Multiple optimized collective algorithms (DoubleTree, OneShot, etc.)
4. Cross-platform support (NVIDIA, AMD, MACA)

**Performance:** 1.3-1.5x speedup on tensor parallel workloads (MLP, Attention) by eliminating kernel launch and synchronization overhead.

**Best for:** Distributed training/inference workloads with heavy communication patterns (tensor parallelism, MoE routing) on modern GPU clusters (H100/A100) with NVSHMEM support.

**Alternative:** If NVSHMEM is unavailable, consider NCCL for standard collectives or custom CUDA kernels with GPUDirect RDMA.

---

**Analysis Complete:** All three phases completed for Triton-distributed
