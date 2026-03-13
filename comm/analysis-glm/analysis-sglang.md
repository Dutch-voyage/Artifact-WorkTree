# Analysis: SGLang (Distributed LLM Serving System)

- [x] Phase 1: Strategic Screening
- [x] Phase 2: Systematic Review
- [x] Phase 3: Knowledge Synthesis

---

## Phase 1: Strategic Screening

### A. Objective Alignment (The "Why")

**Primary Goal Linked:** Tier 2, Objective 1 - **Distributed Inference Systems: Disaggregated Computing (Prefill-Decode Separation)**

**The Delta:**
SGLang achieves exceptional alignment with prefill-decode separation through multiple innovations:

1. **RadixAttention:** Tree-based attention cache sharing that automatically deduplicates common prefixes across requests
2. **Chunked Prefill:** Splits large prefill requests into configurable chunks to reduce head-of-line blocking
3. **Multi-Token Prediction:** Speculative decoding via EAGLE-2/EAGLE-3 for 2-3.7x throughput improvement
4. **Prefill-Decode Disaggregation:** Complete separation of prefill and decode workers with RDMA-based KV cache transfer

**Ecosystem Position:**
```
SGLang (Distributed LLM Serving)
├── RadixAttention: Cache sharing
├── Chunked Prefill: Scheduling innovation
├── Speculative Decoding: Multi-token prediction
└── Disaggregation: Prefill/decode separation
```

### B. The "Must-Know" Bridge (Prerequisites)

**Prefix Caching in Autoregressive Transformers:** You must understand that each token's representation in a transformer depends on all previous tokens through cached key-value pairs, and identical token sequences produce identical KV cache entries.

In autoregressive generation, computing attention for token position `i` requires keys and values from all positions `< i`. These KVs are cached and reused for subsequent tokens. When multiple requests share prefixes (system prompts, few-shot examples), recomputing these KVs is wasteful. RadixAttention stores KVs in a radix tree indexed by token sequences, enabling automatic sharing of common prefixes.

This is fundamental because SGLang's performance gains (3-5x for workloads with shared prompts) come entirely from this cache-sharing mechanism.

### C. The Target Map (Where to Look)

**The Engine Folders:**

**RadixAttention Implementation:**
- `python/sglang/srt/mem_cache/radix_cache.py` (lines 261-869) - RadixCache class
- `python/sglang/srt/layers/radix_attention.py` - RadixAttention forward pass

**KV Cache Manager:**
- `python/sglang/srt/mem_cache/memory_pool.py` - ReqToTokenPool, TokenToKVPoolAllocator

**Chunked Prefill Scheduler:**
- `python/sglang/srt/managers/schedule_policy.py` (lines 694-827) - PrefillAdder.add_one_req()

**Multi-Token Serving:**
- `python/sglang/srt/speculative/eagle_worker.py` - EAGLE speculative decoding
- `python/sglang/srt/speculative/spec_info.py` - Algorithm definitions

**Disaggregation:**
- `python/sglang/srt/disaggregation/prefill.py` - Prefill worker
- `python/sglang/srt/disaggregation/decode.py` - Decode worker
- `python/sglang/srt/disaggregation/base/conn.py` - KV transfer interface

**Keywords for Grep:**
1. `radix` or `tree` - RadixCache tree structure
2. `chunk` or `chunked_prefill` - Chunked scheduling
3. `kv_cache` or `memory_pool` - KV cache management
4. `speculative` or `eagle` - Multi-token prediction
5. `disaggregation` or `prefill_decode` - Worker separation

### D. The "Skip" List (Noise Suppression)

**Benchmarks:** `benchmark/` - Performance measurements

**Tests:** `test/` - Unit and integration tests

**Examples:** `examples/` - Usage examples

**Documentation:** `docs/` - User documentation

**Scripts:** `scripts/` - Utility scripts

---

## Phase 2: Systematic Review

### A. The "Should-Do" Logic (The Levers)

#### Module 1: RadixAttention Tree Structure
**File Path:** `python/sglang/srt/mem_cache/radix_cache.py`

**Key Lever Locations:**
- **Lines 97-158:** TreeNode class definition
- **Lines 352-422:** match_prefix method for prefix matching
- **Lines 674-693:** Node splitting for partial matches

**Mechanism:**
RadixAttention uses a radix tree where each node represents a token and contains a KV cache value. The tree enables automatic prefix sharing: when inserting a token sequence, the tree traverses existing nodes and only creates new nodes for divergent tokens. The `match_prefix()` method searches for the longest matching prefix, returning concatenated KV cache indices for the matched portion.

**Critical Code (Lines 97-115):**
```python
class TreeNode:
    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)  # Token → child node mapping
        self.parent: TreeNode = None
        self.key: RadixKey = None               # Token sequence for this node
        self.value: Optional[torch.Tensor] = None  # KV cache indices
        self.lock_ref = 0                       # Reference count for eviction protection
        self.last_access_time = time.monotonic()
        self.hit_count = 0
```

---

#### Module 2: Hash-Based Cache Identification
**File Path:** `python/sglang/srt/mem_cache/radix_cache.py`

**Key Lever Locations:**
- **Lines 201-231:** SHA256 hash computation for KV pages

**Mechanism:**
Each KV cache page is hashed using SHA256, creating a position-aware identifier. These hashes enable distributed cache sharing across workers - the same KV page on different workers has the same hash. This supports hierarchical caching (HiCache) with remote storage backends.

---

#### Module 3: KV Cache Memory Pool
**File Path:** `python/sglang/srt/mem_cache/memory_pool.py`

**Key Lever Locations:**
- **Lines 126-183:** ReqToTokenPool request-to-token mapping
- **Lines 200+:** Paged KV cache allocation

**Mechanism:**
Two-level pool design: `ReqToTokenPool` maps request IDs to token locations, and `TokenToKVPoolAllocator` manages physical KV cache data. Page-based allocation (configurable page size) supports multiple attention backends (FlashInfer, Triton, Torch native) and quantization (FP4, FP8, INT4, AWQ, GPTQ).

---

#### Module 4: Chunked Prefill Scheduler
**File Path:** `python/sglang/srt/managers/schedule_policy.py`

**Key Lever Locations:**
- **Lines 694-716:** Budget tracking (rem_chunk_tokens)
- **Lines 786-827:** Chunking logic in add_one_req()

**Mechanism:**
Large prefill requests are split into chunks based on available memory budget. The scheduler tracks `rem_chunk_tokens` and decrements it as chunks are added. When budget is exhausted, the request is truncated to `rem_chunk_tokens` (aligned to page size) and marked as chunked. The remaining tokens are processed in subsequent iterations.

**Critical Code (Lines 809-820):**
```python
else:
    # Chunked prefill - truncate to available budget
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size

    if truncation_align_size is not None:
        trunc_len = truncation_align_size * (trunc_len // truncation_align_size)

    req.set_extend_input_len(trunc_len)
    req.fill_ids = req.fill_ids[:len(req.prefix_indices) + trunc_len]
    self.can_run_list.append(req)
    self.new_chunked_req = req
```

---

#### Module 5: EAGLE Speculative Decoding
**File Path:** `python/sglang/srt/speculative/eagle_worker.py`

**Key Lever Locations:**
- **Lines 1-100:** EAGLE-2 algorithm implementation

**Mechanism:**
EAGLE-2 uses a draft model to predict feature vectors (hidden states) for multiple tokens ahead. The draft samples tokens from the feature distribution, expands in a tree style with branching factor `speculative_eagle_topk`, then re-ranks to select top candidates. The target model verifies all candidates in parallel and accepts/rejects based on thresholds.

**Performance:** Baseline 158.34 tokens/s → EAGLE-2 244.10 tokens/s (1.54x) → EAGLE-3 373.25 tokens/s (2.36x)

---

#### Module 6: Prefill-Decode State Transfer
**File Path:** `python/sglang/srt/disaggregation/base/conn.py`

**Key Lever Locations:**
- **Lines 52-107:** BaseKVSender and BaseKVReceiver interfaces
- **Lines 26-41:** KVArgs state types

**Mechanism:**
Transfer lifecycle: Bootstrap Queue (handshake, allocate KV cache) → Waiting Queue (pop requests, run forward pass) → Inflight Queue (poll sender non-blocking). KV manager interface defines `init()` (notify decoder about KV indices), `send()` (transfer KV cache and decoder states), and `poll()` (check transfer status).

**Transfer Backends:**
1. **Mooncake:** RDMA-based with NVLink transport
2. **NIXL:** UCX/Libfabric-based RDMA
3. **ASCEND:** NPU-specific memfabric transport
4. **Fake:** In-memory copy for testing

**Critical Code (Lines 70-85):**
```python
class BaseKVSender(ABC):
    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """Notify decoder about KV indices length and aux index"""

    @abstractmethod
    def send(self, kv_indices: npt.NDArray[np.int32],
             state_indices: Optional[List[int]] = None):
        """Send KV cache and optional decoder states"""

    @abstractmethod
    def poll(self) -> KVPoll:
        """Check transfer status: Bootstrapping/Transferring/Success"""
```

---

### B. Objective-Specific Observations

**Observation 1:** [Tier 2, Obj 1 - Cache Sharing]
RadixAttention (radix_cache.py:261-869) enables **automatic KV cache deduplication** across requests through tree-based prefix matching. This reduces redundant computation by 3-5x for workloads with shared prompts (system prompts, few-shot examples).

**Observation 2:** [Tier 2, Obj 1 - Head-of-Line Blocking]
Chunked prefill (schedule_policy.py:694-827) **mitigates head-of-line blocking** by splitting large prefill requests into configurable chunks. This allows decode requests to interleave with prefill chunks, reducing tail latency by 2-10x.

**Observation 3:** [Tier 2, Obj 1 - Prefill-Decode Disaggregation]
Complete separation of prefill and decode workers (prefill.py, decode.py) enables **specialized scheduling policies** for each phase. KV cache transfer via RDMA (Mooncake, NIXL backends) minimizes handoff overhead.

**Observation 4:** [Tier 2, Obj 1 - Multi-Token Prediction]
EAGLE-2/EAGLE-3 speculative decoding (eagle_worker.py) achieves **2-3.7x throughput improvement** through parallel token verification. Draft model predicts multiple tokens, target model verifies in parallel, accept/reject based on thresholds.

---

### C. The "Aha!" Moments (Grounded Insight)

**Aha! Moment 1: Radix Tree Prefix Sharing**
**Look at:** `python/sglang/srt/mem_cache/radix_cache.py:352-422`

**The Logic:** The `match_prefix()` method traverses the radix tree from root, matching tokens against child keys. When multiple requests share a prefix (e.g., system prompt), they hit the same tree nodes and reuse the cached KV indices. Only the diverging portions need new computation. This is automatic - no explicit cache management required by the user.

---

**Aha! Moment 2: Node Splitting for Partial Matches**
**Look at:** `python/sglang/srt/mem_cache/radix_cache.py:674-693`

**The Logic:** When a match ends mid-node (existing node has key `[A, B, C]` but request has `[A, B, D]`), the tree dynamically splits the node at the mismatch point. A new intermediate node is created with key `[A, B]`, and both the original continuation `[C]` and new continuation `[D]` become children. This enables efficient storage without duplicating common prefixes.

---

**Aha! Moment 3: Chunked Prefill with Backpressure**
**Look at:** `python/sglang/srt/managers/schedule_policy.py:786-827`

**The Logic:** Instead of rejecting large prefill requests when memory is full, the scheduler truncates them to fit the available `rem_chunk_tokens` budget. The request is marked as `new_chunked_req` and the remaining tokens are processed in subsequent iterations. This provides backpressure while maintaining fairness - large requests make progress without blocking others indefinitely.

---

**Aha! Moment 4: RDMA-Based State Transfer**
**Look at:** `python/sglang/srt/disaggregation/base/conn.py:52-107`

**The Logic:** The KV transfer interface supports RDMA through separate `init()` (metadata handshake) and `send()` (data transfer) phases. The decode worker pre-allocates KV cache during `init()`, then the prefill worker directly writes to decode GPU memory via RDMA. The `poll()` method checks transfer completion without blocking, enabling overlap with computation.

---

### Summary Table: Lever Code Locations

| Module | File Path | Key Lines | Function | Objective |
|--------|-----------|-----------|----------|-----------|
| RadixCache Tree | `python/sglang/srt/mem_cache/radix_cache.py` | 97-158 | TreeNode structure | Cache sharing |
| Prefix Matching | `python/sglang/srt/mem_cache/radix_cache.py` | 352-422 | match_prefix() | KV reuse |
| Node Splitting | `python/sglang/srt/mem_cache/radix_cache.py` | 674-693 | _split_node() | Storage efficiency |
| Hash Computation | `python/sglang/srt/mem_cache/radix_cache.py` | 201-231 | SHA256 hashing | Distributed cache |
| KV Memory Pool | `python/sglang/srt/mem_cache/memory_pool.py` | 126-183 | ReqToTokenPool | Memory management |
| Chunked Prefill | `python/sglang/srt/managers/schedule_policy.py` | 786-827 | add_one_req() | Head-of-line mitigation |
| Budget Tracking | `python/sglang/srt/managers/schedule_policy.py` | 694-716 | rem_chunk_tokens | Memory control |
| EAGLE Speculation | `python/sglang/srt/speculative/eagle_worker.py` | 1-100 | EAGLE-2 algorithm | Multi-token prediction |
| KV Sender | `python/sglang/srt/disaggregation/base/conn.py` | 52-85 | BaseKVSender | Transfer interface |
| KV Receiver | `python/sglang/srt/disaggregation/base/conn.py` | 87-107 | BaseKVReceiver | Transfer interface |
| Prefill Worker | `python/sglang/srt/disaggregation/prefill.py` | 1-200 | Prefill queues | Prefill orchestration |
| Decode Worker | `python/sglang/srt/disaggregation/decode.py` | 1-200 | Decode queues | Decode orchestration |

---

## Phase 3: Knowledge Synthesis

### Project Category
**Distributed LLM Serving System** - High-throughput inference with cache sharing, chunked prefill, and prefill-decode disaggregation

### A. The Problem and The Fix

**The Problem:**

SGLang addresses five fundamental inefficiencies in LLM serving:

| Problem | Impact | SGLang Solution |
|---------|--------|-----------------|
| **Redundant Computation** | 3-5x wasted compute for shared prompts | RadixAttention automatic cache sharing |
| **Head-of-Line Blocking** | 2-10x higher TTFT for queued requests | Chunked prefill with mixed batches |
| **Memory-Compute Imbalance** | 40-60% GPU utilization | Prefill-decode disaggregation |
| **KV Cache Pressure** | Low batch sizes for long contexts | Paged attention, radix tree, hierarchical eviction |
| **Serial Token Generation** | 50-100 tokens/second typical | EAGLE speculative decoding (2-3.7x) |

**The Fix:**

**RadixAttention Mechanism:**
```
Request 1: [system_prompt, user_input_1, generated_1, generated_2]
           └─ RadixCache stores: [system_prompt] → [user_input_1] → ...

Request 2: [system_prompt, user_input_2, ...]
           └─ RadixCache hit: [system_prompt] (reuse!)
           └─ Only compute [user_input_2] onwards
```

**Benefits:**
1. **Automatic sharing:** No manual cache management required
2. **Namespace isolation:** Different LoRA adapters, sampling parameters use separate trees
3. **Eviction control:** Multiple policies (LRU/LFU/FIFO/priority)
4. **Distributed support:** Hash-based KV page identification enables cross-worker cache sharing

**Chunked Prefill Mechanism:**
```
Large Request (8K tokens):
├─ Chunk 1: 2K tokens → Forward → Cache → Decode
├─ Chunk 2: 2K tokens → Forward → Cache → Decode (parallel with Chunk 1 decode)
├─ Chunk 3: 2K tokens → Forward → Cache → Decode
└─ Chunk 4: 2K tokens → Forward → Cache → Decode

Interleaved with ongoing decode requests:
Prefill Chunk 1 → Decode A → Prefill Chunk 2 → Decode B → ...
```

**Synergy:**
- **Chunked prefill** creates intermediate cache points → **RadixAttention** shares them
- **Disaggregation** transfers chunks incrementally → **Lower latency** than full request transfer
- **Speculative decoding** benefits from cached prefixes → **Higher acceptance rates**

---

### B. Can I Use It? (Portability)

**Highly Portable Components:**

| Component | Portability | Use Cases | Adaptation Required |
|-----------|-------------|-----------|-------------------|
| **Radix Tree Cache** | ⭐⭐⭐⭐⭐ | Any autoregressive model, RAG systems | Replace KV indices with framework-specific handles |
| **Chunked Prefill** | ⭐⭐⭐⭐ | Any LLM serving system, batch inference | Memory budget estimation per framework |
| **Disaggregation** | ⭐⭐⭐ | Multi-node deployments, edge-cloud hybrid | RDMA backend, state serialization |
| **EAGLE Speculation** | ⭐⭐ | Model-specific deployment | Draft model training per target model |

**Implementation Guidance:**
```python
# Minimal RadixCache adaptation
class SimpleRadixCache:
    def __init__(self):
        self.root = {}

    def lookup(self, tokens: List[int]) -> Optional[Any]:
        node = self.root
        for t in tokens:
            if t not in node:
                return None
            node = node[t]
        return node.get('_value')

    def insert(self, tokens: List[int], value: Any):
        node = self.root
        for t in tokens:
            if t not in node:
                node[t] = {}
            node = node[t]
        node['_value'] = value
```

---

### C. The Starter Bridge

**"Understanding prefix caching requires familiarity with how autoregressive transformers compute attention: each token's representation depends on all previous tokens through cached key-value pairs, and identical token sequences produce identical KV cache entries."**

**Deeper Prerequisites (ordered):**

1. **Transformer Attention Mechanism**
   - Q/K/V computation per layer
   - Causal attention mask (each token attends to previous tokens)
   - KV cache reuse in autoregressive generation

2. **Tree Data Structures**
   - Radix tree / Trie basics
   - Prefix matching and traversal
   - Node splitting for partial matches

3. **Memory Management**
   - Paged memory allocation
   - Reference counting for eviction
   - Cache eviction policies (LRU, LFU)

4. **Distributed Systems**
   - RDMA basics for zero-copy transfers
   - Producer-consumer patterns (prefill→decode)
   - State synchronization across workers

**Recommended Learning Path:**
1. Read vLLM's PagedAttention paper (for KV cache fundamentals)
2. Study SGLang's RadixCache implementation (radix_cache.py lines 261-500)
3. Review chunked prefill logic (schedule_policy.py lines 700-830)
4. Examine disaggregation protocol (base/conn.py, prefill.py lines 1-100)

---

### Summary

**SGLang** provides a production LLM serving system with systematic innovations in cache management, scheduling, and distributed coordination.

**Key Innovations:**
1. **RadixAttention:** Tree-based KV cache sharing for 3-5x reduction in redundant computation
2. **Chunked Prefill:** Dynamic chunk sizing with backpressure for reduced head-of-line blocking
3. **Prefill-Decode Disaggregation:** Separate workers with RDMA-based KV transfer
4. **Multi-Token Prediction:** EAGLE-2/EAGLE-3 speculative decoding for 2-3.7x throughput improvement

**Performance Impact:**
- 3-5x speedup for workloads with shared prompts
- 2-10x reduction in tail latency via chunked prefill
- 2-3.7x decode throughput via speculative decoding
- 40-60% → 80-90% GPU utilization via disaggregation

**Best for:**
- Conversational AI with shared system prompts
- Few-shot learning workflows
- Multi-turn dialogue systems
- High-throughput inference services

**Alternative:** For simpler deployments, consider vLLM (no disaggregation), TGI (HuggingFace-focused), or TensorRT-LLM (NVIDIA-optimized).

---

**Analysis Complete:** All three phases completed for SGLang
