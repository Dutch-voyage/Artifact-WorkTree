# Analysis: vllm
- [x] Phase 1: Strategic Screening
- [ ] Phase 2: Systematic Review
- [ ] Phase 3: Knowledge Synthesis

---

## Phase 1: Strategic Screening

### Project Name: `vllm`

#### A. Objective Alignment (The "Why"):
* **Primary Goal Linked:** Tier 2, Obj 1 — **Distributed Inference** (Disaggregated Computing - Prefill-Decode separation)
* **The Delta:** vLLM implements **disaggregated prefill-decode serving** via KV transfer infrastructure. It shows how to separate prefill and decode into different processes/servers and transfer KV cache between them. Also has excellent MoE support.

#### B. The "Must-Know" Bridge (Prerequisites):
* **PagedAttention:** vLLM's core innovation - understanding KV cache as pages is prerequisite to understanding disaggregation.

#### C. The Target Map (Where to look):
* **The Engine Folder:** `/Users/pobs/workspace/moe_project/vllm/vllm/distributed/kv_transfer/` and `/Users/pobs/workspace/moe_project/vllm/vllm/model_executor/layers/fused_moe/`
* **Keywords for Grep:**
  - `kv_transfer` — KV cache transfer infrastructure
  - `KVConnector` — KV connector implementations
  - `custom_all_reduce` — Custom GPU all-reduce
  - `FusedMoE` — Fused MoE layer
  - `disaggregat` — Disaggregated serving

#### D. The "Skip" List (Noise Suppression):
* **What to ignore:**
  - `vllm/docs/` — documentation
  - `vllm/tests/` — test files
  - `vllm/examples/` — example scripts
  - `vllm/benchmarks/` — benchmarks
  - `vllm/vllm/entrypoints/` — API server (not core engine)

---

## Phase 2: Systematic Review (The Deep-Dive)

### Lever Code Blocks Identified

#### 1. Custom All-Reduce for Multi-GPU
**File:** `/Users/pobs/workspace/moe_project/vllm/vllm/distributed/device_communicators/custom_all_reduce.py:51-80`

**What it does:** Implements custom all-reduce for GPU-to-GPU communication using P2P (peer-to-peer) access.

**Key mechanism (lines 51-80):**
- Checks P2P access via `gpu_p2p_access_check()`
- Supports world sizes [2, 4, 6, 8]
- Falls back to NCCL if custom all-reduce unavailable

**Why it matters:** Bypasses NCCL for intra-node GPU communication for lower latency.

#### 2. KV Transfer Infrastructure
**File:** `/Users/pobs/workspace/moe_project/vllm/vllm/distributed/kv_transfer/kv_transfer_state.py`

**What it does:** Manages KV connector for disaggregated prefill-decode.

**Key mechanism:**
- `KVConnectorBase_V1` — Base class for KV connectors
- `KVConnectorRole` — Defines prefill or decode role
- Supports connectors: MoriIO, Mooncake, LMCache

#### 3. Fused MoE Layer
**File:** `/Users/pobs/workspace/moe_project/vllm/vllm/model_executor/layers/fused_moe/layer.py:1-80`

**What it does:** Implements fused MoE with expert parallelism (EP) support.

**Key mechanism:**
- `FusedMoEMethodBase` — Base class for MoE methods
- `ExpertPlacementStrategy` — Strategy for placing experts
- Supports multiple MoE models: Mixtral, Qwen2-MoE, DeepSeek-V2/V3, etc.

### Knowledge Delta vs. Megatron-LM
* **vs. Megatron-LM:** vLLM focuses on inference; Megatron-LM on training
* **Unique insight:** Shows disaggregated serving architecture for inference

---

## Phase 3: Knowledge Synthesis (The Takeaway)

### Must-Know Summary

**vLLM** demonstrates **disaggregated inference** architecture. The key insights:

1. **Prefill-Decode Separation**: KV transfer enables running prefill and decode on different GPU servers
2. **Custom All-Reduce**: P2P-based all-reduce for lower latency intra-node communication
3. **Fused MoE**: Supports MoE models with expert parallelism for inference

### Portability Assessment

| Technique | Portable? | Notes |
|-----------|-----------|-------|
| KV transfer | High | Protocol-agnostic (can use different connectors) |
| Custom all-reduce | Medium | Requires NVLink/P2P |
| Fused MoE | High | Triton kernels, portable |

### Application to Your Goals

For **Tier 2, Obj 1 (Disaggregated Computing)**:
- vLLM's KV transfer is the reference for prefill-decode separation
- Use custom all-reduce for low-latency multi-GPU inference
- For MoE inference: vLLM provides excellent support

### Files Analyzed
- `vllm/distributed/device_communicators/custom_all_reduce.py` — Custom all-reduce (lines 51-80)
- `vllm/distributed/kv_transfer/kv_transfer_state.py` — KV transfer
- `vllm/model_executor/layers/fused_moe/layer.py` — Fused MoE (lines 1-80)
