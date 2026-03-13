# Analysis: sglang
- [x] Phase 1: Strategic Screening
- [ ] Phase 2: Systematic Review
- [ ] Phase 3: Knowledge Synthesis

---

## Phase 1: Strategic Screening

### Project Name: `sglang`

#### A. Objective Alignment (The "Why"):
* **Primary Goal Linked:** Tier 2, Obj 1 — **Distributed Inference** (Disaggregated Computing)
* **The Delta:** sglang is similar to vLLM but with **advanced MoE dispatchers** (DeepEP, MoriEP, FusedEP) and **RadixAttention** for prefix caching. It also has sophisticated PD (Prefill-Decode) disaggregation with separate servers.

#### B. The "Must-Know" Bridge (Prerequisites):
* **Attention Optimization:** Understanding RadixAttention and how it differs from PagedAttention.

#### C. The Target Map (Where to look):
* **The Engine Folder:** `/Users/pobs/workspace/moe_project/sglang/python/sglang/srt/layers/moe/` and `/Users/pobs/workspace/moe_project/sglang/python/sglang/srt/disaggregation/`
* **Keywords for Grep:**
  - `ep_moe` — Expert parallelism MoE
  - `token_dispatcher` — Token routing
  - `RadixAttention` — Prefix caching
  - `disaggregation` — PD separation
  - `elastic_ep` — Elastic EP

#### D. The "Skip" List (Noise Suppression):
* **What to ignore:**
  - `sglang/docs/` — documentation
  - `sglang/test/` — test files
  - `sglang/benchmark/` — benchmarks
  - `sglang/examples/` — example scripts

---

## Phase 2: Systematic Review (The Deep-Dive)

### Lever Code Blocks Identified

#### 1. Expert Parallelism MoE
**File:** `/Users/pobs_project/sglang/workspace/moe/python/sglang/srt/layers/moe/ep_moe/layer.py`

**What it does:** Implements DeepEPMoE using DeepEP library.

**Key mechanism:**
- Multiple token dispatchers: DeepEP, MoriEP, FusedEP, Standard
- Multiple runner backends: Triton, Deep GEMM, Marlin, FlashInfer

#### 2. Token Dispatchers
**File:** `/Users/pobs/workspace/moe_project/sglang/python/sglang/srt/layers/moe/token_dispatcher/`

**What it does:** Different strategies for routing tokens to experts.

**Dispatcher types:**
- `deepep.py` — DeepEP dispatcher
- `moriep.py` — MoriEP dispatcher
- `fuseep.py` — FusedEP dispatcher

#### 3. Disaggregation
**File:** `/Users/pobs/workspace/moe_project/sglang/python/sglang/srt/disaggregation/`

**What it does:** Separate prefill and decode servers for PD disaggregation.

**Components:**
- `prefill.py` — Prefill server
- `decode.py` — Decode server

### Knowledge Delta vs. vLLM
* **vs. vLLM:** sglang has more advanced EP dispatchers and RadixAttention
* **Unique insight:** Shows multiple MoE backend options for different scenarios

---

## Phase 3: Knowledge Synthesis (The Takeaway)

### Must-Know Summary

**sglang** provides **advanced inference optimizations**. The key insights:

1. **Multiple EP Dispatchers**: DeepEP, MoriEP, FusedEP for different scenarios
2. **RadixAttention**: Efficient prefix caching for repeated prompts
3. **PD Disaggregation**: Sophisticated prefill-decode separation

### Portability Assessment

| Technique | Portable? | Notes |
|-----------|-----------|-------|
| EP dispatchers | Medium | Library-dependent |
| RadixAttention | High | Can be ported |
| PD disaggregation | High | Architecture-agnostic |

### Application to Your Goals

For **Tier 2, Obj 1 (Distributed Inference)**:
- sglang shows the state-of-art in inference optimization
- Choose dispatcher based on hardware (DeepEP for RDMA, FusedEP for fusion)

### Files Analyzed
- `python/sglang/srt/layers/moe/ep_moe/layer.py` — DeepEPMoE
- `python/sglang/srt/layers/moe/token_dispatcher/` — Token dispatchers
- `python/sglang/srt/disaggregation/` — PD disaggregation
