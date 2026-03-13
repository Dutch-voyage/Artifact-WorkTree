### Phase 1: Strategic Screening Template

This template ensures that every minute spent on a codebase directly advances your research goals in GPU networking and MoE optimization.

> **Project Name:** [e.g., `ucxx` or `triton-distributed`]
> **A. Objective Alignment (The "Why"):**
> * **Primary Goal Linked:** Which specific Objective (e.g., Tier 1, Obj 2) does this project help with?
> * **The Delta:** What does this project teach me about that goal that I haven't learned from other repos yet?
> 
> 
> **B. The "Must-Know" Bridge (Prerequisites):**
> * **What is the missing link?** What basic concept (e.g., "Memory Pinning" or "CUDA Streams") do I need to learn *now* so I can understand the high-level logic later?
> 
> 
> **C. The Target Map (Where to look):**
> * **The Engine Folder:** Identify the single most important directory for my objectives.
> * **Keywords for Grep:** Give me 3-5 specific terms to find the "lever" code (e.g., `all_to_all`, `stdexec`, `multi_rail`).
> 
> 
> **D. The "Skip" List (Noise Suppression):**
> * **What to ignore:** Explicitly list folders that are just "noise" (e.g., CLI tools, legacy support, or standard Python wrappers).
> 
> 

---

### Evaluation Rubric: Revised Phase 1

| Metric | The Simple Question | Success Criteria |
| --- | --- | --- |
| **Objective Fit** | Did it link the code to my Tier 1 or Tier 2 goals? | The agent names a specific objective from your list. |
| **No Overlap** | Did it explain the "New Knowledge" (Delta)? | It identifies a unique trick (e.g., "This repo handles multi-rail jitter"). |
| **Starter Ready** | Is the "Prerequisite" actually helpful? | It points to a specific concept that would stop a beginner from understanding the code. |
| **Aggressive Filtering** | Did it hide the junk? | It clearly marks all non-essential files as "Skip." |

---

### Example: Screening `triton-distributed` for your Objectives

> **Project Name:** `triton-distributed`
> **A. Objective Alignment:**
> * **Primary Goal Linked:** Tier 1, Obj 2 (**MoE Collective Optimization**).
> * **The Delta:** It shows how to **fuse** communication (OpenSHMEM) directly into GPU kernels, which is a step beyond standard NCCL calls.
> 
> 
> **B. The "Must-Know" Bridge:**
> * **One-Sided Communication:** You need to understand how "Put/Get" works in networking vs. "Send/Recv."
> 
> 
> **C. The Target Map:**
> * **The Engine Folder:** `lib/Dialect/Triton/Transforms/`
> * **Keywords for Grep:** `persistent`, `shmem`, `overlap`.
> 
> 
> **D. The "Skip" List:**
> * **What to ignore:** `python/`, `tests/`, and any file related to standard TCP networking.
> 
> 