## Phase 2: Systematic Review & Essence Mapping

The goal of Phase 2 is to move from a general map to **Grounded Insight**. You are no longer asking *if* the code is useful; you are identifying the specific lines of code that act as the "Levers" for performance or logic.

### 1. The Essence Extraction Map (Phase 2 Template)

The agent uses the "Engine Folders" and "Keywords" from Phase 1 to produce this detailed map.

> **Target Module:** [Specific Engine Folder from Phase 1]
> **A. The "Should-Do" Logic (The Levers):**
> * **File Path:** [e.g., `src/comm/nccl_all2all.cu`]  **Mechanism:** How the code handles the **All-to-All** shuffle for MoE tokens.
> * **File Path:** [e.g., `runtime/executor.cpp`]  **Mechanism:** The specific implementation of the **Sender/Receiver** state machine.
> 
> 
> **B. Objective-Specific Observations:**
> * **Observation 1:** [Link to Tier 1 Obj] – How this file specifically manages **Direct-to-GPU** memory paths.
> * **Observation 2:** [Link to Tier 2 Obj] – Evidence of **Overlap** between the network send and the math kernel.
> 
> 
> **C. The "Aha!" Moment (Grounded Insight):**
> * **Look at [File Path:Line Number]:** [e.g., `worker.cpp:L215`]
> * **The Logic:** Describe the unique trick (e.g., "This line uses a non-blocking hardware trigger to start the RDMA transfer before the kernel finishes").
> 
> 

---

### 2. Evaluation Rubric: Phase 2 Quality

| Metric | The Simple Question | Success Criteria |
| --- | --- | --- |
| **Grounded Accuracy** | Are the file paths and line numbers correct? | You can navigate directly to the "Lever" code without searching. |
| **Mechanism Focus** | Does it explain the *how*, not just the *what*? | The description explains the logic of the "trick" (e.g., "This handles jitter by..."). |
| **No Repetition** | Did it stay focused on the "Engine"? | The report does not re-analyze the project structure or boilerplate already filtered in Phase 1. |
| **Objective Linkage** | Does the "Aha!" moment serve your Tier 1/2 goals? | The insight directly helps you master one of the objectives listed in your plan. |

---

### Phase 2 Workflow: From Map to Lever

1. **Direct Navigation:** Open the "Engine Folder" identified in Phase 1.
2. **Keyword Hunt:** Run the `grep` commands from Phase 1 to find the entrance to the logic.
3. **Trace the Data:** Follow the primary tensor or memory buffer. If it goes into a standard utility, stop. If it goes into a unique communication primitive or scheduling logic, mark it as a **Lever**.
4. **Identify the "Aha!" Moment:** Find the single point where the most complex objective is solved (e.g., where the **Multi-rail** traffic is split).

---
