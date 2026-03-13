![[Pasted image 20260313165112.png]]
![[Pasted image 20260313165151.png]]
## Solution 1: Train with load-balancing loss
pass 
## Solution 2: Duplicate "hot" experts
![[Pasted image 20260313165322.png]]
## Solution 3: Token dropping

![[Pasted image 20260313165456.png]]

## New Scenario: Chunked Prefill 

### Question 1: does chunking increase/decrease imbalance?
Guess: imbalance is increased 
### Question 2: Can experts routing be predicted 
Expert Prefetching 
[\[2509.07379\] DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference](https://arxiv.org/abs/2509.07379)
[\[2511.10676\] Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2511.10676)
### Question 3: Can chunking balance experts routing ? 
Guess: yes, very likely. 

