general MoE inference
- [ ] serving-level stats
- [ ] hardware-level stats
- [ ] modular-level stats

profiling tools
- [ ] sglang bench serving
	- [ ] scripts (in-progress)
	- [ ] visualization (in-progress)
	- [ ] insights in overall design (required)
- [ ] torch profiler
- [ ] nsight sys (nvtx)

Qwen3.5 ascend training 

1. llama-factory 0.9.4 -> transformers v5.2.0
	1. test fused-moe functionality 
2. torchturbo refactor-branch add qwen3-next patch
3. torchturbo add qwen3-moe vl
4. torchturbo add qwen3-5 vl 


