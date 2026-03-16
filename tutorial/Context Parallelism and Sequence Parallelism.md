## Context Parallelism (Ring Attention)

[图解大模型训练系列：序列并行3，Ring Attention](https://zhuanlan.zhihu.com/p/4963530231)
![](figures/Pasted_image_20260313154648.png)
![](../figures/Pasted_image_20260313154715.png)
![](../figures/Pasted_image_20260313154726.png)
![](../figures/Pasted_image_20260313154737.png)
### compute-communication overlap 
![](../figures/Pasted_image_20260313161422.png)
[\[2412.20501v1\] TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication](https://arxiv.org/abs/2412.20501v1)
### (\*\*\*) balance in computation 
> [!note]
> Recall that chunking is position irrelevant!!!

[# ring attention + flash attention：超长上下文之路](https://zhuanlan.zhihu.com/p/683714620)
![](../figures/Pasted_image_20260313170521.png)
![](../figures/Pasted_image_20260313170539.png)
![](../figures/Pasted_image_20260313170548.png)

### dynamic chunking (Context Parallelism in training) 
pipeline parallelism 
[Speeding Up Variable-Length Training with Dynamic Context Parallelism and NVIDIA Megatron Core \| NVIDIA Technical Blog](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)
![](../figures/Pasted_image_20260313161636.png)
[\[2102.07988\] TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models](https://arxiv.org/abs/2102.07988)
## Sequence Parallelism
deepspeed ulysses

**Non-Attention Layer**
![](../figures/Pasted_image_20260315180427.png)
**Attention Layer**
![](../figures/Pasted_image_20260315180503.png)
![](../figures/Pasted_image_20260315181452.png)