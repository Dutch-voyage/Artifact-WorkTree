## Tier 1: Main Objective

1. **HPC Network Architecture:** Master the fundamental hardware-software interface of modern GPU fabrics, focusing on **Direct-to-GPU** communication and **Multi-rail** network topologies.
2. **Collective Optimization:** Evaluate and optimize communication primitives (All-to-All, All-Reduce) specifically for **MoE (Mixture of Experts)** architectures, focusing on reducing latency jitter and improving bandwidth utilization.
3. **Distributed Profiling:** Develop a methodology for **continuous performance monitoring** that correlates GPU compute kernels with network traffic to identify global bottlenecks in a cluster.

## Tier 2: Side Objective

1. **Distributed Inference Systems:** Understand the architectural trade-offs of **Disaggregated Computing** (separating compute-intensive tasks from memory-intensive tasks) and the networking overhead involved in state transfer.
2. **Modern C++ Concurrency:** Implement high-performance task scheduling using **Sender/Receiver patterns** (`std::execution`) to manage heterogeneous workloads across CPUs, GPUs, and NICs.

## Tier 3 Meta Objective 
See [Rubric.md](Rubric.md)
