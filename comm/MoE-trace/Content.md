
# modular specs

| hardware specs     | A100 |
| ------------------ | ---- |
| TFLOPs (cuda core) |      |
| D2H                |      |
| H2D                |      |
| nvlink             |      |
| nvshemem           |      |


| software specs | triton-distributed | nccl | nvshmem | hccl | Ashmem | RDMA |
| -------------- | ------------------ | ---- | ------- | ---- | ------ | ---- |
| throughout     |                    |      |         |      |        |      |
| latency        |                    |      |         |      |        |      |
| "demos"        |                    |      |         |      |        |      |


| model specs | Qwen3 30-A3 | Qwen3 235-A22 | Qwen3.5 35-A3 | Qwen3.5 122-A10 |
| ----------- | ----------- | ------------- | ------------- | --------------- |
|             |             |               |               |                 |

| model specs | Step 3.5 196-A11 | Ling2 100-A6 | oss 21-A3 | oss 117-A5 |
| ----------- | ---------------- | ------------ | --------- | ---------- |
|             |                  |              |           |            |


| model specs | M2.5 230-A10 |     |     |     |
| ----------- | ------------ | --- | --- | --- |
|             |              |     |     |     |


| kernel specs (fused MoE) | flashinfer | pplx | deepEP | triton-dist |
| ------------------------ | ---------- | ---- | ------ | ----------- |
|                          |            |      |        |             |

1. engine-side (cpu-side, threads/process) 
2. dataset patterns
3. different parallelism 
4. end-to-end prediction model
