# Part IV · 优化 · 性能优化

推理加速技术：量化、投机解码、CUDA Graph + torch.compile、编译内核、存算比定量推导。

← [返回主目录](../README.md) · 📖 [完整目录](../CONTENTS.md)

## 本章目录

| 文件 | 为什么读 |
| --- | --- |
| [`01-quantization`](01-quantization.md) | FP8 / INT4 / AWQ / GPTQ / Marlin 的选型矩阵 |
| [`02-speculative-decoding`](02-speculative-decoding.md) | n-gram / EAGLE / Medusa / MTP 的取舍 |
| [`03-cudagraph-and-compile`](03-cudagraph-and-compile.md) | CUDA Graph + torch.compile 何时开/关/降级 |
| [`04-compilation-internals`](04-compilation-internals.md) | CompilerManager / VllmBackend / 自定义 pass 深度 |
| [`05-roofline`](05-roofline-and-arithmetic-intensity.md) | 🆕 存算比推导：prefill 为何 compute-bound、decode 为何 memory-bound |

