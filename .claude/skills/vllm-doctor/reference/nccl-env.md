# NCCL 环境变量速查（vLLM 生产相关）

## 看门狗 / 超时

| 变量 | 推荐值 | 作用 |
| --- | --- | --- |
| `NCCL_TIMEOUT` | `60` | 单次集合通信超过 60s 就 abort，不要无限等 |
| `NCCL_BLOCKING_WAIT` | `1` | 阻塞等待时让进程能被信号打断（不然 SIGKILL 都收不到） |
| `TORCH_NCCL_ENABLE_MONITORING` | `1` | torch 端额外加 watchdog，把卡死栈打到日志 |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | `60` | torch.distributed 心跳超时 |

> 设这四个的核心目标：**把"NCCL 卡死"转成"60s 内 crash + 自动 restart"**。
> 没设这些时，pod 看起来 Ready，实际不能工作。

## 调试

| 变量 | 推荐值 | 作用 |
| --- | --- | --- |
| `NCCL_DEBUG` | `WARN`（生产）/ `INFO`（排障） | 输出 NCCL 内部状态 |
| `NCCL_DEBUG_SUBSYS` | `INIT,COLL` | 排障时只看初始化和集合通信 |
| `NCCL_DEBUG_FILE` | `/var/log/nccl-%h-%p.log` | 每个 pod / pid 一个 log，方便聚合 |

> 生产不要长期开 `INFO`，会刷爆日志。

## 拓扑 / 传输

| 变量 | 说明 |
| --- | --- |
| `NCCL_IB_DISABLE` | 设 1 关闭 InfiniBand（IB 故障应急用，性能会掉） |
| `NCCL_P2P_DISABLE` | 设 1 关闭 GPU P2P（NVLink 故障应急用） |
| `NCCL_SOCKET_IFNAME` | 显式指定网卡，如 `eth0,eth1`（多网卡机必须设） |
| `NCCL_IB_HCA` | 指定 IB HCA，如 `mlx5_0,mlx5_1` |
| `NCCL_ALGO` | 强制算法（默认 auto），生产不建议改 |

## 常见踩坑

1. **mesh 代理拦截 NCCL 端口**：服务网格 sidecar 把 NCCL 用的 6000-6100 + IB 端口走了 Envoy，NCCL 握手失败。解决：在 sidecar 配置里 exclude 这些端口。
2. **`NCCL_SOCKET_IFNAME` 没设**：多网卡节点上 NCCL 可能选到错误的网卡（控制面网卡而不是 RDMA 网卡），带宽塌方。
3. **rank 间 NCCL 版本不一致**：不同 image tag 跨 worker 部署时容易出现，必须同时升级。

## 来源

`vllm-learning/05-distributed/01-tp-pp-ep.md`、`vllm-learning/08-production-deployment/06-reliability-and-failure-modes.md` L79-111
