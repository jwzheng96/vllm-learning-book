# NCCL / ProcessGroup 环境变量核对框架

变量会随 NCCL、PyTorch 和平台版本变化。先记录镜像中的准确版本，再查对应官方文档和当前环境；本页不提供可直接复制的“推荐值”。任何 env 变更都是需显式批准的集群 mutation，并要先做 staging hang 演练。

## 看门狗 / 超时

| 变量/配置 | 核对内容 |
| --- | --- |
| ProcessGroup timeout | 从应用/框架配置确认单位、作用域与 abort 行为，不假设存在某个 NCCL env |
| `TORCH_NCCL_BLOCKING_WAIT` | 查目标 PyTorch 版本的等待与错误传播语义 |
| `TORCH_NCCL_ENABLE_MONITORING` | 核对是否需要配套 heartbeat/trace 配置，以及进程终止行为 |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 按业务 step 分布和故障演练选择，不使用通用秒数 |

目标是让 collective 无进展可被观测、在批准窗口内失败，并由平台按既定策略恢复；环境变量本身不保证 crash、restart 或数据面可用。

## 调试

| 变量 | 核对内容 |
| --- | --- | --- |
| `NCCL_DEBUG` | 输出级别、性能/日志量和敏感信息风险 |
| `NCCL_DEBUG_SUBSYS` | 只启用调查所需子系统 |
| `NCCL_DEBUG_FILE` | 路径权限、磁盘配额、每 rank 区分与回收 |

> 生产不要长期开 `INFO`，会刷爆日志。

## 拓扑 / 传输

| 变量 | 说明 |
| --- | --- |
| `NCCL_IB_DISABLE` | 设 1 关闭 InfiniBand（IB 故障应急用，性能会掉） |
| `NCCL_P2P_DISABLE` | 设 1 关闭 GPU P2P（NVLink 故障应急用） |
| `NCCL_SOCKET_IFNAME` | 只有拓扑和 route 证据支持时才显式指定网卡 |
| `NCCL_IB_HCA` | 只有 inventory 与 HCA 映射确认后才指定 |
| `NCCL_ALGO` | 强制算法（默认 auto），生产不建议改 |

## 常见踩坑

1. **mesh / 网络策略改变通信路径**：从实际 transport、监听端口和连接证据生成排除规则，不照抄端口段。
2. **接口选择与拓扑不符**：多网卡节点先记录自动选择结果、route/RDMA 映射和性能，再决定是否覆盖。
3. **rank 间 NCCL 版本不一致**：不同 image tag 跨 worker 部署时容易出现，必须同时升级。

## 来源

`vllm-learning/05-distributed/01-tp-pp-ep.md`、`vllm-learning/08-production-deployment/06-reliability-and-failure-modes.md`。变量语义以目标 NCCL/PyTorch 版本的官方文档为准。
