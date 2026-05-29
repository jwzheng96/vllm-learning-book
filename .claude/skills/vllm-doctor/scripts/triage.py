#!/usr/bin/env python3
"""triage.py — Phase 2 决策树：吃 golden3.json 吐 playbook 路由结果。

用法：
  python3 triage.py < golden3.json                    # 单次路由
  python3 triage.py --verify v1.json v2.json v3.json  # 三次采样判定是否恢复

环境变量（可调阈值）：
  TTFT_SLO_MS              默认 2000
  QUEUE_HIGH               默认 50
  KV_HIGH                  默认 0.9
  KV_CRITICAL              默认 0.95
  PREEMPT_HIGH_PER_SEC     默认 0.5
  PREFIX_CACHE_DROP_FROM   默认 0.5  （命中率绝对值低于该阈值视为塌方）
  FORMAT_COMPLIANCE_LOW    默认 0.9
"""

from __future__ import annotations

import json
import os
import sys


def env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


TTFT_SLO_MS = env_f("TTFT_SLO_MS", 2000)
QUEUE_HIGH = env_f("QUEUE_HIGH", 50)
KV_HIGH = env_f("KV_HIGH", 0.9)
KV_CRITICAL = env_f("KV_CRITICAL", 0.95)
PREEMPT_HIGH = env_f("PREEMPT_HIGH_PER_SEC", 0.5)
PREFIX_CACHE_LOW = env_f("PREFIX_CACHE_DROP_FROM", 0.5)
FORMAT_COMPLIANCE_LOW = env_f("FORMAT_COMPLIANCE_LOW", 0.9)
THROUGHPUT_DEAD = 1e-6  # 视为 0


def route(g: dict) -> dict:
    """返回 {playbook, confidence, reason, alternatives}."""
    ttft = float(g.get("ttft_p99_ms", 0))
    queue = float(g.get("queue", 0))
    kv = float(g.get("kv_usage", 0))
    tput = float(g.get("throughput", 0))
    running = float(g.get("running", 0))
    cache_hit = float(g.get("prefix_cache_hit_rate", 1))
    preempt = float(g.get("preempt_rate_per_sec", 0))
    failed = float(g.get("request_failed_rate", 0))
    fmt_ok = float(g.get("format_compliance_rate", 1))

    candidates: list[tuple[float, str, str]] = []

    # NCCL hang：分支强证据，最先判
    if tput <= THROUGHPUT_DEAD and running > 0:
        candidates.append(
            (0.95, "02-nccl-hang", f"throughput≈0 AND running={running:.0f} >0 — 进程在但不出 token")
        )

    # 抢占级联 / OOM
    kv_pressure = kv >= KV_HIGH
    preempting = preempt >= PREEMPT_HIGH
    ttft_bad = ttft > TTFT_SLO_MS
    queue_bad = queue >= QUEUE_HIGH

    if kv >= KV_CRITICAL and (preempting or queue_bad):
        candidates.append(
            (0.9, "03-gpu-oom", f"kv={kv:.2f} 接近 OOM 边缘 + 队列/抢占同时高")
        )
    if kv_pressure and (preempting or (ttft_bad and queue_bad)):
        candidates.append(
            (0.85, "01-preempt-cascade", f"kv={kv:.2f} preempt={preempt:.2f}/s + TTFT/queue 高")
        )

    # 重试雪崩：失败率突增 + 队列异常但 KV 不算高（KV 高就是真过载，不是雪崩）
    if failed > 0.1 and not kv_pressure:
        candidates.append(
            (0.75, "04-retry-storm", f"request_failed={failed:.2f}/s 上升但 kv={kv:.2f} 不算紧")
        )

    # prefix cache 命中率塌方
    if cache_hit < PREFIX_CACHE_LOW:
        candidates.append(
            (0.7, "05-cache-hit-regression", f"prefix_cache_hit={cache_hit:.2f} 低于 {PREFIX_CACHE_LOW}")
        )

    # 冷启动：TTFT 高但 KV/queue 不高，多半在加载
    if ttft_bad and not kv_pressure and queue < QUEUE_HIGH and running < 5:
        candidates.append(
            (0.6, "06-cold-start", f"TTFT={ttft:.0f}ms 高，但 KV/queue 都低，running={running:.0f} 少")
        )

    # 输出质量：格式合规率塌
    if fmt_ok < FORMAT_COMPLIANCE_LOW:
        candidates.append(
            (0.7, "07-output-quality", f"format_compliance={fmt_ok:.2f} < {FORMAT_COMPLIANCE_LOW}")
        )

    # LoRA 抖动需要额外信号，决策树这里给低分占位，让 playbook 自己再验证
    # （TTFT 高 + 没命中其他强证据 + 部署里有 lora 是必要条件，留给运维补强）

    if not candidates:
        return {
            "playbook": "none",
            "confidence": 0.0,
            "reason": "Golden 3 都在正常范围；如果有体感故障，请人工核对客户端日志或开 OTel trace。",
            "alternatives": [],
        }

    candidates.sort(key=lambda x: -x[0])
    top = candidates[0]
    return {
        "playbook": top[1],
        "confidence": top[0],
        "reason": top[2],
        "alternatives": [
            {"playbook": p, "confidence": c, "reason": r} for c, p, r in candidates[1:]
        ],
    }


def verify(samples: list[dict]) -> dict:
    """三次采样都未命中任何 playbook → RESOLVED；否则附最严重的一次。"""
    routings = [route(s) for s in samples]
    if all(r["playbook"] == "none" for r in routings):
        return {
            "status": "RESOLVED",
            "samples": routings,
        }
    worst = max(routings, key=lambda r: r["confidence"])
    return {
        "status": "NOT_RESOLVED",
        "still_routing_to": worst,
        "samples": routings,
    }


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--verify":
        files = args[1:]
        samples = [json.load(open(f)) for f in files]
        json.dump(verify(samples), sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    g = json.load(sys.stdin)
    json.dump(route(g), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
