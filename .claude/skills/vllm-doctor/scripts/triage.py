#!/usr/bin/env python3
"""triage.py — Phase 2 决策树：吃 golden3.json 吐 playbook 路由结果。

用法：
  python3 triage.py < golden3.json                    # 单次路由
  python3 triage.py --verify v1.json v2.json v3.json  # 三次采样判定是否恢复

线上路由必须显式提供本服务验证过的阈值：
  TTFT_SLO_MS, QUEUE_HIGH, KV_HIGH, PREEMPT_HIGH_PER_SEC,
  PREFIX_CACHE_DROP_FROM, RUNNING_LOW
若传入网关失败率或格式合规率，还要提供 REQUEST_FAILED_HIGH 或
FORMAT_COMPLIANCE_LOW。仅离线 fixture 可设置
VLLM_DOCTOR_USE_EXAMPLE_THRESHOLDS=1 使用脚本内合成阈值。
"""

from __future__ import annotations

import json
import os
import sys


EXAMPLE_MODE = os.environ.get("VLLM_DOCTOR_USE_EXAMPLE_THRESHOLDS") == "1" or bool(
    os.environ.get("VLLM_DOCTOR_FIXTURE")
)


def env_f(name: str, example: float) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return example if EXAMPLE_MODE else None
    try:
        return float(raw)
    except ValueError:
        return None


TTFT_SLO_MS = env_f("TTFT_SLO_MS", 2000)
QUEUE_HIGH = env_f("QUEUE_HIGH", 50)
KV_HIGH = env_f("KV_HIGH", 0.9)
PREEMPT_HIGH = env_f("PREEMPT_HIGH_PER_SEC", 0.5)
PREFIX_CACHE_LOW = env_f("PREFIX_CACHE_DROP_FROM", 0.5)
RUNNING_LOW = env_f("RUNNING_LOW", 5)
REQUEST_FAILED_HIGH = env_f("REQUEST_FAILED_HIGH", 0.1)
FORMAT_COMPLIANCE_LOW = env_f("FORMAT_COMPLIANCE_LOW", 0.9)
THROUGHPUT_DEAD = 1e-6  # 视为 0


def route(g: dict) -> dict:
    """返回 {playbook, confidence, reason, alternatives}."""
    required = (
        "ttft_p99_ms",
        "queue",
        "kv_usage",
        "throughput",
        "running",
        "prefix_cache_hit_rate",
        "preempt_rate_per_sec",
    )
    missing = [key for key in required if g.get(key) is None]
    if missing:
        return {
            "playbook": "none",
            "confidence": 0.0,
            "reason": "insufficient evidence; missing metrics: " + ", ".join(missing),
            "alternatives": [],
        }

    threshold_config = {
        "TTFT_SLO_MS": TTFT_SLO_MS,
        "QUEUE_HIGH": QUEUE_HIGH,
        "KV_HIGH": KV_HIGH,
        "PREEMPT_HIGH_PER_SEC": PREEMPT_HIGH,
        "PREFIX_CACHE_DROP_FROM": PREFIX_CACHE_LOW,
        "RUNNING_LOW": RUNNING_LOW,
    }
    if g.get("request_failed_rate") is not None:
        threshold_config["REQUEST_FAILED_HIGH"] = REQUEST_FAILED_HIGH
    if g.get("format_compliance_rate") is not None:
        threshold_config["FORMAT_COMPLIANCE_LOW"] = FORMAT_COMPLIANCE_LOW
    missing_thresholds = [
        name for name, value in threshold_config.items() if value is None
    ]
    if missing_thresholds:
        return {
            "playbook": "none",
            "confidence": 0.0,
            "reason": "insufficient evidence; missing thresholds: "
            + ", ".join(missing_thresholds),
            "alternatives": [],
        }

    assert TTFT_SLO_MS is not None
    assert QUEUE_HIGH is not None
    assert KV_HIGH is not None
    assert PREEMPT_HIGH is not None
    assert PREFIX_CACHE_LOW is not None
    assert RUNNING_LOW is not None

    ttft = float(g["ttft_p99_ms"])
    queue = float(g["queue"])
    kv = float(g["kv_usage"])
    tput = float(g["throughput"])
    running = float(g["running"])
    cache_hit = float(g["prefix_cache_hit_rate"])
    preempt = float(g["preempt_rate_per_sec"])
    failed_raw = g.get("request_failed_rate")
    failed = None if failed_raw is None else float(failed_raw)
    fmt_raw = g.get("format_compliance_rate")
    fmt_ok = None if fmt_raw is None else float(fmt_raw)
    oom_killed = float(g.get("oom_killed") or 0)

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

    if oom_killed > 0:
        candidates.append(
            (0.95, "03-gpu-oom", f"observed oom_killed={oom_killed:.0f}")
        )
    if kv_pressure and (preempting or (ttft_bad and queue_bad)):
        candidates.append(
            (0.85, "01-preempt-cascade", f"kv={kv:.2f} preempt={preempt:.2f}/s + TTFT/queue 高")
        )

    # 重试雪崩：失败率突增 + 队列异常但 KV 不算高（KV 高就是真过载，不是雪崩）
    if (
        failed is not None
        and REQUEST_FAILED_HIGH is not None
        and failed > REQUEST_FAILED_HIGH
        and not kv_pressure
    ):
        candidates.append(
            (0.75, "04-retry-storm", f"request_failed={failed:.2f}/s 上升但 kv={kv:.2f} 不算紧")
        )

    # prefix cache 命中率塌方
    if cache_hit < PREFIX_CACHE_LOW:
        candidates.append(
            (0.7, "05-cache-hit-regression", f"prefix_cache_hit={cache_hit:.2f} 低于 {PREFIX_CACHE_LOW}")
        )

    # 冷启动：TTFT 高但 KV/queue 不高，多半在加载
    if ttft_bad and not kv_pressure and queue < QUEUE_HIGH and running < RUNNING_LOW:
        candidates.append(
            (0.6, "06-cold-start", f"TTFT={ttft:.0f}ms 高，但 KV/queue 都低，running={running:.0f} 少")
        )

    # 输出质量：格式合规率塌
    if (
        fmt_ok is not None
        and FORMAT_COMPLIANCE_LOW is not None
        and fmt_ok < FORMAT_COMPLIANCE_LOW
    ):
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
    """汇总路由状态；真正恢复仍需逐条满足命中 playbook 的验证门。"""
    routings = [route(s) for s in samples]
    insufficient = [
        routing
        for routing in routings
        if routing["reason"].startswith("insufficient evidence")
    ]
    if insufficient:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "samples": routings,
        }
    if all(r["playbook"] == "none" for r in routings):
        return {
            "status": "NO_ACTIVE_ROUTE",
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
