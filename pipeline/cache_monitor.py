from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def extract_cache_metrics(provider: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes cache metrics across providers.
    Designed to work with your _extract_usage() outputs.
    """
    usage = usage or {}
    if provider == "openrouter":
        ptd = usage.get("prompt_tokens_details") or {}
        cached = _as_int(ptd.get("cached_tokens", 0))
        wrote = _as_int(ptd.get("cache_write_tokens", 0))
        return {
            "prompt_tokens": _as_int(usage.get("prompt_tokens", usage.get("input", 0))),
            "completion_tokens": _as_int(usage.get("completion_tokens", usage.get("output", 0))),
            "total_tokens": _as_int(usage.get("total_tokens", usage.get("total", 0))),
            "cached_tokens": cached,
            "cache_write_tokens": wrote,
            "cost": usage.get("cost", None),
            "generation_id": usage.get("generation_id", None),
        }

    if provider == "anthropic":
        return {
            "input_tokens": _as_int(usage.get("input_tokens", usage.get("input", 0))),
            "output_tokens": _as_int(usage.get("output_tokens", usage.get("output", 0))),
            "cache_read_input_tokens": _as_int(usage.get("cache_read_input_tokens", 0)),
            "cache_creation_input_tokens": _as_int(usage.get("cache_creation_input_tokens", 0)),
        }

    # fallback
    return dict(usage)


@dataclass
class CacheEvent:
    ts: float
    provider: str
    step: str
    model: str
    elapsed_s: float
    metrics: Dict[str, Any]
    meta: Dict[str, Any]


class CacheMonitor:
    """
    Live cache monitor.
    Enable via env var CACHE_MONITOR=1 (or pass enabled=True).
    Optionally writes JSONL to out_path for later analysis.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        *,
        print_each: bool = True,
        out_path: Optional[Path] = None,
    ) -> None:
        if enabled is None:
            enabled = os.getenv("CACHE_MONITOR", "").strip().lower() in {"1", "true", "yes", "y"}
        self.enabled = bool(enabled)
        self.print_each = bool(print_each)
        self.out_path = out_path
        self._totals: Dict[Tuple[str, str, str], Dict[str, float]] = {}

        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        provider: str,
        step: str,
        model: str,
        usage: Dict[str, Any],
        elapsed_s: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        meta = meta or {}
        metrics = extract_cache_metrics(provider, usage)
        ev = CacheEvent(
            ts=time.time(),
            provider=provider,
            step=step or "unknown",
            model=model,
            elapsed_s=_as_float(elapsed_s, 0.0),
            metrics=metrics,
            meta=meta,
        )

        self._accumulate(ev)

        if self.print_each:
            print(self._format_line(ev))

        if self.out_path:
            with self.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

    def _accumulate(self, ev: CacheEvent) -> None:
        key = (ev.provider, ev.step, ev.model)
        bucket = self._totals.setdefault(
            key,
            {
                "calls": 0.0,
                "elapsed_s": 0.0,
                "cached_tokens": 0.0,
                "cache_write_tokens": 0.0,
                "cache_read_input_tokens": 0.0,
                "cache_creation_input_tokens": 0.0,
                "cost": 0.0,
            },
        )
        bucket["calls"] += 1.0
        bucket["elapsed_s"] += ev.elapsed_s

        if ev.provider == "openrouter":
            bucket["cached_tokens"] += float(_as_int(ev.metrics.get("cached_tokens", 0)))
            bucket["cache_write_tokens"] += float(_as_int(ev.metrics.get("cache_write_tokens", 0)))
            cost = ev.metrics.get("cost", None)
            if cost is not None:
                bucket["cost"] += float(_as_float(cost, 0.0))

        if ev.provider == "anthropic":
            bucket["cache_read_input_tokens"] += float(_as_int(ev.metrics.get("cache_read_input_tokens", 0)))
            bucket["cache_creation_input_tokens"] += float(_as_int(ev.metrics.get("cache_creation_input_tokens", 0)))

    @staticmethod
    def _format_line(ev: CacheEvent) -> str:
        if ev.provider == "openrouter":
            cached = _as_int(ev.metrics.get("cached_tokens", 0))
            wrote = _as_int(ev.metrics.get("cache_write_tokens", 0))
            cost = ev.metrics.get("cost", None)
            pt = _as_int(ev.metrics.get("prompt_tokens", 0))
            ct = _as_int(ev.metrics.get("completion_tokens", 0))
            bits = [
                f"[CACHE] {ev.step}",
                f"model={ev.model}",
                f"pt={pt:,}",
                f"ct={ct:,}",
                f"cached={cached:,}",
                f"write={wrote:,}",
                f"elapsed={ev.elapsed_s:.2f}s",
            ]
            if cost is not None:
                bits.append(f"cost={cost}")
            return " | ".join(bits)

        if ev.provider == "anthropic":
            read = _as_int(ev.metrics.get("cache_read_input_tokens", 0))
            create = _as_int(ev.metrics.get("cache_creation_input_tokens", 0))
            it = _as_int(ev.metrics.get("input_tokens", 0))
            ot = _as_int(ev.metrics.get("output_tokens", 0))
            return (
                f"[CACHE] {ev.step} | model={ev.model} | in={it:,} out={ot:,} "
                f"| read={read:,} create={create:,} | elapsed={ev.elapsed_s:.2f}s"
            )

        return f"[CACHE] {ev.step} | model={ev.model} | elapsed={ev.elapsed_s:.2f}s"

    def summary(self) -> None:
        if not self.enabled:
            return
        print("\n=== CACHE MONITOR SUMMARY ===")
        for (provider, step, model), t in sorted(self._totals.items()):
            calls = int(t["calls"])
            if provider == "openrouter":
                print(
                    f"{provider} | {step} | {model} | calls={calls} "
                    f"| cached={int(t['cached_tokens']):,} write={int(t['cache_write_tokens']):,} "
                    f"| cost={t['cost']:.6f} | elapsed={t['elapsed_s']:.1f}s"
                )
            elif provider == "anthropic":
                print(
                    f"{provider} | {step} | {model} | calls={calls} "
                    f"| read={int(t['cache_read_input_tokens']):,} create={int(t['cache_creation_input_tokens']):,} "
                    f"| elapsed={t['elapsed_s']:.1f}s"
                )