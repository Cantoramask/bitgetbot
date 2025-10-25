#!/usr/bin/env python3
# bitgetbot/ai/reasoner.py
"""
ai/reasoner.py
Optional OpenAI advisor that votes on entries and returns a confidence score.

Plain English.
This is the advisor. It can say yes or no and a confidence score based on a short snapshot.
If OPENAI_API_KEY is missing or ADVISOR_ENABLED is false, it always approves so it never blocks you.

Definition.
API key is a secret string that proves who you are to a service like OpenAI.
"""

from __future__ import annotations

import os
import json
import threading
from typing import Any, Dict, Tuple, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class Reasoner:
    def __init__(self, logger=None):
        self.log = logger
        self.enabled = os.getenv("ADVISOR_ENABLED", "false").lower() in ("1", "true", "y", "yes")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = None
        if self.enabled and self.api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

        self.mode = os.getenv("ADVISOR_MODE", "warn").lower()
        self.req_timeout_sec = float(os.getenv("ADVISOR_TIMEOUT_SEC", "1.5"))
        self.max_retries = int(os.getenv("ADVISOR_MAX_RETRIES", "1") or 1)

    # ---------- internal helpers ----------

    def _call_llm(self, prompt: str) -> str:
        """
        Blocking call. Wrapped by _safe_llm to avoid stalling any event loop.
        """
        if not self.client:
            raise RuntimeError("no_client")

        try:
            # Prefer strict JSON on newer SDKs
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=self.req_timeout_sec,  # supported on newer SDKs
            )
            return (resp.choices[0].message.content or "").strip()
        except TypeError:
            # Fallback for older SDKs without response_format/timeout kw
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return (resp.choices[0].message.content or "").strip()

    def _safe_llm(self, prompt: str, timeout: float) -> Tuple[bool, Optional[str], str]:
        """
        Run _call_llm in a thread with a hard timeout.
        Returns (ok, content, err_note)
        """
        result: Dict[str, Any] = {"content": None, "err": None}

        def target():
            try:
                result["content"] = self._call_llm(prompt)
            except Exception as e:
                result["err"] = f"advisor_error: {e}"

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(timeout=timeout)
        if th.is_alive():
            return False, None, "advisor_timeout"
        if result["err"] is not None:
            return False, None, str(result["err"])
        return True, str(result["content"] or ""), ""

    # ---------- public API ----------

    def evaluate(self, snapshot: Dict[str, Any]) -> Tuple[bool, float, str, float, Dict[str, float]]:
        """
        Returns (allow, confidence, note, edge_bps, suggested_overrides)

        When disabled or unavailable it returns a conservative pass:
        (True, 0.65, "disabled", 0.0, {}).
        """
        if not self.enabled or not self.client:
            return True, 0.65, "disabled", 0.0, {}

        fees_bps = float(snapshot.get("fees_bps", 6.0))
        slip_bps = float(snapshot.get("slip_bps", 1.0))
        total_cost_bps = fees_bps + slip_bps

        policy = (
            "You are a cautious trading gate. Consider fees, slippage, funding direction, "
            "realised volatility regime, and actual leverage. Prefer WAIT when edge is small. "
            "Only ALLOW if edge_bps exceeds total_cost_bps by a modest buffer. "
            "Output STRICT JSON only with keys: "
            '{"allow": boolean, "confidence": number, "edge_bps": number, '
            '"note": string, "overrides": {"trail_pct_init": number, '
            '"trail_pct_tight": number, "intelligence_sec": integer}}. '
            "Confidence is 0..1. If uncertain, lower confidence rather than forcing allow."
        )
        prompt = f"{policy} Snapshot: {snapshot} total_cost_bps={total_cost_bps}"

        ok = False
        content = None
        err_note = ""
        for _ in range(self.max_retries + 1):
            ok, content, err_note = self._safe_llm(prompt, self.req_timeout_sec)
            if ok and content:
                break

        if not ok or not content:
            return True, 0.65, err_note or "advisor_fallback", 0.0, {}

        try:
            data = json.loads(content)
            allow = bool(data.get("allow", True))
            conf = float(data.get("confidence", 0.65))
            edge_bps = float(data.get("edge_bps", 0.0))
            note = str(data.get("note", ""))
            overrides = data.get("overrides") or {}
            safe_overrides: Dict[str, float] = {}
            if "trail_pct_init" in overrides:
                safe_overrides["trail_pct_init"] = float(overrides["trail_pct_init"])
            if "trail_pct_tight" in overrides:
                safe_overrides["trail_pct_tight"] = float(overrides["trail_pct_tight"])
            if "intelligence_sec" in overrides:
                safe_overrides["intelligence_sec"] = float(overrides["intelligence_sec"])
        except Exception:
            if self.log:
                try:
                    self.log.info(f"[AI] parse_fallback raw={str(content)[:200]!r}")
                except Exception:
                    pass
            allow = True
            conf = 0.65
            edge_bps = 0.0
            note = "parse_fallback"
            safe_overrides = {}

        if self.mode == "warn" and not allow:
            allow = True
            note = f"warn: {note}"

        note_oneline = " ".join(str(note).split())[:300]
        return allow, conf, note_oneline, edge_bps, safe_overrides

    def decide(self, strategy, params, context: Dict[str, Any]):
        """
        Returns side_choice, reason, trail_pct_to_use, decision_dict.

        side_choice is 'long', 'short', or 'wait'.
        decision_dict includes confidence 0..1, edge_bps, and suggested parameter overrides.
        This function is PURE: it does not mutate params. The orchestrator may apply
        suggested_overrides with its own bounds/decay logic.
        """
        side = "wait"
        reason = "no_signal"
        trail_from_strategy = None
        caution = None

        try:
            if strategy is not None:
                if hasattr(strategy, "decide"):
                    out = strategy.decide()
                    if isinstance(out, dict):
                        side = str(out.get("side", "wait")) or "wait"
                        reason = str(out.get("reason", "strategy_decide")) or "strategy_decide"
                        if out.get("trail_pct") is not None:
                            try:
                                trail_from_strategy = float(out["trail_pct"])
                            except Exception:
                                trail_from_strategy = None
                        caution = out.get("caution")
                elif hasattr(strategy, "suggest"):
                    side = strategy.suggest() or "wait"
                    reason = "strategy_suggest"
                elif hasattr(strategy, "vote"):
                    v = strategy.vote()
                    if isinstance(v, dict):
                        side = v.get("side", "wait")
                        reason = v.get("reason", "vote")
                    elif isinstance(v, tuple) and len(v) >= 1:
                        side = v[0] or "wait"
                        reason = "vote_tuple"
        except Exception:
            side = "wait"
            reason = "strategy_error"

        # prefer actual leverage provided by orchestrator/position
        lev_real: Optional[int] = None
        try:
            lev_real = int(context.get("actual_leverage"))
        except Exception:
            pass
        if not lev_real:
            try:
                lev_real = int(context.get("leverage"))
            except Exception:
                try:
                    lev_real = int(os.getenv("LEVERAGE", "5") or 5)
                except Exception:
                    lev_real = 5

        # derive current and bound values for clamping overrides
        trail_init = float(getattr(params, "trail_pct_init", 0.003) or 0.003)
        trail_tight = float(getattr(params, "trail_pct_tight", 0.0015) or 0.0015)
        intel_sec = int(getattr(params, "intelligence_sec", 3) or 3)

        min_trail_init = float(getattr(params, "min_trail_init", 0.0005) or 0.0005)
        max_trail_init = float(getattr(params, "max_trail_init", 0.05) or 0.05)
        min_trail_tight = float(getattr(params, "min_trail_tight", 0.0003) or 0.0003)
        max_trail_tight = float(getattr(params, "max_trail_tight", 0.03) or 0.03)
        intel_min = int(os.getenv("ADVISOR_INTEL_MIN_SEC", "1") or 1)
        intel_max = int(os.getenv("ADVISOR_INTEL_MAX_SEC", "10") or 10)
        fees_bps = float(context.get("fees_bps", 6.0))
        slip_bps = float(context.get("slip_bps", 1.0))
        total_cost_bps = fees_bps + slip_bps

        snap = {
            "side": side,
            "trail_init": trail_init,
            "trail_tight": trail_tight,
            "intel_sec": intel_sec,
            "leverage": lev_real,
            "funding": context.get("funding"),
            "open_interest": context.get("open_interest"),
            "volatility": context.get("volatility"),
            "fees_bps": fees_bps,
            "slip_bps": slip_bps,
            "total_cost_bps": total_cost_bps,
            "caution": caution,
        }
        allow, confidence, note, edge_bps, overrides = self.evaluate(snap)

        # Clamp overrides here so nothing insane slips through
        ov = dict(overrides or {})
        if "trail_pct_init" in ov:
            try:
                v = float(ov["trail_pct_init"])
                ov["trail_pct_init"] = max(min_trail_init, min(max_trail_init, v))
            except Exception:
                ov.pop("trail_pct_init", None)
        if "trail_pct_tight" in ov:
            try:
                v = float(ov["trail_pct_tight"])
                ov["trail_pct_tight"] = max(min_trail_tight, min(max_trail_tight, v))
            except Exception:
                ov.pop("trail_pct_tight", None)
        if "intelligence_sec" in ov:
            try:
                v = int(float(ov["intelligence_sec"]))
                ov["intelligence_sec"] = max(intel_min, min(intel_max, v))
            except Exception:
                ov.pop("intelligence_sec", None)

        decision = {
            "allow": True if self.mode == "warn" else bool(allow),
            "confidence": float(confidence),
            "note": note,
            "edge_bps": float(edge_bps),
            "suggested_overrides": ov,
        }

        use_trail = float(trail_init) if trail_from_strategy is None else float(trail_from_strategy)
        return side, ("advisor_warn" if not allow or confidence < 0.70 else reason), float(use_trail), decision
