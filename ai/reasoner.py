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
from typing import Any, Dict, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


class Reasoner:
    def __init__(self):
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

    def evaluate(self, snapshot: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Returns (allow, confidence, note).
        When disabled or unavailable it returns (True, 1.0, "disabled").
        """
        if not self.enabled or not self.client:
            return True, 1.0, "disabled"

        policy = (
            "Never block a trade. If leverage is high and parameters are not tightened, lower confidence and add a caution note. "
            "Shorter intelligence_sec at higher leverage; smaller (tighter) trail percentages at higher leverage. "
            "Consider funding and volatility; when mismatched, reduce confidence and add a caution."
        )
        prompt = (
            "You are a cautious trading gate. Reply as compact JSON with keys "
            '{"allow": boolean, "confidence": number 0..1, "note": string}. '
            f"Policy: {policy} Snapshot: {snapshot}"
        )

        try:
            import json
            data = json.loads(content)
            allow = bool(data.get("allow", True))
            conf = float(data.get("confidence", 0.75 if allow else 0.55))
            note = str(data.get("note", ""))
        except Exception:
            low = content.lower()
            allow = ("\"allow\": true" in low) or ("allow: true" in low) or ("allow\":true" in low)
            conf = 0.75 if allow else 0.55
            note = content

        # override: never hard-block when ADVISOR_MODE=warn
        if self.mode == "warn" and not allow:
            allow = True
            note = f"warn: {note}"

        note_oneline = " ".join(str(note).split())[:300]
        return allow, conf, note_oneline

    def decide(self, strategy, params, context: Dict[str, Any]):
        """
        Returns side_choice, reason, trail_pct_to_use, decision_dict.

        side_choice is 'long', 'short', or 'wait'.
        decision_dict includes confidence 0..1 so the orchestrator can scale stake and trail.
        """
        side = "wait"
        reason = "no_signal"
        trail_from_strategy = None
        caution = None

        # Use the strategy if available. Prefer a proper decide() method when present.
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

        # Read leverage from context (fallback to env only if missing)
        try:
            lev_real = int(context.get("leverage"))
        except Exception:
            lev_real = int(os.getenv("LEVERAGE", "5") or 5)

        # Auto-tighten when leverage is high (>=20)
        try:
            if lev_real >= 20:
                params.trail_pct_init = max(params.min_trail_init, params.trail_pct_init * 0.7)
                params.trail_pct_tight = max(params.min_trail_tight, params.trail_pct_tight * 0.7)
                params.intelligence_sec = max(2, int(params.intelligence_sec * 0.6))
        except Exception:
            pass

        # Advisor snapshot and evaluation
        snap = {
            "side": side,
            "trail_init": params.trail_pct_init,
            "trail_tight": params.trail_pct_tight,
            "intel_sec": params.intelligence_sec,
            "leverage": lev_real,
            "funding": context.get("funding"),
            "open_interest": context.get("open_interest"),
            "volatility": context.get("volatility"),
            "caution": caution,
        }
        allow, confidence, note = self.evaluate(snap)

        # Never hard-block on allow=False. Treat as warn and continue.
        if not allow:
            reason = "advisor_warn"
            decision = {"allow": True, "confidence": float(confidence), "note": f"warn: {note}"}

        # Existing hard block on very low confidence or explicit can't-manage phrasing
        txt = (note or "").lower()
        says_cant_manage = any(k in txt for k in ("cannot manage", "can't manage", "unmanageable", "do not trade", "do not proceed"))
        hard_block = (confidence < 0.30) or says_cant_manage
        if hard_block:
            reason = "advisor_warn"
            decision = {"allow": True, "confidence": float(confidence), "note": f"warn: {note}"}

        # Warn-but-allow band: 0.30 <= confidence < 0.70
        if confidence < 0.70:
            reason = "advisor_warn"
            try:
                params.trail_pct_init = max(params.min_trail_init, params.trail_pct_init * 0.9)
                params.trail_pct_tight = max(params.min_trail_tight, params.trail_pct_tight * 0.9)
                params.intelligence_sec = max(2, int(params.intelligence_sec * 0.8))
            except Exception:
                pass
            decision = {"allow": True, "confidence": float(confidence), "note": note}
        else:
            decision = {"allow": True, "confidence": float(confidence), "note": note}

        use_trail = params.trail_pct_init if trail_from_strategy is None else float(trail_from_strategy)
        return side, reason, float(use_trail), decision
