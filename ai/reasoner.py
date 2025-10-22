#!/usr/bin/env python3
# bitgetbot/ai/reasoner.py
"""
ai/reasoner.py
Optional OpenAI advisor that votes on entries and returns a confidence score.

Plain English.
This is the advisor. It can say yes or no and a confidence score based on a short snapshot. If OPENAI_API_KEY is missing or ADVISOR_ENABLED is false, it always approves so it never blocks you.

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

    def evaluate(self, snapshot: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Returns (allow, confidence, note).
        When disabled or unavailable it returns (True, 1.0, "disabled").
        """
        if not self.enabled or not self.client:
            return True, 1.0, "disabled"

        policy = (
            "Block if leverage is high and parameters are not tightened. "
            "Ensure intelligence_sec is shorter at higher leverage, trails are tighter in High volatility, "
            "and notional is within cap. If funding is extreme or volatility is mismatched with regime, lower confidence."
        )
        prompt = (
            "You are a cautious trading gate. Reply as JSON with keys allow:boolean, confidence:0..1, note:string. "
            f"Policy: {policy}\n"
            "Approve entries only when risk seems reasonable given leverage, stake, volatility and funding context. "
            "Snapshot follows as JSON:\n"
            f"{snapshot}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            allow = "true" in content.lower()
            conf = 0.75 if allow else 0.55
            return allow, conf, content[:200]
        except Exception as e:
            return True, 1.0, f"advisor_error: {e}"

    def decide(self, strategy, params, context: Dict[str, Any]):
        """
        Returns side_choice, reason, trail_pct_to_use, decision_dict.

        side_choice is 'long', 'short', or 'wait'.
        decision_dict includes confidence 0..1 so the orchestrator can scale stake and trail.
        """
        # Try to get a side from the local strategy if available
        side = "wait"
        reason = "no_signal"
        try:
            if strategy is not None:
                if hasattr(strategy, "suggest"):
                    side = strategy.suggest()
                    reason = "strategy_suggest"
                elif hasattr(strategy, "vote"):
                    v = strategy.vote()
                    # Expect v to be a dict or tuple; fall back safely
                    if isinstance(v, dict):
                        side = v.get("side", "wait")
                        reason = v.get("reason", "vote")
                    elif isinstance(v, tuple) and len(v) >= 1:
                        side = v[0] or "wait"
                        reason = "vote_tuple"
        except Exception:
            side = "wait"
            reason = "strategy_error"

        # Build snapshot for the advisor
        snap = {
            "side": side,
            "trail_init": params.trail_pct_init,
            "trail_tight": params.trail_pct_tight,
            "intel_sec": params.intelligence_sec,
            "lev": os.getenv("LEVERAGE", "5"),
            "funding": context.get("funding"),
            "open_interest": context.get("open_interest"),
            "volatility": context.get("volatility"),
        }
        allow, confidence, note = self.evaluate(snap)
        if not allow:
            side = "wait"
            reason = "advisor_block"
        decision = {"allow": allow, "confidence": float(confidence), "note": note}
        return side, reason, float(params.trail_pct_init), decision
