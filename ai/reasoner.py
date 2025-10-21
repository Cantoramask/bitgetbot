#!/usr/bin/env python3
# bitgetbot/ai/reasoner.py
"""
ai/reasoner.py
Optional OpenAI advisor that votes on entries.

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

        # snapshot expected keys
        # price, trail_init, trail_tight, intel_sec, stake, lev, vol, side
        # effective_notional, cooldowns, atrp
        policy = (
            "Block if leverage is high and parameters are not tightened. "
            "Ensure intelligence_sec is shorter at higher leverage, trails are tighter in High, "
            "and effective notional is within cap. If atrp indicates High but vol is not High, veto."
        )
        prompt = (
            "You are a cautious trading gate. Reply as JSON with keys allow:boolean, confidence:0..1, note:string. "
            f"Policy: {policy}\n"
            "Approve entries only when risk seems reasonable given leverage, notional and volatility regime. "
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
