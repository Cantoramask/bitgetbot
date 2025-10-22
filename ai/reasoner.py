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

        # Build a compact, deterministic prompt. Snapshot may include leverage, trails, intel_sec, funding, oi, vol.
        policy = (
            "Block if leverage is high and parameters are not tightened. "
            "Shorter intelligence_sec at higher leverage; smaller (tighter) trail percentages at higher leverage. "
            "Consider funding and volatility; return cautious when mismatched."
        )
        prompt = (
            "You are a cautious trading gate. Reply as compact JSON with keys "
            '{"allow": boolean, "confidence": number 0..1, "note": string}. '
            f"Policy: {policy} Snapshot: {snapshot}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            # Try to parse JSON; fall back to heuristics if needed.
            try:
                import json  # local import to keep header unchanged
                data = json.loads(content)
                allow = bool(data.get("allow", True))
                conf = float(data.get("confidence", 0.75 if allow else 0.55))
                note = str(data.get("note", ""))
            except Exception:
                low = content.lower()
                allow = "allow\"?: ?true" in low or "\"allow\": true" in low or "allow: true" in low
                conf = 0.75 if allow else 0.55
                note = content
            # One-line, trimmed note for logs
            note_oneline = " ".join(str(note).split())[:300]
            return allow, conf, note_oneline
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
                    if isinstance(v, dict):
                        side = v.get("side", "wait")
                        reason = v.get("reason", "vote")
                    elif isinstance(v, tuple) and len(v) >= 1:
                        side = v[0] or "wait"
                        reason = "vote_tuple"
        except Exception:
            side = "wait"
            reason = "strategy_error"

        # Read the real leverage from context (fallback to env only if missing)
        try:
            lev_real = int(context.get("leverage"))  # provided by orchestrator
        except Exception:
            lev_real = int(os.getenv("LEVERAGE", "5") or 5)

        # Auto-tighten when leverage is high (>=20): smaller trails, faster intelligence cycle.
        try:
            if lev_real >= 20:
                params.trail_pct_init = max(params.min_trail_init, params.trail_pct_init * 0.7)
                params.trail_pct_tight = max(params.min_trail_tight, params.trail_pct_tight * 0.7)
                params.intelligence_sec = max(2, int(params.intelligence_sec * 0.6))
        except Exception:
            pass

        # Build snapshot for the advisor (include real leverage)
        snap = {
            "side": side,
            "trail_init": params.trail_pct_init,
            "trail_tight": params.trail_pct_tight,
            "intel_sec": params.intelligence_sec,
            "leverage": lev_real,
            "funding": context.get("funding"),
            "open_interest": context.get("open_interest"),
            "volatility": context.get("volatility"),
        }
        allow, confidence, note = self.evaluate(snap)

        # Only block if the advisor is VERY unsure (<0.30) or explicitly says it can't manage.
        txt = (note or "").lower()
        says_cant_manage = any(k in txt for k in ("cannot manage", "can't manage", "unmanageable", "do not trade", "do not proceed"))
        hard_block = (confidence < 0.30) or says_cant_manage

        if hard_block:
            side = "wait"
            reason = "advisor_block"
            decision = {"allow": False, "confidence": float(confidence), "note": note}
            return side, reason, float(params.trail_pct_init), decision

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

        return side, reason, float(params.trail_pct_init), decision

        # Warn-but-allow band: 0.30 <= confidence < 0.70
        if confidence < 0.70:
            reason = "advisor_warn"
            # Auto-tighten when advisor is uneasy
            tighten = 0.7  # 30% tighter
            try:
                params.trail_pct_init = max(params.min_trail_init, params.trail_pct_init * tighten)
                params.trail_pct_tight = max(params.min_trail_tight, params.trail_pct_tight * tighten)
                params.intelligence_sec = max(5, int(params.intelligence_sec * 0.6))
            except Exception:
                pass
            decision = {"allow": True, "confidence": float(confidence), "note": note}
        else:
            # Confident approval
            decision = {"allow": True, "confidence": float(confidence), "note": note}

        return side, reason, float(params.trail_pct_init), decision
