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

        # --- new: tunable edge gate knobs via env ---
        # All values are basis points unless marked otherwise.
        self.green_hurdle_bps = float(os.getenv("ADVISOR_GREEN_BPS", "5.0"))            # allow if net >= +0.05%
        self.grey_low_bps = float(os.getenv("ADVISOR_GREY_LOW_BPS", "-10.0"))           # grey if net in [-0.10%, +0.05%)
        self.grey_high_bps = float(os.getenv("ADVISOR_GREY_HIGH_BPS", "5.0"))
        self.grey_stake_mult = float(os.getenv("ADVISOR_GREY_STAKE_MULT", "0.35"))      # suggest 35% size in grey zone
        self.grey_trail_tighten = float(os.getenv("ADVISOR_GREY_TIGHTEN_PCT", "0.30"))  # tighten trails by 30% in grey

        # funding modelling based on expected hold time, bounded
        self.exp_hold_hours = float(os.getenv("ADVISOR_EXPECTED_HOLD_HOURS", "1.0"))
        self.max_hold_hours = float(os.getenv("ADVISOR_MAX_HOLD_HOURS", "2.0"))

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

    def _funding_bps_estimate(self, funding_rate_per_hour: Optional[float]) -> float:
        """
        Convert a per-hour funding rate to basis points for a bounded expected hold.
        funding_rate_per_hour is a decimal, for example 0.0001 means 1 bp per hour.
        """
        if funding_rate_per_hour is None:
            return 0.0
        hold = max(0.0, min(self.exp_hold_hours, self.max_hold_hours))
        # Use absolute value for a conservative cost estimate
        return abs(funding_rate_per_hour) * hold * 10000.0

    # ---------- public API ----------

    def evaluate(self, snapshot: Dict[str, Any]) -> Tuple[bool, float, str, float, Dict[str, float]]:
        """
        Returns (allow, confidence, note, edge_bps, suggested_overrides)

        When disabled or unavailable it returns a conservative pass:
        (True, 0.75, "disabled", 0.0, {}).
        """
        if not self.enabled or not self.client:
            return True, 0.75, "disabled", 0.0, {}

        fees_bps = float(snapshot.get("fees_bps", 6.0))
        slip_bps = float(snapshot.get("slip_bps", 1.0))
        funding_rate = snapshot.get("funding")  # per hour as decimal
        funding_bps = self._funding_bps_estimate(funding_rate)
        total_cost_bps = fees_bps + slip_bps + funding_bps

        policy = (
            "You are a cautious trading gate. Estimate the expected gross edge in basis points "
            "from this snapshot and return STRICT JSON. Consider leverage, funding direction, "
            "open interest, volatility regime, and provided trail parameters. Do not include fees "
            "or slippage or funding in your edge_bps. Only output JSON with keys: "
            '{"allow": boolean, "confidence": number, "edge_bps": number, '
            '"note": string, "overrides": {"trail_pct_init": number, '
            '"trail_pct_tight": number, "intelligence_sec": integer}}. '
            "Confidence is 0..1. If uncertain, lower confidence rather than forcing allow."
        )
        prompt = f"{policy} Snapshot: {snapshot} total_cost_bps_hint={total_cost_bps:.2f}"

        ok = False
        content = None
        err_note = ""
        for _ in range(self.max_retries + 1):
            ok, content, err_note = self._safe_llm(prompt, self.req_timeout_sec)
            if ok and content:
                break

        if not ok or not content:
            return False, 0.60, err_note or "advisor_timeout", 0.0, {}

        try:
            data = json.loads(content)
            allow_llm = bool(data.get("allow", True))
            conf = float(data.get("confidence", 0.65))
            edge_gross_bps = float(data.get("edge_bps", 0.0))
            note = str(data.get("note", ""))
            overrides = data.get("overrides") or {}
            safe_overrides: Dict[str, float] = {}
            if "trail_pct_init" in overrides:
                safe_overrides["trail_pct_init"] = float(overrides["trail_pct_init"])
            if "trail_pct_tight" in overrides:
                safe_overrides["trail_pct_tight"] = float(overrides["trail_pct_tight"])
            if "intelligence_sec" in overrides:
                safe_overrides["intelligence_sec"] = int(overrides["intelligence_sec"])
        except Exception:
            if self.log:
                try:
                    self.log.info(f"[AI] parse_fallback raw={str(content)[:200]!r}")
                except Exception:
                    pass
            allow_llm = False
            conf = 0.60
            edge_gross_bps = 0.0
            note = "parse_fallback"
            safe_overrides = {}

        # --- new: local economic gate on NET edge (gross minus realistic costs) ---
        net_edge_bps = edge_gross_bps - total_cost_bps

        decision_note = note
        allow_final = False
        # default no extra suggestions
        ov_out: Dict[str, float] = dict(safe_overrides)

        if net_edge_bps >= self.green_hurdle_bps:
            allow_final = True
            decision_note = f"green net={net_edge_bps:.2f}bps gross={edge_gross_bps:.2f} cost={total_cost_bps:.2f} {note}"
        elif self.grey_low_bps <= net_edge_bps < self.grey_high_bps:
            allow_final = True
            # tighten trails and suggest small stake for a probe
            try:
                trail_init = float(snapshot.get("trail_init", 0.002))
                trail_tight = float(snapshot.get("trail_tight", 0.001))
                tighten = max(0.0, min(self.grey_trail_tighten, 0.9))
                ov_out["trail_pct_init"] = max(0.0003, trail_init * (1.0 - tighten))
                ov_out["trail_pct_tight"] = max(0.0002, trail_tight * (1.0 - tighten))
            except Exception:
                pass
            ov_out["stake_mult"] = float(self.grey_stake_mult)  # orchestrator may apply if supported
            conf = min(conf, 0.75)
            decision_note = f"grey probe net={net_edge_bps:.2f}bps gross={edge_gross_bps:.2f} cost={total_cost_bps:.2f} {note}"
        else:
            allow_final = False
            decision_note = f"red net={net_edge_bps:.2f}bps gross={edge_gross_bps:.2f} cost={total_cost_bps:.2f} {note}"

        # Respect warn mode semantics. In warn mode we do not force allow when advisor says no,
        # but our local economic gate still determines the recommendation.
        if self.mode == "warn" and not allow_final:
            decision_note = f"warn: {decision_note}"

        note_oneline = " ".join(str(decision_note).split())[:300]
        # Return NET edge in edge_bps so the log reflects actionable economics.
        return allow_final, float(conf), note_oneline, float(net_edge_bps), ov_out

    def decide(self, strategy, params, context: Dict[str, Any]):
        """
        Returns side_choice, reason, trail_pct_to_use, decision_dict.

        side_choice is 'long', 'short', or 'wait'.
        decision_dict includes confidence 0..1, edge_bps, and suggested parameter overrides.
        This function is PURE: it does not mutate params. The orchestrator may apply
        suggested_overrides with its own bounds/decay/reset logic.
        """
        side = "wait"
        reason = "no_signal"
        trail_from_strategy = None
        caution = None

        # Strategy vote if available
        try:
            if strategy is not None:
                side = strategy.decide_side() or "wait"
                trail_from_strategy = getattr(strategy, "trail_pct", None)
        except Exception:
            side = "wait"

        # Build snapshot for the LLM gate
        snapshot = {
            "side": side,
            "trail_init": float(params.trail_pct_init),
            "trail_tight": float(params.trail_pct_tight),
            "intel_sec": int(params.intelligence_sec),
            "lev": int(context.get("actual_leverage") or context.get("leverage") or 1),
            "funding": context.get("funding"),
            "open_interest": context.get("open_interest"),
            "volatility": context.get("volatility"),
            "fees_bps": float(context.get("fees_bps", 6.0)),
            "slip_bps": float(context.get("slip_bps", 1.0)),
        }

        allow, confidence, note, edge_bps, ov = self.evaluate(snapshot)

        decision = {
            "allow": bool(allow),  # warn mode no longer forces allow=True
            "confidence": float(confidence),
            "note": note,
            "edge_bps": float(edge_bps),
            "suggested_overrides": ov,
        }

        use_trail = float(params.trail_pct_init) if trail_from_strategy is None else float(trail_from_strategy)
        return side, ("advisor_warn" if (not allow or confidence < 0.70) else reason), float(use_trail), decision
