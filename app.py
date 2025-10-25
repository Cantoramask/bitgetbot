#!/usr/bin/env python3
"""
app.py
Entry point for the Bitget bot.

What this file does:
1) Shows the exact startup prompts you requested, in the same order and format.
2) Loads defaults from environment if present so you can just press Enter.
3) Sets up a daily log file in ./logs and prints to the console.
4) Tries to run orchestrator.Orchestrator(cfg, logger). If not present yet,
   runs a harmless fallback heartbeat so you can launch today.

Plain-English definitions:
- Environment variable: a named value stored outside your code (for example in a .env file)
  that your program can read for settings like API keys or default leverage.
- API key: a pair of secret values (key and secret) from the exchange that proves who you are.
- Async means the program can wait for things like network without freezing the whole app.

Run:
    python app.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
import textwrap
import logging
import signal
from dataclasses import dataclass
from exchange.adapter import BitgetAdapter

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure local package imports (exchange/, ai/, etc.) resolve when running from repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class AppConfig:
    symbol: str
    stake_usdt: float
    leverage: int
    live: bool
    margin_mode: str
    vol_profile: str  # "auto" | "High" | "Medium" | "Low"
    takeover: bool    # whether to take over an open position if detected
    intelligence_check_sec: int = 10

    # >>> Added to satisfy orchestrator & new features (Items 1 & 10)
    max_notional_usdt: float = 5000.0          # cap for stake * leverage (read from env)
    tp1_fraction: float = 0.30                 # first partial take-profit fraction
    tp2_fraction: float = 0.30                 # second partial take-profit fraction
    context_refresh_sec: int = 30              # Item 1: funding/OI context refresh interval

    @staticmethod
    def from_inputs(
        symbol: str,
        stake_usdt: float,
        leverage: int,
        live: bool,
        margin_mode: str,
        vol_profile: str,
        takeover: bool,
        intelligence_check_sec: int | None = None,
    ) -> "AppConfig":
        # helpers for booleans from env
        def _env_bool(name: str, default: bool=False) -> bool:
            return (os.getenv(name, str(default)).strip().lower() in ("1", "true", "y", "yes", "on"))

        return AppConfig(
            symbol=_normalize_symbol(symbol or os.getenv("SYMBOL", "BTC/USDT:USDT")),
            stake_usdt=stake_usdt if stake_usdt is not None else float(os.getenv("BASE_USDT_PER_TRADE", "50")),
            leverage=leverage if leverage is not None else int(float(os.getenv("LEVERAGE", "5"))),
            live=bool(live) if live is not None else _env_bool("LIVE", False),
            margin_mode=(margin_mode or os.getenv("MARGIN_MODE", "cross")).lower(),
            vol_profile=_normalize_vol(vol_profile or os.getenv("VOL_PROFILE", "auto")),
            takeover=bool(takeover),
            intelligence_check_sec=(
                intelligence_check_sec if intelligence_check_sec is not None
                else int(float(os.getenv("INTELLIGENCE_CHECK_SEC", "10")))
            ),
            # >>> Added: pull new fields from env with safe defaults
            max_notional_usdt=float(os.getenv("RISK_MAX_NOTIONAL_DEFAULT", "5000")),
            tp1_fraction=float(os.getenv("TP1_FRACTION", "0.30")),
            tp2_fraction=float(os.getenv("TP2_FRACTION", "0.30")),
            context_refresh_sec=int(float(os.getenv("CONTEXT_REFRESH_SEC", "30"))),
        )
# ---------------------------
# Simple console + file logger
# ---------------------------
def _setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    today = dt.datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join("logs", f"journal_{today}.log")

    logger = logging.getLogger("bitgetbot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%H:%M:%S"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------
# Input helpers
# ---------------------------
def _normalize_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    # Accept "BTC", "BTC/USDT", "BTCUSDT", or "BTC/USDT:USDT"
    if s.endswith(":USDT") and "/" in s:
        return s
    if "/" in s and not s.endswith(":USDT"):
        return f"{s}:USDT"
    if s.endswith("USDT") and "/" not in s:
        base = s[:-4]
        return f"{base}/USDT:USDT"
    if s.isalpha():
        return f"{s}/USDT:USDT"
    return s or "BTC/USDT:USDT"


def _normalize_vol(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ("auto",):
        return "auto"
    if v in ("h", "high"):
        return "High"
    if v in ("l", "low"):
        return "Low"
    return "Medium"


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default if default is not None else "")


def _ask_yes_no(prompt: str, default_yes: bool | None = None) -> bool:
    # Matches the visual you want: "(Y/N)" with optional [Y] or [N]
    yn = "Y/N"
    if default_yes is True:
        prompt_full = f"{prompt} ({yn}) [Y]: "
    elif default_yes is False:
        prompt_full = f"{prompt} ({yn}) [N]: "
    else:
        prompt_full = f"{prompt} ({yn}): "
    val = input(prompt_full).strip().lower()
    if not val and default_yes is not None:
        return bool(default_yes)
    return val in ("y", "yes", "1", "true")


def _safe_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _safe_int(s: str, default: int) -> int:
    try:
        return int(float(s))
    except Exception:
        return default


# ---------------------------
# Import Orchestrator or make a safe fallback
# ---------------------------
class _FallbackOrchestrator:
    """
    If orchestrator.py is not ready yet, we provide a harmless loop that
    logs a heartbeat every few seconds so the app can be started today.
    """

    def __init__(self, cfg: AppConfig, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self._stop = asyncio.Event()

    async def run(self):
        self.logger.info("[START] Fallback mode active (orchestrator not ready). No trading will occur.")
        self.logger.info(
            f"[CFG] symbol={self.cfg.symbol} stake={self.cfg.stake_usdt:.2f}USDT lev={self.cfg.leverage} "
            f"live={self.cfg.live} margin={self.cfg.margin_mode} vol={self.cfg.vol_profile} takeover={self.cfg.takeover}"
        )
        self.logger.info("When orchestrator.py is implemented, the real trading loop will run here.")
        try:
            while not self._stop.is_set():
                self.logger.info("[HB] waiting for orchestrator module... press Ctrl+C to stop")
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        self.logger.info("[STOP] Fallback orchestrator stopped.")

    def request_stop(self):
        self._stop.set()


def _load_orchestrator(cfg: AppConfig, logger: logging.Logger):
    try:
        import importlib

        # Try the real package path first: bitgetbot/orchestrator.py
        try:
            orch_module = importlib.import_module("bitgetbot.orchestrator")
        except ModuleNotFoundError:
            # Fallback to a flat file named orchestrator.py if present
            orch_module = importlib.import_module("orchestrator")

        Orchestrator = getattr(orch_module, "Orchestrator", None)
        if Orchestrator is None:
            raise AttributeError("Orchestrator class not found in orchestrator module")
        return Orchestrator(cfg, logger)
    except Exception as e:
        logger.info(f"[INFO] Using fallback orchestrator: {e}")
        return _FallbackOrchestrator(cfg, logger)


# ---------------------------
# Graceful shutdown
# ---------------------------
def _install_signal_handlers(loop, stopper):
    def _handle_sig():
        try:
            stopper()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            # Windows may not support this fully; ignore.
            pass


# ---------------------------
# Main
# ---------------------------
async def _async_main():
    # Resolve default symbol before the takeover prompt so it can appear in parentheses
    default_symbol = _normalize_symbol(os.getenv("SYMBOL", "BTC/USDT:USDT"))
    default_stake = os.getenv("BASE_USDT_PER_TRADE", "50.0")
    default_lev = os.getenv("LEVERAGE", "5")
    default_live_env = os.getenv("LIVE", "").lower() in ("1", "true", "y", "yes")
    default_margin = os.getenv("MARGIN_MODE", "cross")
    default_vol = _normalize_vol(os.getenv("VOL_PROFILE", "auto"))

    # Detect real open positions FIRST; only ask to take over if there are any
    chosen_symbol = None
    open_positions = []
    try:
        tmp_adapter = BitgetAdapter(
            logger=logging.getLogger("bitgetbot-boot"),
            symbol=default_symbol,
            leverage=int(default_lev),
            margin_mode=default_margin,
            live=default_live_env,
        )
        await tmp_adapter.connect()
        try:
            open_positions = await tmp_adapter.get_all_open_positions()
        except Exception:
            raw = await tmp_adapter.get_open_position(default_symbol)
            open_positions = [raw] if raw else []
    except Exception:
        open_positions = []

    if not open_positions:
        takeover = False
    elif len(open_positions) == 1:
        only = open_positions[0]
        detected_sym = _normalize_symbol(str(only.get("symbol", default_symbol)))
        ans = _ask_yes_no(
            f"Open perpetual position detected ({detected_sym}). Take over management now using existing size and leverage?",
            default_yes=None
        )
        takeover = bool(ans)
        chosen_symbol = detected_sym if takeover else None
    else:
        print("Open perpetual positions detected:")
        for i, p in enumerate(open_positions, 1):
            try:
                ep = float(p.get("entry_price", 0.0))
                sz = float(p.get("size_usdt", 0.0))
                lev = int(p.get("leverage", int(default_lev)))
            except Exception:
                ep, sz, lev = 0.0, 0.0, int(default_lev)
            print(f"  {i}) {p.get('symbol')} {p.get('side')} ~{sz:.2f}USDT @ {ep:.6f} x{lev}")
        sel = input("Take over which position? Enter number or press Enter to skip: ").strip()
        if sel.isdigit():
            idx = int(sel) - 1
            if 0 <= idx < len(open_positions):
                chosen_symbol = _normalize_symbol(str(open_positions[idx].get("symbol", default_symbol)))
        takeover = bool(chosen_symbol)

    if takeover:
        # Skip ALL further prompts; use env/defaults and the detected symbol
        symbol_in = chosen_symbol or default_symbol
        live_in = True
        stake_in = default_stake
        lev_in = default_lev
        margin_in = default_margin
        vol_in = "auto"
    else:
        examples = textwrap.dedent(
            """\
            Symbol format examples:
              BTC/USDT:USDT  Bitget perpetuals
              BTC/USDT       auto adds :USDT
              BTCUSDT        converts to BTC/USDT:USDT
              BTC            converts to BTC/USDT:USDT"""
        )
        print(examples)
        symbol_in = _ask("Symbol", default_symbol)
        live_in = _ask_yes_no("Live trading?", default_yes=False if not default_live_env else True)
        stake_in = _ask("USDT margin per trade", default_stake)
        lev_in = _ask("Leverage x", default_lev)
        margin_in = _ask("Margin mode cross or isolated", default_margin)
        vol_in = "auto"

    # Build config
    cfg = AppConfig.from_inputs(
        symbol=symbol_in,
        stake_usdt=_safe_float(stake_in, float(default_stake)),
        leverage=max(1, min(100, _safe_int(lev_in, int(default_lev)))),
        live=bool(live_in),
        margin_mode=margin_in,
        vol_profile=vol_in,
        takeover=bool(takeover),
    )

    logger = _setup_logging()
    logger.info("[BOOT] Bitget bot app starting")
    logger.info(
        f"[CFG] symbol={cfg.symbol} stake={cfg.stake_usdt:.2f}USDT lev={cfg.leverage} live={cfg.live} "
        f"margin={cfg.margin_mode} vol={cfg.vol_profile} takeover={cfg.takeover}"
    )

    orchestrator = _load_orchestrator(cfg, logger)

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()

    def _stop_all():
        try:
            if hasattr(orchestrator, "request_stop"):
                orchestrator.request_stop()
        except Exception:
            pass
        stopping.set()

    _install_signal_handlers(loop, _stop_all)

    # Run orchestrator until stop requested
    main_task = asyncio.create_task(orchestrator.run())
    await stopping.wait()
    main_task.cancel()
    with contextlib_suppress(asyncio.CancelledError):
        await main_task

    logger.info("[EXIT] Bye")


class contextlib_suppress:
    """Small local suppress context to avoid importing contextlib for one use."""

    def __init__(self, *exceptions):
        self.exceptions = exceptions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self.exceptions)


def main():
    try:
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    except Exception:
        pass
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
