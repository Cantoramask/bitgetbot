#!/usr/bin/env python3
import os
import logging
import time

# Force visible console output for this test
os.environ["LOG_MINIMAL"] = "false"
os.environ["DECISION_INTERVAL_SEC"] = "20"
os.environ["HEARTBEAT_INTERVAL_SEC"] = "20"

from journal_logger import JournalLogger  # uses your edited file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
pylog = logging.getLogger("test")
jl = JournalLogger(pylog)

print("== send rapid 'wait' decisions: only the first should print, then silence for ~20s ==")
for i in range(8):
    jl.decision(side="wait", reason="advisor_block", trail_pct=0.003, cautions=f"conf=0.8 high lev test i={i}")
    time.sleep(0.5)

print("== send an action: should print immediately even inside the 20s window ==")
jl.trade_open("TEST/USDT", "long", 50.0, 2.50, 10, reason="demo")

print("== sleep 21s and send one more 'wait' decision: should print again after window ==")
time.sleep(21)
jl.decision(side="wait", reason="advisor_block", trail_pct=0.003, cautions="conf=0.7 end-of-window")
print("== done ==")