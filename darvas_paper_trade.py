"""Daily paper-trading tracker for the frozen Darvas V3 rules.

Put this file beside darvas_nse_backtest.py. Run it once after market close on
each NSE trading day. It makes virtual orders only; it never sends real orders.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import darvas_nse_backtest as v3

PAPER_DIR = Path("darvas_paper_output")
STATE_PATH = PAPER_DIR / "paper_state.json"
TRADES_PATH = PAPER_DIR / "paper_trades.csv"
EQUITY_PATH = PAPER_DIR / "paper_equity.csv"
INITIAL_CAPITAL = 100_000.0
HISTORY_DAYS = 420


def read_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"cash": INITIAL_CAPITAL, "positions": {}, "buys": {}, "sells": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_csv(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def close_price(data: dict[str, pd.DataFrame], ticker: str, day: pd.Timestamp) -> float:
    series = data[ticker].loc[:day, "Close"]
    return float(series.iloc[-1]) if len(series) else 0.0


def main() -> None:
    PAPER_DIR.mkdir(exist_ok=True)
    today = pd.Timestamp(date.today())
    start = today - pd.Timedelta(days=HISTORY_DAYS)
    state = read_state()
    if state.get("last_processed") == str(today.date()):
        print(f"{today.date()} was already processed. Nothing changed.")
        return

    raw = {ticker: v3.prepare(d) for ticker in v3.TICKERS if not (d := v3.load(ticker, start, today)).empty}
    benchmark = v3.load(v3.BENCHMARK, start, today)
    if benchmark.empty:
        raise RuntimeError("NIFTY data is unavailable; paper tracker stopped without changes.")
    if today not in benchmark.index:
        print("Today is not an NSE trading day, or the closing data is not available yet. Nothing changed.")
        return

    bench = benchmark["Adj Close"].astype(float)
    market_ok = bool(bench.iloc[-1] > bench.rolling(200, min_periods=200).mean().iloc[-1])
    nifty_6m = bench.pct_change(v3.RELATIVE_STRENGTH_DAYS).iloc[-1]
    if pd.isna(nifty_6m):
        print("Not enough NIFTY history for the 200-day/6-month filters. Nothing changed.")
        return

    # Execute virtual orders queued from the previous trading day's close.
    for ticker, reason in list(state["sells"].items()):
        if ticker not in state["positions"] or ticker not in raw or today not in raw[ticker].index:
            continue
        position = state["positions"].pop(ticker)
        exit_price = float(raw[ticker].at[today, "Open"]) * (1 - v3.SLIPPAGE_RATE)
        gross = position["shares"] * exit_price
        cost = gross * v3.BROKERAGE_RATE
        state["cash"] += gross - cost
        net_pnl = gross - cost - position["shares"] * position["entry"] - position["entry_cost"]
        append_csv(TRADES_PATH, {
            "ticker": ticker, "entry_date": position["entry_date"], "exit_date": str(today.date()),
            "shares": position["shares"], "entry_price": position["entry"], "exit_price": exit_price,
            "initial_stop": position["initial_stop"], "exit_reason": reason,
            "costs": position["entry_cost"] + cost, "net_pnl": net_pnl,
            "return_pct": net_pnl / (position["shares"] * position["entry"] + position["entry_cost"]),
        })
    state["sells"] = {}

    for ticker, order in list(state["buys"].items()):
        if ticker in state["positions"] or ticker not in raw or today not in raw[ticker].index:
            continue
        if len(state["positions"]) >= v3.MAX_POSITIONS:
            continue
        fill = float(raw[ticker].at[today, "Open"]) * (1 + v3.SLIPPAGE_RATE)
        risk = fill - order["stop"]
        if risk <= 0:
            continue
        positions_value = sum(p["shares"] * close_price(raw, symbol, today) for symbol, p in state["positions"].items())
        equity = state["cash"] + positions_value
        shares = min(
            math.floor(equity * v3.RISK_PER_TRADE / risk),
            math.floor(equity * v3.MAX_POSITION_FRACTION / (fill * (1 + v3.BROKERAGE_RATE))),
            math.floor(state["cash"] / (fill * (1 + v3.BROKERAGE_RATE))),
        )
        if shares >= 1:
            entry_cost = shares * fill * v3.BROKERAGE_RATE
            state["cash"] -= shares * fill + entry_cost
            state["positions"][ticker] = {
                "shares": shares, "entry_date": str(today.date()), "entry": fill, "entry_cost": entry_cost,
                "initial_stop": order["stop"], "trailing_stop": order["stop"], "high_close": fill,
            }
    state["buys"] = {}

    # Evaluate closing stops, creating virtual sells for tomorrow's open.
    for ticker, position in state["positions"].items():
        if ticker not in raw or today not in raw[ticker].index:
            continue
        close = float(raw[ticker].at[today, "Close"])
        position["high_close"] = max(position["high_close"], close)
        position["trailing_stop"] = max(position["initial_stop"], position["high_close"] * (1 - v3.TRAILING_STOP_PCT))
        if close <= position["trailing_stop"]:
            state["sells"][ticker] = "trailing_stop" if position["trailing_stop"] > position["initial_stop"] else "initial_stop"

    # Scan today's completed candles and queue qualifying entries for tomorrow.
    if market_ok:
        available = v3.MAX_POSITIONS - len(state["positions"])
        candidates = []
        for ticker, data in raw.items():
            if ticker in state["positions"] or today not in data.index:
                continue
            i = data.index.get_loc(today)
            darvas_box = v3.box(data, i)
            if not darvas_box or i < v3.RELATIVE_STRENGTH_DAYS:
                continue
            stock_6m = float(data["Close"].iloc[i] / data["Close"].iloc[i - v3.RELATIVE_STRENGTH_DAYS] - 1)
            row = data.iloc[i]
            if stock_6m > nifty_6m and row.Close > darvas_box[0] * (1 + v3.BREAKOUT_BUFFER) and pd.notna(row.volume_avg) and row.Volume >= row.volume_avg * v3.VOLUME_MULTIPLE:
                candidates.append((float(row.Volume / row.volume_avg), ticker, darvas_box))
        for _, ticker, (top, bottom) in sorted(candidates, reverse=True)[:max(available, 0)]:
            state["buys"][ticker] = {"top": top, "bottom": bottom, "stop": bottom * (1 - v3.INITIAL_STOP_BUFFER)}

    positions_value = sum(p["shares"] * close_price(raw, ticker, today) for ticker, p in state["positions"].items())
    equity = state["cash"] + positions_value
    append_csv(EQUITY_PATH, {"date": str(today.date()), "cash": state["cash"], "positions_value": positions_value, "equity": equity, "open_positions": len(state["positions"]), "market_ok": market_ok})
    state["last_processed"] = str(today.date())
    save_state(state)

    print(f"Paper-trading update for {today.date()}")
    print(f"NIFTY regime: {'ON' if market_ok else 'OFF'} | Virtual equity: ₹{equity:,.2f}")
    print(f"Open positions: {len(state['positions'])} | Buys queued for next open: {list(state['buys'])}")
    print(f"Sells queued for next open: {list(state['sells'])}")
    print(f"Files saved in: {PAPER_DIR.resolve()}")


if __name__ == "__main__":
    main()
