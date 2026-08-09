"""NSE Darvas Box strategy backtest using Yahoo Finance.

Version 3: Darvas entries are allowed only when NIFTY is above its 200-day
moving average and the stock has outperformed NIFTY over six months. Yearly
return reporting compounds exactly to total return.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ============================ CONFIG ========================================
BENCHMARK = "^NSEI"
YEARS_OF_HISTORY = 2
END_DATE = "2026-08-07"
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.01
MAX_POSITIONS = 5
MAX_POSITION_FRACTION = 0.25

LOOKBACK_HIGH_DAYS = 20
CONFIRMATION_DAYS = 3
BREAKOUT_BUFFER = 0.001
VOLUME_LOOKBACK_DAYS = 20
VOLUME_MULTIPLE = 1.50
RELATIVE_STRENGTH_DAYS = 126  # approximately six months of trading days
MIN_BOX_HEIGHT_PCT, MAX_BOX_HEIGHT_PCT = 0.02, 0.25

INITIAL_STOP_BUFFER = 0.002
TRAILING_STOP_PCT = 0.10
BROKERAGE_RATE = 0.0003
SLIPPAGE_RATE = 0.0005
REFRESH_CACHE = False

CACHE_DIR = Path("data_cache")
OUTPUT_DIR = Path("darvas_output")


def load_nifty500_tickers():
    """Load Nifty 500 symbols and convert them to Yahoo Finance format."""
    csv_path = Path("nifty500.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "\nERROR: nifty500.csv was not found.\n"
            "Put it in the same folder as darvas_nse_backtest.py.\n"
            f"Expected location: {csv_path.resolve()}"
        )

    print(f"\nReading Nifty 500 file: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    symbol_column = next(
        (column for column in df.columns if str(column).strip().lower() == "symbol"),
        None,
    )
    if symbol_column is None:
        raise ValueError(f"Could not find a Symbol column. Columns: {list(df.columns)}")

    symbols = (
        df[symbol_column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    symbols = symbols[(symbols != "") & (symbols != "NAN") & (symbols != "SYMBOL")]
    tickers = sorted(
        set(symbol if symbol.endswith(".NS") else f"{symbol}.NS" for symbol in symbols)
    )
    if not tickers:
        raise ValueError("No valid stock symbols were found in nifty500.csv.")

    print(f"Loaded {len(tickers)} Nifty 500 symbols.")
    return tickers


TICKERS = load_nifty500_tickers()


@dataclass
class Position:
    shares: int
    entry_date: pd.Timestamp
    entry: float
    entry_cost: float
    initial_stop: float
    trailing_stop: float
    high_close: float
    top: float
    bottom: float


def name(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace(".", "_")


def f(value) -> float:
    return float(value) if pd.notna(value) and np.isfinite(value) else 0.0


def load(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Read the local cache, otherwise download from Yahoo Finance."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / (name(ticker) + ".csv")
    required_columns = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

    if path.exists() and not REFRESH_CACHE:
        try:
            data = pd.read_csv(path, index_col="Date", parse_dates=True)
            complete = data.index.min() <= start and data.index.max() >= end - pd.Timedelta(days=7)
            if required_columns <= set(data.columns) and complete:
                print(f"  {ticker}: cached")
                return data.loc[start:end].copy()
        except (OSError, ValueError, KeyError) as error:
            print(f"  {ticker}: bad cache ignored ({error})")

    print(f"  {ticker}: downloading")
    try:
        data = yf.download(
            ticker,
            start=start.date(),
            end=(end + pd.Timedelta(days=1)).date(),
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data.index.name = "Date"
        if data.empty or not required_columns <= set(data.columns):
            raise ValueError("Yahoo returned no complete OHLCV history")
        data = data[~data.index.duplicated(keep="last")].sort_index()
        data.to_csv(path)
        time.sleep(0.15)
        return data.loc[start:end].copy()
    except Exception as error:
        print(f"  WARNING: {ticker} skipped ({type(error).__name__}: {error})")
        return pd.DataFrame()


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy().replace([np.inf, -np.inf], np.nan)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    data = data[
        (data.Open > 0)
        & (data.High > 0)
        & (data.Low > 0)
        & (data.Close > 0)
        & (data.Volume >= 0)
    ]
    # These shifted fields explicitly exclude today's data.
    data["prior_high"] = (
        data.High.rolling(LOOKBACK_HIGH_DAYS, min_periods=LOOKBACK_HIGH_DAYS).max().shift(1)
    )
    data["volume_avg"] = (
        data.Volume.rolling(VOLUME_LOOKBACK_DAYS, min_periods=VOLUME_LOOKBACK_DAYS).mean().shift(1)
    )
    return data


def box(data: pd.DataFrame, i: int) -> Optional[tuple[float, float]]:
    """Build a Darvas box using only completed candles before today."""
    start = i - CONFIRMATION_DAYS
    if start < 0:
        return None

    top = data["prior_high"].iloc[start]
    if pd.isna(top) or top <= 0:
        return None

    confirmation_window = data.iloc[start:i]
    if confirmation_window["High"].max() > top:
        return None

    bottom = float(confirmation_window["Low"].min())
    height = (top - bottom) / top
    if bottom > 0 and MIN_BOX_HEIGHT_PCT <= height <= MAX_BOX_HEIGHT_PCT:
        return float(top), bottom
    return None


def dd(series: pd.Series) -> float:
    return float((series / series.cummax() - 1).min()) if len(series) else 0.0


def sharpe(series: pd.Series) -> float:
    returns = series.pct_change().dropna()
    if len(returns) <= 1 or not returns.std(ddof=1):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))


def backtest(
    data: Dict[str, pd.DataFrame],
    days: pd.DatetimeIndex,
    market_ok: pd.Series,
    nifty_return_6m: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash, positions, buys, sells, trades, curve = INITIAL_CAPITAL, {}, {}, {}, [], []

    def sell(ticker: str, day: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal cash
        position = positions.pop(ticker)
        exit_price = price * (1 - SLIPPAGE_RATE)
        gross = position.shares * exit_price
        exit_cost = gross * BROKERAGE_RATE
        cash += gross - exit_cost
        net = gross - exit_cost - position.shares * position.entry - position.entry_cost
        trades.append(
            dict(
                ticker=ticker,
                entry_date=position.entry_date.date(),
                exit_date=day.date(),
                shares=position.shares,
                entry_price=position.entry,
                exit_price=exit_price,
                box_top=position.top,
                box_bottom=position.bottom,
                initial_stop=position.initial_stop,
                exit_reason=reason,
                gross_pnl=gross - position.shares * position.entry,
                costs=position.entry_cost + exit_cost,
                net_pnl=net,
                return_pct=net / (position.shares * position.entry + position.entry_cost),
            )
        )

    for day in days:
        # Orders from yesterday's close execute at today's open.
        for ticker, reason in list(sells.items()):
            if ticker in positions and day in data[ticker].index:
                sell(ticker, day, f(data[ticker].at[day, "Open"]), reason)
            sells.pop(ticker, None)

        for ticker, order in list(buys.items()):
            buys.pop(ticker, None)
            if ticker in positions or len(positions) >= MAX_POSITIONS or day not in data[ticker].index:
                continue
            fill = f(data[ticker].at[day, "Open"]) * (1 + SLIPPAGE_RATE)
            risk = fill - order["stop"]
            if risk <= 0:
                continue
            value = cash + sum(
                position.shares * f(data[symbol].loc[:day, "Close"].iloc[-1])
                for symbol, position in positions.items()
                if len(data[symbol].loc[:day])
            )
            shares = min(
                math.floor(value * RISK_PER_TRADE / risk),
                math.floor(value * MAX_POSITION_FRACTION / (fill * (1 + BROKERAGE_RATE))),
                math.floor(cash / (fill * (1 + BROKERAGE_RATE))),
            )
            if shares < 1:
                continue
            entry_cost = shares * fill * BROKERAGE_RATE
            cash -= shares * fill + entry_cost
            positions[ticker] = Position(
                shares, day, fill, entry_cost, order["stop"], order["stop"], fill,
                order["top"], order["bottom"],
            )

        # Close-based checks create orders for the next session only.
        for ticker, position in list(positions.items()):
            if day not in data[ticker].index:
                continue
            close = f(data[ticker].at[day, "Close"])
            position.high_close = max(position.high_close, close)
            position.trailing_stop = max(
                position.initial_stop,
                position.high_close * (1 - TRAILING_STOP_PCT),
            )
            if close <= position.trailing_stop:
                sells[ticker] = (
                    "trailing_stop"
                    if position.trailing_stop > position.initial_stop
                    else "initial_stop"
                )

        # V2: new entries are allowed only above NIFTY's 200-day moving average.
        available = MAX_POSITIONS - len(positions) - len(buys)
        if available > 0 and bool(market_ok.get(day, False)):
            candidates = []
            for ticker, ticker_data in data.items():
                if ticker in positions or ticker in buys or day not in ticker_data.index:
                    continue
                i = ticker_data.index.get_loc(day)
                darvas_box = box(ticker_data, i)
                if not darvas_box:
                    continue
                row = ticker_data.iloc[i]

                # V3: the stock must beat NIFTY over the previous six months.
                if i < RELATIVE_STRENGTH_DAYS:
                    continue
                stock_return_6m = (
                    ticker_data["Close"].iloc[i]
                    / ticker_data["Close"].iloc[i - RELATIVE_STRENGTH_DAYS]
                    - 1
                )
                benchmark_return_6m = nifty_return_6m.get(day, np.nan)
                if pd.isna(benchmark_return_6m) or stock_return_6m <= benchmark_return_6m:
                    continue

                if (
                    row.Close > darvas_box[0] * (1 + BREAKOUT_BUFFER)
                    and pd.notna(row.volume_avg)
                    and row.Volume >= row.volume_avg * VOLUME_MULTIPLE
                ):
                    candidates.append((float(row.Volume / row.volume_avg), ticker, darvas_box))

            for _, ticker, (top, bottom) in sorted(candidates, reverse=True)[:available]:
                buys[ticker] = {
                    "top": top,
                    "bottom": bottom,
                    "stop": bottom * (1 - INITIAL_STOP_BUFFER),
                }

        positions_value = sum(
            position.shares * f(data[ticker].loc[:day, "Close"].iloc[-1])
            for ticker, position in positions.items()
            if len(data[ticker].loc[:day])
        )
        curve.append(
            dict(
                date=day,
                cash=cash,
                positions_value=positions_value,
                equity=cash + positions_value,
                open_positions=len(positions),
            )
        )

    final = days[-1]
    for ticker in list(positions):
        sell(ticker, final, f(data[ticker].loc[:final, "Close"].iloc[-1]), "end_of_test")

    equity = pd.DataFrame(curve).set_index("date")
    equity.iloc[-1, equity.columns.get_loc("cash")] = cash
    equity.iloc[-1, equity.columns.get_loc("positions_value")] = 0
    equity.iloc[-1, equity.columns.get_loc("equity")] = cash
    equity.iloc[-1, equity.columns.get_loc("open_positions")] = 0
    return pd.DataFrame(trades), equity


def main() -> None:
    if INITIAL_CAPITAL <= 0 or not 0 < RISK_PER_TRADE <= 1 or MAX_POSITIONS < 1:
        raise ValueError("Check capital/risk/position settings")

    end = pd.Timestamp(END_DATE) if END_DATE else pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=YEARS_OF_HISTORY) - pd.Timedelta(days=90)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Getting Yahoo data: {start.date()} through {end.date()}")

    data = {
        ticker: prepare(raw)
        for ticker in TICKERS
        if not (raw := load(ticker, start, end)).empty
    }
    if not data:
        raise RuntimeError("No usable stock data. Check ticker names and connection.")

    days = pd.DatetimeIndex(sorted(set().union(*(raw.index for raw in data.values()))))
    days = days[(days >= end - pd.DateOffset(years=YEARS_OF_HISTORY)) & (days <= end)]
    if len(days) < 30:
        raise RuntimeError("Insufficient trading history")

    benchmark_data = load(BENCHMARK, start, end)
    bench = (
        benchmark_data.reindex(days)["Adj Close"].ffill().dropna()
        if not benchmark_data.empty
        else pd.Series(dtype=float)
    )

    # Today's close decides eligibility; any entry fills at tomorrow's open.
    nifty_sma_200 = bench.rolling(200, min_periods=200).mean()
    market_ok = (bench > nifty_sma_200).reindex(days).fillna(False)
    nifty_return_6m = bench.pct_change(RELATIVE_STRENGTH_DAYS)

    trades, equity = backtest(data, days, market_ok, nifty_return_6m)
    bcurve = bench / bench.iloc[0] * INITIAL_CAPITAL if len(bench) else pd.Series(dtype=float)
    if len(bcurve):
        equity["nifty_equity"] = bcurve.reindex(equity.index).ffill()

    final = float(equity.equity.iloc[-1])
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    gross_profit = trades.loc[trades.net_pnl > 0, "net_pnl"].sum() if len(trades) else 0
    gross_loss = -trades.loc[trades.net_pnl < 0, "net_pnl"].sum() if len(trades) else 0
    streak = best_streak = 0
    for pnl in trades.net_pnl if len(trades) else []:
        streak = streak + 1 if pnl < 0 else 0
        best_streak = max(best_streak, streak)

    metrics = {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": final,
        "total_return": final / INITIAL_CAPITAL - 1,
        "cagr": (final / INITIAL_CAPITAL) ** (1 / years) - 1,
        "trades": len(trades),
        "win_rate": float((trades.net_pnl > 0).mean()) if len(trades) else 0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "max_drawdown": dd(equity.equity),
        "sharpe": sharpe(equity.equity),
        "max_losing_streak": best_streak,
        "market_filter": "NIFTY > 200-day moving average; stock 6-month return > NIFTY",
        "market_eligible_days": int(market_ok.sum()),
        "tickers_with_data": list(data),
    }
    if len(bcurve):
        metrics.update(
            nifty_final_value=float(bcurve.iloc[-1]),
            nifty_total_return=float(bcurve.iloc[-1] / INITIAL_CAPITAL - 1),
            nifty_cagr=float((bcurve.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1),
        )

    # Returns use the prior year-end as the next year's starting value.
    strategy_year_end = equity["equity"].resample("YE").last()
    strategy_year_start = strategy_year_end.shift(1).fillna(INITIAL_CAPITAL)
    yearly = pd.DataFrame({
        "strategy_return": strategy_year_end / strategy_year_start - 1
    })
    if len(bcurve):
        nifty_year_end = bcurve.resample("YE").last()
        nifty_year_start = nifty_year_end.shift(1).fillna(INITIAL_CAPITAL)
        yearly["nifty_return"] = nifty_year_end / nifty_year_start - 1

    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    equity.to_csv(OUTPUT_DIR / "equity_curve.csv")
    yearly.to_csv(OUTPUT_DIR / "yearly_returns.csv")
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2)

    print("\nDARVAS BACKTEST RESULTS")
    percent_metrics = {
        "total_return", "cagr", "win_rate", "max_drawdown",
        "nifty_total_return", "nifty_cagr",
    }
    for key, value in metrics.items():
        if isinstance(value, float) and key in percent_metrics:
            print(f"{key:22}: {value:.2%}")
        else:
            print(f"{key:22}: {value}")
    print(f"\nSaved output files to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
        sys.exit(130)
    except Exception as error:
        print(f"Fatal error: {type(error).__name__}: {error}")
        sys.exit(1)
