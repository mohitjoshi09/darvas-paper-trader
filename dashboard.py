"""Read-only dashboard for the Darvas paper-trading ledger.

Run locally with: streamlit run dashboard.py
The GitHub Actions workflow commits darvas_paper_output/ after each scan, so a
Streamlit deployment connected to the same repository refreshes from the
latest virtual portfolio state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from io import BytesIO

import pandas as pd
import requests
import streamlit as st


PAPER_DIR = Path("darvas_paper_output")
STATE_PATH = PAPER_DIR / "paper_state.json"
EQUITY_PATH = PAPER_DIR / "paper_equity.csv"
TRADES_PATH = PAPER_DIR / "paper_trades.csv"
INITIAL_CAPITAL = 100_000.0
DEFAULT_REPOSITORY = "mohitjoshi09/darvas-paper-trader"
DEFAULT_BRANCH = "main"


def money(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"₹{value:,.2f}"


def local_file(name: str) -> bytes | None:
    path = PAPER_DIR / name
    return path.read_bytes() if path.exists() else None


@st.cache_data(ttl=60, show_spinner=False)
def github_file(repository: str, branch: str, name: str, token: str) -> bytes | None:
    url = f"https://raw.githubusercontent.com/{repository}/{branch}/darvas_paper_output/{name}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def csv_from_bytes(data: bytes | None, date_columns: list[str]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    return pd.read_csv(BytesIO(data), parse_dates=date_columns)


def load_data(source: str, repository: str, branch: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    token = os.environ.get("DARVAS_GITHUB_TOKEN", "")
    read_file = local_file if source == "Local folder" else lambda name: github_file(repository, branch, name, token)
    state_bytes = read_file("paper_state.json")
    if not state_bytes:
        raise FileNotFoundError("paper_state.json is not available from the selected data source.")
    state = json.loads(state_bytes.decode("utf-8"))
    equity = csv_from_bytes(read_file("paper_equity.csv"), ["date"])
    trades = csv_from_bytes(read_file("paper_trades.csv"), ["entry_date", "exit_date"])
    return state, equity, trades


def positions_table(state: dict) -> pd.DataFrame:
    rows = []
    for ticker, position in state.get("positions", {}).items():
        entry = float(position.get("entry", 0.0))
        shares = int(position.get("shares", 0))
        last_price = position.get("last_price")
        market_value = position.get("market_value")
        invested = entry * shares + float(position.get("entry_cost", 0.0))
        rows.append({
            "Ticker": ticker,
            "Entry date": position.get("entry_date"),
            "Shares": shares,
            "Entry": entry,
            "Last close": last_price,
            "Market value": market_value,
            "Unrealised P&L": position.get("unrealized_pnl"),
            "Unrealised return": position.get("unrealized_return_pct"),
            "Trailing stop": position.get("trailing_stop"),
            "Initial stop": position.get("initial_stop"),
            "Invested": invested,
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Darvas Paper Trader", page_icon="📈", layout="wide")
st.title("Darvas Paper Trader")
st.caption("Virtual portfolio only — no orders are placed from this dashboard.")

with st.sidebar:
    st.header("Data source")
    source = st.radio("Read paper ledger from", ["GitHub repository", "Local folder"])
    repository = st.text_input("GitHub repository", value=os.environ.get("DARVAS_GITHUB_REPO", DEFAULT_REPOSITORY))
    branch = st.text_input("Branch", value=os.environ.get("DARVAS_GITHUB_BRANCH", DEFAULT_BRANCH))
    if source == "GitHub repository":
        st.caption("For a private repository, set DARVAS_GITHUB_TOKEN in your terminal before starting the dashboard.")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    state, equity, trades = load_data(source, repository, branch)
except (FileNotFoundError, requests.RequestException) as error:
    st.error(str(error))
    st.stop()

latest = equity.iloc[-1] if not equity.empty else pd.Series(dtype=float)
latest_equity = float(latest.get("equity", INITIAL_CAPITAL))
cash = float(state.get("cash", latest.get("cash", INITIAL_CAPITAL)))
positions = positions_table(state)
invested_cost = float(positions["Invested"].sum()) if not positions.empty else 0.0
latest_state_value = float(positions["Market value"].fillna(0).sum()) if not positions.empty else 0.0
position_value = latest_state_value or float(latest.get("positions_value", 0.0))
total_return = latest_equity / INITIAL_CAPITAL - 1
net_pnl = latest_equity - INITIAL_CAPITAL
realized_pnl = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
unrealized_pnl = None if positions.empty or positions["Unrealised P&L"].isna().all() else float(positions["Unrealised P&L"].sum())

metrics = st.columns(4)
metrics[0].metric("Initial capital", money(INITIAL_CAPITAL))
metrics[1].metric("Virtual equity", money(latest_equity))
metrics[2].metric("Net P&L", money(net_pnl))
metrics[3].metric("Total return", f"{total_return:.2%}")

metrics = st.columns(4)
metrics[0].metric("Cash", money(cash))
metrics[1].metric("Open position cost", money(invested_cost))
metrics[2].metric("Open position value", money(position_value))
metrics[3].metric("Open positions", len(state.get("positions", {})))

metrics = st.columns(3)
metrics[0].metric("Realised P&L", money(realized_pnl))
metrics[1].metric("Unrealised P&L", money(unrealized_pnl))
metrics[2].metric("NIFTY 200-day filter", "ON" if latest.get("market_ok", False) else "OFF")

last_processed = state.get("last_processed", "Not processed yet")
st.caption(f"Last processed trading day: {last_processed}")

if not equity.empty:
    st.subheader("Equity curve")
    st.line_chart(equity.set_index("date")[["equity"]], y_label="Virtual equity (₹)")

st.subheader("Open positions")
if positions.empty:
    st.info("No open virtual positions.")
else:
    visible = positions.drop(columns=["Invested"])
    st.dataframe(
        visible,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry": st.column_config.NumberColumn(format="₹%.2f"),
            "Last close": st.column_config.NumberColumn(format="₹%.2f"),
            "Market value": st.column_config.NumberColumn(format="₹%.2f"),
            "Unrealised P&L": st.column_config.NumberColumn(format="₹%.2f"),
            "Unrealised return": st.column_config.NumberColumn(format="%.2f%%"),
            "Trailing stop": st.column_config.NumberColumn(format="₹%.2f"),
            "Initial stop": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )

left, right = st.columns(2)
with left:
    st.subheader("Queued buys")
    queued_buys = state.get("buys", {})
    if queued_buys:
        st.dataframe(pd.DataFrame.from_dict(queued_buys, orient="index").rename_axis("Ticker").reset_index(), use_container_width=True, hide_index=True)
    else:
        st.caption("No buys queued for the next open.")
with right:
    st.subheader("Queued sells")
    queued_sells = state.get("sells", {})
    if queued_sells:
        st.dataframe(pd.DataFrame(list(queued_sells.items()), columns=["Ticker", "Reason"]), use_container_width=True, hide_index=True)
    else:
        st.caption("No sells queued for the next open.")

st.subheader("Closed trades")
if trades.empty:
    st.info("No completed paper trades yet.")
else:
    total_pnl = float(trades["net_pnl"].sum())
    win_rate = float((trades["net_pnl"] > 0).mean())
    a, b, c = st.columns(3)
    a.metric("Completed trades", len(trades))
    b.metric("Closed P&L", money(total_pnl))
    c.metric("Win rate", f"{win_rate:.1%}")
    st.dataframe(
        trades.sort_values("exit_date", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
            "exit_price": st.column_config.NumberColumn("Exit", format="₹%.2f"),
            "costs": st.column_config.NumberColumn("Costs", format="₹%.2f"),
            "net_pnl": st.column_config.NumberColumn("Net P&L", format="₹%.2f"),
            "return_pct": st.column_config.NumberColumn("Return", format="%.2f%%"),
        },
    )
