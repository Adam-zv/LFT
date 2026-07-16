"""
Personal P&L - YOUR return, not the market's.

A quoted performance ("the S&P did +12%") says nothing about what YOU
earned, because you added and withdrew money along the way. Two distinct
answers:

    Money-weighted return (XIRR)  the annual rate that makes all your
                                  dated cash flows grow into today's
                                  value. This is YOUR personal return -
                                  it rewards/punishes your timing.

    Realized / unrealized P&L     per position, using the average-cost
                                  method: selling locks in realized P&L
                                  against the average purchase price;
                                  what you still hold carries unrealized
                                  P&L against the latest price.

Transactions come from a CSV (date,ticker,quantity,price[,fees]) with
negative quantities for sells - the format IBKR Flex exports are easily
reshaped into.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------- XIRR

def xirr(dates: list, amounts: list[float], guess_bounds=(-0.95, 10.0)) -> float:
    """
    Internal rate of return for dated cash flows (like Excel's XIRR).

    Convention: money you PUT IN is negative, money you GET OUT (or the
    final portfolio value) is positive. Solved by bisection - robust, no
    derivative needed. Raises ValueError if flows never change sign.
    """
    if len(dates) != len(amounts) or len(dates) < 2:
        raise ValueError("Need at least two dated cash flows.")
    amounts = [float(a) for a in amounts]
    if not (min(amounts) < 0 < max(amounts)):
        raise ValueError("Cash flows must contain both inflows and outflows.")

    d0 = dates[0]
    def _days(d):
        if isinstance(d, str):
            d = datetime.fromisoformat(d).date()
        if isinstance(d, datetime):
            d = d.date()
        if isinstance(d0, str):
            base = datetime.fromisoformat(d0).date()
        elif isinstance(d0, datetime):
            base = d0.date()
        else:
            base = d0
        return (d - base).days

    years = np.array([_days(d) / 365.25 for d in dates])
    cf = np.array(amounts)

    def npv(rate):
        return float((cf / (1 + rate) ** years).sum())

    lo, hi = guess_bounds
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        raise ValueError("No IRR in the search interval.")
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def money_weighted_return(transactions: pd.DataFrame,
                          final_value: float,
                          final_date=None) -> float:
    """
    Personal annual return from a transaction table.
    Buys are money in (negative flow), sells money out (positive flow),
    plus today's portfolio value as the closing positive flow.
    """
    final_date = final_date or date.today()
    flows_d, flows_a = [], []
    for _, t in transactions.sort_values("date").iterrows():
        cash = -float(t["quantity"]) * float(t["price"]) - float(t.get("fees", 0) or 0)
        flows_d.append(t["date"])
        flows_a.append(cash)
    flows_d.append(final_date)
    flows_a.append(float(final_value))
    return xirr(flows_d, flows_a)


# --------------------------------------------------------- transaction P&L

@dataclass
class PositionPnL:
    ticker: str
    quantity: float          # still held
    avg_cost: float
    invested: float          # cost basis of what is still held
    realized: float          # locked-in P&L from sells (net of fees)
    unrealized: float
    last_price: float

    @property
    def total(self) -> float:
        return self.realized + self.unrealized


def load_transactions_csv(path: str | Path) -> pd.DataFrame:
    """
    CSV columns: date,ticker,quantity,price[,fees].
    Negative quantity = sell. Lines starting with '#' ignored.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tk = (row.get("ticker") or "").strip().upper()
            if not tk or tk.startswith("#"):
                continue
            rows.append({
                "date": datetime.fromisoformat(row["date"].strip()).date(),
                "ticker": tk,
                "quantity": float(row["quantity"]),
                "price": float(row["price"]),
                "fees": float(row.get("fees") or 0),
            })
    if not rows:
        raise ValueError(f"No transactions found in {path}")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def pnl_report(transactions: pd.DataFrame,
               last_prices: pd.Series) -> pd.DataFrame:
    """
    Average-cost P&L per ticker.

    Buys raise the average cost; each sell realizes
    (sell price - average cost) * quantity - fees. Fees on buys go into
    the cost basis. Short selling is not supported (educational scope).
    """
    results = []
    for tk, txs in transactions.groupby("ticker"):
        qty, avg = 0.0, 0.0
        realized = 0.0
        for _, t in txs.sort_values("date").iterrows():
            q, p, fee = float(t["quantity"]), float(t["price"]), float(t.get("fees", 0) or 0)
            if q > 0:                                   # buy
                avg = (qty * avg + q * p + fee) / (qty + q)
                qty += q
            else:                                       # sell
                sell_q = min(-q, qty)
                if sell_q < -q - 1e-9:
                    raise ValueError(f"{tk}: selling more than held "
                                     f"(short selling not supported).")
                realized += sell_q * (p - avg) - fee
                qty -= sell_q
        last = float(last_prices.get(tk, np.nan))
        unreal = qty * (last - avg) if qty > 0 and not np.isnan(last) else 0.0
        results.append(PositionPnL(tk, round(qty, 6), round(avg, 4),
                                   round(qty * avg, 2), round(realized, 2),
                                   round(unreal, 2), round(last, 4)))
    df = pd.DataFrame([r.__dict__ | {"total": r.total} for r in results])
    return df.set_index("ticker").sort_values("total", ascending=False)


def performance_summary(transactions: pd.DataFrame,
                        last_prices: pd.Series,
                        final_date=None) -> pd.Series:
    """One-look personal performance report."""
    pnl = pnl_report(transactions, last_prices)
    market_value = float((pnl["quantity"] * pnl["last_price"]).sum())
    invested_total = float(-(-transactions["quantity"].clip(lower=0)
                             * transactions["price"]).sum())
    mwr = money_weighted_return(transactions, market_value, final_date)
    return pd.Series({
        "Positions": int((pnl["quantity"] > 0).sum()),
        "Market value": round(market_value, 2),
        "Total invested (buys)": round(abs(invested_total), 2),
        "Realized P&L": round(float(pnl["realized"].sum()), 2),
        "Unrealized P&L": round(float(pnl["unrealized"].sum()), 2),
        "Total P&L": round(float(pnl["total"].sum()), 2),
        "Money-weighted return (ann.)": round(mwr, 4),
    })
