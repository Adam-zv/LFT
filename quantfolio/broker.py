"""
IBKR connection - READ-ONLY portfolio import.

Three ways to get your real positions into the engine:

1. Live TWS API (recommended): run Trader Workstation or IB Gateway on
   your machine, enable the API (see setup notes below), then
   `IBKRClient().fetch()` reads positions, cash and account value through
   the `ib_async` library. The connection is opened with readonly=True:
   this code can never place an order.

2. CSV import: export your positions to a CSV with columns
   ticker,quantity,avg_cost and load them with `from_csv(path)`.

3. Demo mode: `demo_positions()` returns a plausible fake portfolio so
   every page of the app works without an IBKR account.

IBKR setup (one-time, in Trader Workstation):
    File > Global Configuration > API > Settings
      [x] Enable ActiveX and Socket Clients
      [x] Read-Only API                     <- belt and braces
      Socket port: 7497 (paper) or 7496 (live)
    Keep TWS running while you use the app.

Install the client library:  pip install ib_async
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float = 0.0
    currency: str = "USD"

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost


@dataclass
class AccountSnapshot:
    """Read-only picture of a brokerage account."""
    positions: list[Position] = field(default_factory=list)
    cash: float = 0.0
    net_liquidation: float = 0.0
    source: str = "demo"

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    def to_frame(self) -> pd.DataFrame:
        rows = [{"ticker": p.ticker, "quantity": p.quantity,
                 "avg_cost": p.avg_cost, "cost_basis": p.cost_basis,
                 "currency": p.currency} for p in self.positions]
        return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()

    def weights(self, last_prices: pd.Series) -> pd.Series:
        """Current market-value weights given latest prices."""
        values = {p.ticker: p.quantity * float(last_prices[p.ticker])
                  for p in self.positions if p.ticker in last_prices}
        s = pd.Series(values, dtype=float)
        return s / s.sum() if s.sum() > 0 else s

    def market_value(self, last_prices: pd.Series) -> float:
        return float(sum(p.quantity * float(last_prices[p.ticker])
                         for p in self.positions if p.ticker in last_prices))


# --------------------------------------------------------------- demo / CSV

def demo_positions() -> AccountSnapshot:
    """Plausible fake account so the app works without IBKR."""
    return AccountSnapshot(
        positions=[
            Position("AAPL", 25, 165.0),
            Position("MSFT", 15, 310.0),
            Position("AMZN", 20, 140.0),
            Position("JNJ", 30, 155.0),
            Position("TLT", 40, 95.0),
            Position("GLD", 18, 180.0),
        ],
        cash=4_500.0,
        net_liquidation=0.0,   # computed later from prices
        source="demo",
    )


def from_csv(path: str | Path) -> AccountSnapshot:
    """
    Load positions from a CSV with header: ticker,quantity[,avg_cost].
    Lines starting with '#' are ignored.
    """
    positions = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tk = (row.get("ticker") or "").strip().upper()
            if not tk or tk.startswith("#"):
                continue
            positions.append(Position(
                ticker=tk,
                quantity=float(row.get("quantity") or 0),
                avg_cost=float(row.get("avg_cost") or 0),
            ))
    if not positions:
        raise ValueError(f"No positions found in {path}")
    return AccountSnapshot(positions=positions, source=f"csv:{path}")


# ------------------------------------------------------------- live IBKR

class IBKRClient:
    """
    Read-only client for a running TWS / IB Gateway.

    Usage:
        snap = IBKRClient(port=7497).fetch()   # 7497 paper, 7496 live
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 42, timeout: float = 10.0):
        self.host, self.port, self.client_id, self.timeout = host, port, client_id, timeout

    def fetch(self) -> AccountSnapshot:
        try:
            from ib_async import IB
        except ImportError as exc:
            raise RuntimeError(
                "ib_async is not installed. Run: pip install ib_async"
            ) from exc

        ib = IB()
        try:
            # readonly=True: the API session cannot transmit orders
            ib.connect(self.host, self.port, clientId=self.client_id,
                       readonly=True, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach TWS/IB Gateway on {self.host}:{self.port}. "
                f"Is TWS running with the API enabled? ({exc})"
            ) from exc

        try:
            positions = [
                Position(
                    ticker=p.contract.symbol,
                    quantity=float(p.position),
                    avg_cost=float(p.avgCost),
                    currency=p.contract.currency or "USD",
                )
                for p in ib.positions()
                if p.contract.secType == "STK" and p.position != 0
            ]

            cash = net_liq = 0.0
            for av in ib.accountSummary():
                if av.tag == "TotalCashValue" and av.currency == "USD":
                    cash = float(av.value)
                elif av.tag == "NetLiquidation" and av.currency == "USD":
                    net_liq = float(av.value)

            return AccountSnapshot(positions=positions, cash=cash,
                                   net_liquidation=net_liq, source="ibkr")
        finally:
            ib.disconnect()
