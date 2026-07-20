"""
QUANTFOLIO - portfolio management software (console edition).

A standalone application: navigate pages with numbers, enter data at the
prompts. No browser, no GUI - just run:

    python app.py

Your portfolio and settings are saved to app_state.json automatically, so
everything is still there next time you launch.

Pages:
    1 Portfolio    view/edit positions, import from IBKR / CSV / demo
    2 Market data  per-asset performance and risk metrics
    3 Health check concentration, diversification, risk flags
    4 Optimization Markowitz strategies and efficient frontier
    5 Rebalance    exact trade list toward a target strategy
    6 Projection   Monte Carlo simulation of your portfolio
    7 Backtest     honest walk-forward strategy comparison
    8 AI analyst   plain-English explanation of your numbers
    9 Settings     dates, risk-free rate, benchmark, costs
"""

from __future__ import annotations

import json
import datetime as dt
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

from quantfolio import data, metrics, optimization as opt
from quantfolio import montecarlo as mc, walkforward as wf, advisor, broker, report
from quantfolio.ai_analyst import AIAnalyst
from quantfolio.store import PriceStore

APP_DIR = Path(__file__).parent
STATE_FILE = APP_DIR / "app_state.json"
OUTPUT_DIR = APP_DIR / "output"

pd.set_option("display.width", 160)

DEFAULT_STATE = {
    "positions": [],          # [{ticker, quantity, avg_cost}]
    "cash": 0.0,
    "settings": {
        "benchmark": "SPY",
        "start": "2020-01-01",
        "end": "today",
        "risk_free": 0.03,
        "tc_bps": 10.0,
        "fee_annual": 0.005,      # ongoing fees drag for projections (0.5%/yr)
        "inflation": 0.025,       # for real (inflation-adjusted) projections

        "lookback": 252,
        "max_weight": 0.25,
        "ibkr_host": "127.0.0.1",
        "ibkr_port": 7497,
        "ibkr_client_id": 42,
        "ibkr_autosync": 1,
        "ibkr_watch_minutes": 5,
    },
}


# ------------------------------------------------------------------ helpers

def ask(prompt: str, default=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        raise SystemExit(0)
    return raw if raw else (str(default) if default is not None else "")


def ask_float(prompt: str, default: float) -> float:
    raw = ask(prompt, default)
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        print("  Invalid number, keeping", default)
        return default


def header(title: str):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


class App:
    def __init__(self):
        self.state = self._load_state()
        self._prices = None            # cached price DataFrame
        self._prices_key = None        # (tickers, start, end) of the cache
        self._context_cache_key = None
        self._context_cache_value = None
        self._strategy_cache_key = None
        self._strategy_cache_value = None
        OUTPUT_DIR.mkdir(exist_ok=True)
        try:
            self.store = PriceStore(APP_DIR / "quantfolio_prices.db")
        except sqlite3.OperationalError:
            self.store = PriceStore(Path(tempfile.gettempdir()) / "quantfolio_prices.db")

    # ----------------------------------------------------------- state

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                loaded = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                state = json.loads(json.dumps(DEFAULT_STATE))
                state.update({k: loaded[k] for k in loaded if k != "settings"})
                state["settings"].update(loaded.get("settings", {}))
                if state["settings"].get("end") == "2025-06-30":
                    state["settings"]["end"] = "today"
                return state
            except (json.JSONDecodeError, KeyError):
                print("(!) app_state.json unreadable, starting fresh.")
        return json.loads(json.dumps(DEFAULT_STATE))

    def save(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ data

    @property
    def snapshot(self) -> broker.AccountSnapshot:
        return broker.AccountSnapshot(
            positions=[broker.Position(p["ticker"], p["quantity"], p.get("avg_cost", 0.0))
                       for p in self.state["positions"]],
            cash=self.state.get("cash", 0.0),
        )

    def universe(self) -> list[str]:
        s = self.state["settings"]
        return sorted({p["ticker"] for p in self.state["positions"]} | {s["benchmark"]})

    @staticmethod
    def _resolve_date(value: str, *, today: dt.date | None = None) -> str:
        """Resolve dynamic date settings while preserving explicit backtests."""
        raw = str(value).strip()
        if raw.lower() in {"today", "auto", "now", ""}:
            return (today or dt.date.today()).isoformat()
        return pd.Timestamp(raw).date().isoformat()

    def analysis_dates(self) -> tuple[str, str]:
        s = self.state["settings"]
        start = self._resolve_date(s["start"])
        end = self._resolve_date(s["end"])
        if start >= end:
            raise ValueError(f"Analysis start ({start}) must be before end ({end}).")
        return start, end

    def prices(self) -> tuple[pd.DataFrame, str]:
        start, end = self.analysis_dates()
        key = (tuple(self.universe()), start, end)
        if self._prices is None or self._prices_key != key:
            print("Loading prices...")
            self._prices, self._source = self.store.get_prices(
                list(key[0]), start, end)
            self._prices_key = key
            print(f"  source: {self._source} | {len(self._prices)} days")
            if self._source == "synthetic":
                print("  (!) synthetic data - offline mode; numbers are realistic "
                      "but NOT real market prices")
        return self._prices, self._source

    def require_positions(self) -> bool:
        if not self.state["positions"]:
            print("\n(!) No positions yet. Go to page 1 to add or import some.")
            return False
        return True

    def portfolio_context(self):
        """Common objects used by most pages."""
        s = self.state["settings"]
        prices, _ = self.prices()
        positions_key = tuple(sorted(
            (p["ticker"], float(p["quantity"]), float(p.get("avg_cost", 0.0)))
            for p in self.state["positions"]))
        key = (self._prices_key, id(prices), positions_key,
               float(self.state.get("cash", 0.0)), s["benchmark"])
        if key == self._context_cache_key:
            return self._context_cache_value
        held = [p["ticker"] for p in self.state["positions"] if p["ticker"] in prices.columns]
        asset_prices = prices[held]
        asset_returns = data.to_returns(asset_prices)
        bench = data.to_returns(prices)[s["benchmark"]]
        last = prices.iloc[-1]
        snap = self.snapshot
        weights = snap.weights(last)
        value = snap.market_value(last) + snap.cash
        result = (s, prices, asset_returns, bench, last, snap, weights, value)
        self._context_cache_key = key
        self._context_cache_value = result
        return result

    # ------------------------------------------------------------ pages

    def page_portfolio(self):
        while True:
            header("1. PORTFOLIO")
            snap = self.snapshot
            if snap.positions:
                df = snap.to_frame()
                try:
                    prices, _ = self.prices()
                    last = prices.iloc[-1]
                    df["last_price"] = [round(float(last.get(t, float("nan"))), 2)
                                        for t in df.index]
                    df["market_value"] = (df["quantity"] * df["last_price"]).round(2)
                    total = df["market_value"].sum() + snap.cash
                    df["weight"] = (df["market_value"] / df["market_value"].sum()).round(4)
                    print(df.to_string())
                    print(f"\nCash: {snap.cash:,.2f} | Total value: {total:,.2f}")
                except Exception as exc:  # noqa: BLE001
                    print(df.to_string())
                    print(f"(prices unavailable: {exc})")
            else:
                print("No positions.")

            print("\n[a] add/edit position  [d] delete position  [c] set cash")
            print("[i] import from IBKR   [f] import from CSV   [m] load demo portfolio")
            print("[x] back to menu")
            choice = ask("Choice").lower()

            if choice == "a":
                tk = ask("Ticker (e.g. AAPL)").upper()
                if not tk:
                    continue
                qty = ask_float("Quantity (shares)", 0)
                cost = ask_float("Average cost per share (optional)", 0)
                self.state["positions"] = [p for p in self.state["positions"]
                                           if p["ticker"] != tk]
                if qty > 0:
                    self.state["positions"].append(
                        {"ticker": tk, "quantity": qty, "avg_cost": cost})
                self._prices = None
                self.save()
            elif choice == "d":
                tk = ask("Ticker to delete").upper()
                self.state["positions"] = [p for p in self.state["positions"]
                                           if p["ticker"] != tk]
                self._prices = None
                self.save()
            elif choice == "c":
                self.state["cash"] = ask_float("Cash amount", self.state.get("cash", 0.0))
                self.save()
            elif choice == "i":
                self._import_ibkr()
            elif choice == "f":
                path = ask("CSV path (columns: ticker,quantity,avg_cost)")
                try:
                    snap = broker.from_csv(path)
                    self._adopt_snapshot(snap)
                except Exception as exc:  # noqa: BLE001
                    print(f"(!) Import failed: {exc}")
            elif choice == "m":
                self._adopt_snapshot(broker.demo_positions())
                print("Demo portfolio loaded.")
            elif choice == "x":
                return

    def _import_ibkr(self):
        s = self.state["settings"]
        port = int(ask_float("TWS/Gateway port (7497 paper, 7496 live)", s["ibkr_port"]))
        s["ibkr_port"] = port
        print("Connecting to IBKR (read-only)...")
        try:
            snap = broker.IBKRClient(
                host=str(s.get("ibkr_host", "127.0.0.1")),
                port=port,
                client_id=int(s.get("ibkr_client_id", 42)),
            ).fetch()
            self._adopt_snapshot(snap)
            print(f"Imported {len(snap.positions)} positions | cash {snap.cash:,.2f} "
                  f"| net liquidation {snap.net_liquidation:,.2f}")
        except RuntimeError as exc:
            print(f"(!) {exc}")
            print("    Checklist: TWS or IB Gateway running? API enabled in")
            print("    Global Configuration > API > Settings? Correct port?")

    def _adopt_snapshot(self, snap: broker.AccountSnapshot):
        old_universe = self.universe()
        self.state["positions"] = [
            {"ticker": p.ticker, "quantity": p.quantity, "avg_cost": p.avg_cost}
            for p in snap.positions]
        if snap.source == "ibkr" or snap.cash:
            self.state["cash"] = snap.cash
        self.store.upsert_instruments([
            {"ticker": p.ticker, "currency": p.currency,
             "source": snap.source or "portfolio"}
            for p in snap.positions])
        # Quantity/cost changes do not invalidate market prices. Only a new or
        # removed symbol requires another data query.
        if self.universe() != old_universe:
            self._prices = None
            self._prices_key = None
        self.save()

    def page_market(self):
        header("2. MARKET DATA - metrics per asset")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        stats = metrics.summary(asset_returns, bench, s["risk_free"])
        print(stats.round(3).to_string())
        port_r = (asset_returns * weights.reindex(asset_returns.columns).fillna(0)).sum(axis=1)
        pstats = metrics.summary(port_r.rename("MyPortfolio"), bench, s["risk_free"])
        print("\nYour portfolio (current weights):")
        print(pstats.round(3).to_string())

    def page_health(self):
        header("3. HEALTH CHECK")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        table, flags = advisor.health_check(weights, asset_returns, bench, s["risk_free"])
        print(table.to_string())
        print("\nFlags:")
        for f in flags:
            print(f"  - {f}")

    def _target_strategies(self, asset_returns):
        s = self.state["settings"]
        cache_key = (id(asset_returns), float(s["risk_free"]),
                     float(s["max_weight"]))
        if cache_key == self._strategy_cache_key:
            strategies, mu, cov = self._strategy_cache_value
            return ({name: weights.copy() for name, weights in strategies.items()},
                    mu.copy(), cov.copy())
        mu, cov = opt.annualized_inputs(
            asset_returns, shrinkage=True, mean_shrinkage=0.5)
        n = len(mu)
        cap = max(s["max_weight"], 1.0 / n + 0.01)
        bounds = opt.weight_bounds(n, cap)
        strategies = {
            "MaxSharpe (LW + cap)": opt.max_sharpe_weights(mu, cov, s["risk_free"], bounds),
            "MinVol (LW + cap)": opt.min_volatility_weights(cov, bounds),
            "RiskParity": opt.risk_parity_weights(cov, bounds),
            "EqualWeight": opt.equal_weights(list(asset_returns.columns)),
        }
        self._strategy_cache_key = cache_key
        self._strategy_cache_value = (strategies, mu, cov)
        return ({name: weights.copy() for name, weights in strategies.items()},
                mu.copy(), cov.copy())

    def page_optimization(self):
        header("4. OPTIMIZATION - Markowitz strategies")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        strategies, mu, cov = self._target_strategies(asset_returns)
        print("Weights per strategy (Ledoit-Wolf covariance, "
              f"{s['max_weight']:.0%} cap):\n")
        alloc = pd.DataFrame(strategies)
        alloc["Current"] = weights.reindex(alloc.index).fillna(0.0)
        print(alloc.round(3).to_string())
        print("\nEx-ante return / volatility / Sharpe:")
        for name, w in {**strategies, "Current": weights}.items():
            wv = w.reindex(mu.index).fillna(0.0).values
            r_, v_ = opt.portfolio_return(wv, mu), opt.portfolio_volatility(wv, cov)
            print(f"  {name:<22} {r_:7.2%}  {v_:7.2%}  {(r_ - s['risk_free']) / v_:5.2f}")
        if ask("\nSave efficient frontier chart? (y/n)", "n").lower() == "y":
            cap = max(s["max_weight"], 1.0 / len(mu) + 0.01)
            bounds = opt.weight_bounds(len(mu), cap)
            frontier = opt.efficient_frontier(mu, cov, n_points=100, bounds=bounds)
            cloud = opt.random_portfolios(mu, cov, n=3000, bounds=bounds)
            highlights = {
                "MaxSharpe": (opt.portfolio_volatility(strategies["MaxSharpe (LW + cap)"].values, cov),
                              opt.portfolio_return(strategies["MaxSharpe (LW + cap)"].values, mu)),
                "MinVol": (opt.portfolio_volatility(strategies["MinVol (LW + cap)"].values, cov),
                           opt.portfolio_return(strategies["MinVol (LW + cap)"].values, mu)),
            }
            report.plot_efficient_frontier(frontier, cloud, highlights, None,
                                           OUTPUT_DIR / "app_frontier.png")
            print(f"Saved: {OUTPUT_DIR / 'app_frontier.png'}")

    def page_rebalance(self):
        header("5. REBALANCE ADVISOR")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        strategies, mu, cov = self._target_strategies(asset_returns)
        names = list(strategies)
        print("Target strategy:")
        for i, n in enumerate(names, 1):
            print(f"  [{i}] {n}")
        try:
            pick = int(ask("Choice", 1))
            target = strategies[names[pick - 1]]
        except (ValueError, IndexError):
            print("Invalid choice.")
            return
        proposal = advisor.propose_rebalance(
            weights, target, last, snap.market_value(last),
            tc_bps=s["tc_bps"], min_trade_value=ask_float("Ignore trades below (value)", 100))
        if proposal.empty:
            print("\nAlready aligned with the target (within the minimum trade size).")
            return
        print(f"\nTrades to reach '{names[pick - 1]}' "
              f"(REVIEW AND PLACE THEM YOURSELF - nothing is executed):\n")
        print(proposal.to_string())
        to = advisor.turnover(proposal, snap.market_value(last))
        print(f"\nTurnover: {to:.1%} | Total estimated costs: "
              f"{proposal['est_cost'].sum():,.2f}")
        if ask("Save as CSV? (y/n)", "n").lower() == "y":
            out = OUTPUT_DIR / "rebalance_proposal.csv"
            proposal.to_csv(out)
            print(f"Saved: {out}")

    def page_projection(self):
        header("6. PROJECTION - Monte Carlo")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        years = ask_float("Horizon in years", 3)
        horizon = int(years * 252)
        w = weights.reindex(asset_returns.columns).fillna(0.0)
        rf = s["risk_free"]; fee = s.get("fee_annual", 0.005); infl = s.get("inflation", 0.025)
        print("Simulating 2 x 20,000 paths (realistic: anchored expected "
              "returns, net of %.1f%% fees, real vs %.1f%% inflation)..."
              % (fee * 100, infl * 100))
        gbm = mc.simulate_gbm(asset_returns, w, value, horizon, n_sims=20000,
                              risk_free=rf, cost_annual=fee, inflation=infl)
        boot = mc.simulate_bootstrap(asset_returns, w, value, horizon, n_sims=20000,
                                     risk_free=rf, cost_annual=fee, inflation=infl)
        summary = pd.concat([gbm.summary(), boot.summary()], axis=1)
        summary.columns = ["GBM", "Bootstrap"]
        print(summary.to_string())
        print("\nNote: a projection is a RANGE of outcomes, not a promise. "
              "The median is the central scenario; P5/P95 frame the plausible band.")
        report.plot_monte_carlo(gbm, OUTPUT_DIR / "app_montecarlo.png")
        print(f"\nChart saved: {OUTPUT_DIR / 'app_montecarlo.png'}")

    def page_backtest(self):
        header("7. BACKTEST - walk-forward (honest, no look-ahead)")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        held = list(asset_returns.columns)
        print("This compares strategies on YOUR tickers, out-of-sample.")
        print("It can take ~30-60 seconds...")
        strategies = {
            "MaxSharpe WF": wf.make_max_sharpe(s["risk_free"],
                                               max(s["max_weight"], 1 / len(held) + 0.01)),
            "MinVol WF": wf.make_min_vol(max(s["max_weight"], 1 / len(held) + 0.01)),
            "EqualWeight 1/N": wf.make_equal_weight(),
        }
        try:
            curves, results = wf.compare_walk_forward(
                prices[held], strategies, lookback=s["lookback"], tc_bps=s["tc_bps"])
        except ValueError as exc:
            print(f"(!) {exc}")
            return
        curves[s["benchmark"]] = (10_000 * (1 + bench.reindex(curves.index)).cumprod())
        stats = metrics.summary(curves.pct_change().dropna(),
                                bench.reindex(curves.index), s["risk_free"])
        print(stats.loc[["CAGR", "Annualized volatility", "Sharpe",
                         "Max drawdown", "Calmar"]].round(3).to_string())
        report.plot_equity_curves(curves, OUTPUT_DIR / "app_walkforward.png",
                                  title="Walk-forward on your tickers")
        print(f"\nChart saved: {OUTPUT_DIR / 'app_walkforward.png'}")

    def page_ai(self):
        header("8. AI ANALYST")
        if not self.require_positions():
            return
        s, prices, asset_returns, bench, last, snap, weights, value = self.portfolio_context()
        port_r = (asset_returns * weights.reindex(asset_returns.columns).fillna(0)).sum(axis=1)
        pstats = metrics.summary(port_r.rename("MyPortfolio"), bench, s["risk_free"])
        table, flags = advisor.health_check(weights, asset_returns, bench, s["risk_free"])
        analyst = AIAnalyst()
        mode = "Claude API" if analyst.api_key else "offline rule engine (set ANTHROPIC_API_KEY for the full AI)"
        print(f"Mode: {mode}\n")
        print(analyst.explain(
            pstats, table,
            context=f"Current portfolio: {dict(weights.round(3))} | "
                    f"flags: {'; '.join(flags)}"))

    def page_settings(self):
        while True:
            header("9. SETTINGS")
            s = self.state["settings"]
            for i, (k, v) in enumerate(s.items(), 1):
                print(f"  [{i}] {k:<12} = {v}")
            print("  [x] back")
            choice = ask("Edit which").lower()
            if choice == "x":
                self.save()
                self._prices = None
                return
            keys = list(s)
            try:
                key = keys[int(choice) - 1]
            except (ValueError, IndexError):
                continue
            if key == "benchmark":
                s[key] = ask(f"New value for {key}", s[key]).upper()
            elif key in ("start", "end"):
                s[key] = ask(f"New value for {key} (YYYY-MM-DD)", s[key])
            elif key == "ibkr_host":
                s[key] = ask(f"New value for {key}", s[key])
            elif key in ("lookback", "ibkr_port", "ibkr_client_id",
                         "ibkr_autosync", "ibkr_watch_minutes"):
                s[key] = int(ask_float(f"New value for {key}", s[key]))
            else:
                s[key] = ask_float(f"New value for {key}", s[key])

    # ------------------------------------------------------------- loop

    def run(self):
        print("\n" + "#" * 62)
        print("#  LFT (Le Fort) - portfolio management software (v0.4)")
        print("#  Educational tool - nothing here is investment advice")
        print("#" * 62)
        pages = {
            "1": self.page_portfolio, "2": self.page_market,
            "3": self.page_health, "4": self.page_optimization,
            "5": self.page_rebalance, "6": self.page_projection,
            "7": self.page_backtest, "8": self.page_ai,
            "9": self.page_settings,
        }
        while True:
            s = self.state["settings"]
            n = len(self.state["positions"])
            print(f"\n{'-' * 62}")
            print(f"Portfolio: {n} positions | cash {self.state.get('cash', 0):,.0f} | "
                  f"{s['start']} -> {s['end']} | rf {s['risk_free']:.1%} | bench {s['benchmark']}")
            print("[1] Portfolio   [2] Market data  [3] Health check")
            print("[4] Optimization [5] Rebalance   [6] Projection")
            print("[7] Backtest    [8] AI analyst   [9] Settings   [0] Quit")
            choice = ask("Page")
            if choice == "0":
                self.save()
                print("State saved. Goodbye.")
                return
            page = pages.get(choice)
            if page:
                try:
                    page()
                except KeyboardInterrupt:
                    print("\n(interrupted)")
                except Exception as exc:  # noqa: BLE001
                    print(f"\n(!) Error on this page: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    try:
        App().run()
    except (KeyboardInterrupt, SystemExit):
        print("\nGoodbye.")
        sys.exit(0)
