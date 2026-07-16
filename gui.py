"""
LFT - desktop application (windowed edition).

A real standalone window: sidebar navigation on the left, pages on the
right, tables, input forms and charts embedded in the window.

Launch:  python gui.py        (or double-click LFT.bat)

Reuses the exact same engine and the same saved state (app_state.json)
as the console edition (app.py) - you can switch between the two freely.
"""

from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app import App                      # tested backend: state, data, strategies
from quantfolio import data, metrics, optimization as opt
from quantfolio import montecarlo as mc, walkforward as wf, advisor, broker
from quantfolio.ai_analyst import AIAnalyst
from quantfolio import performance as perf

PAGES = ["Portfolio", "Market data", "Health check", "Optimization",
         "Rebalance", "Projection", "Backtest", "Performance", "AI analyst",
         "IBKR connection", "Settings"]


# ------------------------------------------------------------ small helpers

def df_to_tree(tree: ttk.Treeview, df: pd.DataFrame, index_name: str = ""):
    """Fill a Treeview from a DataFrame (index shown as first column)."""
    tree.delete(*tree.get_children())
    cols = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    tree.configure(columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=max(70, min(160, 11 * len(c))), anchor="e")
    tree.column(cols[0], anchor="w", width=150)
    for idx, row in df.iterrows():
        vals = [idx] + [("" if pd.isna(v) else round(v, 4) if isinstance(v, float) else v)
                        for v in row]
        tree.insert("", "end", values=vals)


def make_tree(parent, height=12) -> ttk.Treeview:
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, pady=4)
    tree = ttk.Treeview(frame, show="headings", height=height)
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    return tree


class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LFT - Le Fort - portfolio management")
        self.geometry("1180x760")
        self.minsize(980, 640)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        self.backend = App()             # engine + persisted state
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._ibkr_watch_job = None

        # ---- layout: sidebar | content
        sidebar = tk.Frame(self, bg="#1f2732", width=180)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        brand = tk.Frame(sidebar, bg="#1f2732")
        brand.pack(fill="x", pady=(18, 16))
        tk.Label(brand, text="LFT", bg="#1f2732", fg="white",
                 font=("Segoe UI", 19, "bold")).pack(anchor="w", padx=18)
        self._nav_buttons = {}
        for name in PAGES:
            b = tk.Button(sidebar, text=name, relief="flat", anchor="w",
                          bg="#1f2732", fg="#c9d4e0", activebackground="#324153",
                          activeforeground="white", padx=18, pady=8,
                          font=("Segoe UI", 10),
                          command=lambda n=name: self.show_page(n))
            b.pack(fill="x")
            self._nav_buttons[name] = b
        tk.Label(sidebar, text="Educational tool.\nNot investment advice.",
                 bg="#1f2732", fg="#6b7a8c", font=("Segoe UI", 8),
                 justify="left", padx=18, pady=14).pack(side="bottom", fill="x")

        self.status = tk.StringVar(value="Ready.")
        statusbar = tk.Label(self, textvariable=self.status, anchor="w",
                             bd=1, relief="sunken", padx=8)
        statusbar.pack(side="bottom", fill="x")

        self.content = ttk.Frame(self, padding=14)
        self.content.pack(side="right", fill="both", expand=True)

        self.after(100, self._poll)
        self.show_page("Portfolio")
        if int(self.backend.state["settings"].get("ibkr_autosync", 0)):
            self.after(800, self._autosync_ibkr)
        self._schedule_ibkr_watch()
        self.after(1500, self._show_data_freshness)

    def _show_data_freshness(self):
        """Startup status-bar note: how fresh the cached market data is."""
        def work():
            cov = self.backend.store.coverage()
            return None if cov.empty else str(cov["end"].max())

        def done(last):
            if last:
                self.status.set(f"Market data cached through {last} - "
                                f"newer days download automatically when needed.")
            else:
                self.status.set("No market data cached yet - it will download "
                                "on first use (internet required; offline = "
                                "synthetic demo data).")

        self.run_async(work, done, "Checking data freshness...",
                       silent_errors=True)

    def _ibkr_client(self, timeout=10.0):
        s = self.backend.state["settings"]
        return broker.IBKRClient(
            host=str(s.get("ibkr_host", "127.0.0.1")),
            port=int(s.get("ibkr_port", 7497)),
            client_id=int(s.get("ibkr_client_id", 42)),
            timeout=timeout,
        )

    def _schedule_ibkr_watch(self):
        if self._ibkr_watch_job is not None:
            try:
                self.after_cancel(self._ibkr_watch_job)
            except tk.TclError:
                pass
            self._ibkr_watch_job = None
        watch = int(self.backend.state["settings"].get("ibkr_watch_minutes", 0))
        if watch > 0:
            self._ibkr_watch_job = self.after(watch * 60_000, self._watch_ibkr)

    def _watch_ibkr(self):
        """Hands-free mode: while the app is open, periodically re-import
        from IBKR; when positions changed, refresh and surface the health
        warnings. Configure it from the IBKR connection page."""
        self._ibkr_watch_job = None
        watch = int(self.backend.state["settings"].get("ibkr_watch_minutes", 0))
        if watch <= 0:
            return
        before = {(p_["ticker"], p_["quantity"], p_.get("avg_cost", 0.0))
                  for p_ in self.backend.state["positions"]}
        before_cash = round(float(self.backend.state.get("cash", 0.0)), 2)

        def work():
            snap = self._ibkr_client(timeout=5).fetch()
            after = {(p_.ticker, p_.quantity, p_.avg_cost)
                     for p_ in snap.positions}
            if after == before and round(float(snap.cash), 2) == before_cash:
                return None
            return snap

        def done(snap):
            if snap is None:
                return
            self.backend._adopt_snapshot(snap)
            try:
                s, prices, ar, bench, last, sn, weights, value = \
                    self.backend.portfolio_context()
                _, flags = advisor.health_check(weights, ar, bench,
                                                s["risk_free"])
                warn = "\n".join(f"- {f}" for f in flags[:4])
                messagebox.showinfo(
                    "LFT - portfolio updated from IBKR",
                    f"Your IBKR positions changed "
                    f"({len(snap.positions)} positions now).\n\n"
                    f"Watch out for:\n{warn}")
            except Exception:  # noqa: BLE001
                self.status.set("IBKR watch: portfolio updated.")
            self.show_page("Portfolio")

        self.run_async(work, done, "IBKR watch: checking for changes...",
                       silent_errors=True)
        self._schedule_ibkr_watch()

    def _autosync_ibkr(self):
        """Optional silent refresh from IBKR at startup (Settings:
        ibkr_autosync = 1). Fails quietly if TWS is not running."""
        def work():
            return self._ibkr_client(timeout=5).fetch()

        def done(snap):
            self.backend._adopt_snapshot(snap)
            self.status.set(f"IBKR auto-sync: {len(snap.positions)} "
                            f"positions imported.")
            self.show_page("Portfolio")

        self.run_async(work, done, "Auto-syncing with IBKR...",
                       silent_errors=True)

    # ------------------------------------------------------ infrastructure

    def _poll(self):
        try:
            while True:
                (status, payload), on_done = self._queue.get_nowait()
                self._busy = False
                if status == "silent_err":
                    self.status.set(f"(background task failed: {payload})")
                elif status == "err":
                    self.status.set(f"Error: {payload}")
                    messagebox.showerror("LFT", str(payload))
                else:
                    on_done(payload)
                    self.status.set("Done.")
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def run_async(self, work, on_done, busy_msg="Computing...",
                  silent_errors=False):
        if self._busy:
            self.status.set("Please wait, a computation is running...")
            return
        self._busy = True
        self.status.set(busy_msg)

        def worker():
            try:
                self._queue.put((("ok", work()), on_done))
            except Exception as exc:  # noqa: BLE001
                kind = "silent_err" if silent_errors else "err"
                self._queue.put(((kind, exc), on_done))
        threading.Thread(target=worker, daemon=True).start()

    def clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    def show_page(self, name: str):
        for n, b in self._nav_buttons.items():
            b.configure(bg="#324153" if n == name else "#1f2732")
        self.clear_content()
        ttk.Label(self.content, text=name, font=("Segoe UI", 16, "bold")
                  ).pack(anchor="w", pady=(0, 10))
        getattr(self, "page_" + name.lower().replace(" ", "_"))()

    def need_positions(self) -> bool:
        if not self.backend.state["positions"]:
            ttk.Label(self.content, foreground="#a33",
                      text="No positions yet. Open the Portfolio page and add "
                           "or import some (the 'Load demo' button is the "
                           "fastest way to try the app).").pack(anchor="w")
            return True
        return False

    def embed_chart(self, fig: Figure, parent=None):
        canvas = FigureCanvasTkAgg(fig, master=parent or self.content)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=6)

    # --------------------------------------------------------- 1 portfolio

    def page_portfolio(self):
        top = ttk.Frame(self.content)
        top.pack(fill="x")
        ttk.Label(top, text="Ticker").grid(row=0, column=0, padx=3)
        ttk.Label(top, text="Quantity").grid(row=0, column=1, padx=3)
        ttk.Label(top, text="Avg cost").grid(row=0, column=2, padx=3)
        self.e_tk = ttk.Entry(top, width=9)
        self.e_qty = ttk.Entry(top, width=9)
        self.e_cost = ttk.Entry(top, width=9)
        self.e_tk.grid(row=1, column=0, padx=3)
        self.e_qty.grid(row=1, column=1, padx=3)
        self.e_cost.grid(row=1, column=2, padx=3)
        ttk.Button(top, text="Add / update", command=self._pos_add).grid(row=1, column=3, padx=6)
        ttk.Button(top, text="Delete selected", command=self._pos_del).grid(row=1, column=4, padx=6)
        ttk.Button(top, text="Load demo", command=self._pos_demo).grid(row=1, column=5, padx=6)
        ttk.Button(top, text="Import CSV", command=self._pos_csv).grid(row=1, column=6, padx=6)
        ttk.Button(top, text="Import IBKR", command=self._pos_ibkr).grid(row=1, column=7, padx=6)

        cash_row = ttk.Frame(self.content)
        cash_row.pack(fill="x", pady=(8, 0))
        ttk.Label(cash_row, text="Cash:").pack(side="left")
        self.e_cash = ttk.Entry(cash_row, width=12)
        self.e_cash.insert(0, str(self.backend.state.get("cash", 0.0)))
        self.e_cash.pack(side="left", padx=4)
        ttk.Button(cash_row, text="Set", command=self._set_cash).pack(side="left")
        self.lbl_total = ttk.Label(cash_row, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_total.pack(side="right")

        self.tree_pos = make_tree(self.content, height=14)
        self._pos_refresh()

    def _pos_refresh(self):
        snap = self.backend.snapshot
        if not snap.positions:
            self.tree_pos.delete(*self.tree_pos.get_children())
            self.lbl_total.config(text="Portfolio empty")
            return

        def work():
            prices, source = self.backend.prices()
            return prices.iloc[-1], source

        def done(payload):
            last, source = payload
            df = snap.to_frame()
            df["last_price"] = [round(float(last.get(t, float("nan"))), 2) for t in df.index]
            df["market_value"] = (df["quantity"] * df["last_price"]).round(2)
            df["weight"] = (df["market_value"] / df["market_value"].sum()).round(4)
            df_to_tree(self.tree_pos, df, "ticker")
            total = df["market_value"].sum() + snap.cash
            note = "  [SYNTHETIC DATA - offline]" if source == "synthetic" else ""
            self.lbl_total.config(text=f"Total value: {total:,.2f}{note}")

        self.run_async(work, done, "Loading prices...")

    def _pos_add(self):
        tk_ = self.e_tk.get().strip().upper()
        if not tk_:
            return
        try:
            qty = float(self.e_qty.get().replace(",", ".") or 0)
            cost = float(self.e_cost.get().replace(",", ".") or 0)
        except ValueError:
            messagebox.showwarning("LFT", "Quantity and cost must be numbers.")
            return
        st = self.backend.state
        st["positions"] = [p for p in st["positions"] if p["ticker"] != tk_]
        if qty > 0:
            st["positions"].append({"ticker": tk_, "quantity": qty, "avg_cost": cost})
        self.backend._prices = None
        self.backend.save()
        self._pos_refresh()

    def _pos_del(self):
        sel = self.tree_pos.selection()
        if not sel:
            return
        tickers = {self.tree_pos.item(s)["values"][0] for s in sel}
        st = self.backend.state
        st["positions"] = [p for p in st["positions"] if p["ticker"] not in tickers]
        self.backend._prices = None
        self.backend.save()
        self._pos_refresh()

    def _pos_demo(self):
        self.backend._adopt_snapshot(broker.demo_positions())
        self._pos_refresh()

    def _pos_csv(self):
        path = filedialog.askopenfilename(
            title="Positions CSV (ticker,quantity,avg_cost)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.backend._adopt_snapshot(broker.from_csv(path))
            self._pos_refresh()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("LFT", f"Import failed: {exc}")

    def _pos_ibkr(self):
        port = self.backend.state["settings"].get("ibkr_port", 7497)

        def work():
            return self._ibkr_client().fetch()

        def done(snap):
            self.backend._adopt_snapshot(snap)
            messagebox.showinfo("LFT",
                                f"Imported {len(snap.positions)} positions from IBKR "
                                f"(cash {snap.cash:,.2f}).")
            self._pos_refresh()

        self.run_async(work, done, f"Connecting to IBKR on port {port} (read-only)...")

    def _set_cash(self):
        try:
            self.backend.state["cash"] = float(self.e_cash.get().replace(",", "."))
            self.backend.save()
            self._pos_refresh()
        except ValueError:
            messagebox.showwarning("LFT", "Cash must be a number.")

    # -------------------------------------------------------- 2 market data

    def page_market_data(self):
        if self.need_positions():
            return
        ttk.Button(self.content, text="Compute metrics",
                   command=self._market_run).pack(anchor="w")
        self.tree_market = make_tree(self.content, height=11)
        ttk.Label(self.content, text="Correlation matrix - the heart of "
                  "diversification (blue = moves together, red = opposite):"
                  ).pack(anchor="w", pady=(8, 0))
        self.chart_corr = ttk.Frame(self.content)
        self.chart_corr.pack(fill="both", expand=True)
        self._market_run()

    def _market_run(self):
        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            stats = metrics.summary(asset_returns, bench, s["risk_free"])
            port_r = (asset_returns * weights.reindex(asset_returns.columns)
                      .fillna(0)).sum(axis=1)
            stats["MyPortfolio"] = metrics.summary(
                port_r.rename("MyPortfolio"), bench, s["risk_free"]).iloc[:, 0]
            corr = asset_returns.corr()
            return stats.round(4), corr

        def done(payload):
            stats, corr = payload
            df_to_tree(self.tree_market, stats, "metric")
            n = len(corr)
            fig = Figure(figsize=(7.5, max(2.8, 0.42 * n + 1.2)), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(n), corr.columns, rotation=45,
                          ha="right", fontsize=8)
            ax.set_yticks(range(n), corr.index, fontsize=8)
            for i in range(n):
                for j in range(n):
                    v = corr.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(v) > 0.6 else "black")
            fig.colorbar(im, shrink=0.85)
            fig.tight_layout()
            for w_ in self.chart_corr.winfo_children():
                w_.destroy()
            self.embed_chart(fig, self.chart_corr)

        self.run_async(work, done)

    # ------------------------------------------------------- 3 health check

    def page_health_check(self):
        if self.need_positions():
            return
        self.tree_health = make_tree(self.content, height=9)
        self.txt_flags = tk.Text(self.content, height=4, wrap="word",
                                 font=("Segoe UI", 10))
        self.txt_flags.pack(fill="x", pady=6)
        self.chart_health = ttk.Frame(self.content)
        self.chart_health.pack(fill="both", expand=True)

        def work():
            from quantfolio import regime
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            table, flags = advisor.health_check(weights, asset_returns, bench,
                                                s["risk_free"])
            _, cov = opt.annualized_inputs(asset_returns, shrinkage=True)
            rc = advisor.risk_contributions(weights, cov)
            regimes = regime.classify_regimes(bench)
            cur, run = regime.current_regime(regimes)
            trans = regime.transition_matrix(regimes)
            exp_dur = regime.expected_regime_duration(trans, cur)
            bench_curve = (1 + bench).cumprod()
            flags = flags + [f"MARKET REGIME: currently '{cur}' for {run} days "
                             f"(typical stay: {exp_dur:.0f} days)."]
            return table, flags, rc, regimes, bench_curve

        def done(payload):
            from quantfolio.regime import REGIME_COLORS
            table, flags, rc, regimes, bench_curve = payload
            df_to_tree(self.tree_health, table.to_frame("value"), "check")
            self.txt_flags.delete("1.0", "end")
            self.txt_flags.insert("1.0", "\n".join(f"- {f}" for f in flags))

            fig = Figure(figsize=(7.5, 4.2), dpi=100)
            ax1 = fig.add_subplot(121)
            rc_plot = rc[rc["weight"] > 0].iloc[::-1]
            y = range(len(rc_plot))
            ax1.barh([i + 0.2 for i in y], rc_plot["weight"], height=0.38,
                     label="Weight (money)", color="#9db4c8")
            ax1.barh([i - 0.2 for i in y], rc_plot["risk_contribution"],
                     height=0.38, label="Risk contribution", color="#d85a30")
            ax1.set_yticks(list(y), rc_plot.index)
            ax1.set_title("Who carries the risk?", fontsize=10)
            ax1.legend(fontsize=7)
            ax1.grid(alpha=0.3, axis="x")

            ax2 = fig.add_subplot(122)
            ax2.plot(bench_curve.index, bench_curve, color="black", lw=1.1)
            spans, start_d, cur_lbl = [], None, None
            for d, lbl in regimes.items():
                if lbl != cur_lbl:
                    if cur_lbl is not None:
                        spans.append((start_d, d, cur_lbl))
                    start_d, cur_lbl = d, lbl
            spans.append((start_d, regimes.index[-1], cur_lbl))
            for a, b, lbl in spans:
                ax2.axvspan(a, b, color=REGIME_COLORS.get(lbl, "#ccc"), alpha=0.25)
            ax2.set_title("Market regimes (benchmark)", fontsize=10)
            ax2.tick_params(axis="x", labelsize=7)
            ax2.grid(alpha=0.3)
            handles = [Patch(facecolor=c, alpha=0.4, label=l)
                       for l, c in REGIME_COLORS.items()]
            ax2.legend(handles=handles, fontsize=6, loc="upper left")

            fig.tight_layout()
            for w_ in self.chart_health.winfo_children():
                w_.destroy()
            self.embed_chart(fig, self.chart_health)

        self.run_async(work, done)

    # ------------------------------------------------------- 4 optimization

    def page_optimization(self):
        if self.need_positions():
            return
        bl_row = ttk.Frame(self.content)
        bl_row.pack(fill="x", pady=(0, 6))
        ttk.Label(bl_row, text="My view (Black-Litterman):").pack(side="left")
        self.e_bl_ticker = ttk.Entry(bl_row, width=8)
        self.e_bl_ticker.pack(side="left", padx=4)
        ttk.Label(bl_row, text="will return").pack(side="left")
        self.e_bl_ret = ttk.Entry(bl_row, width=6)
        self.e_bl_ret.insert(0, "10")
        self.e_bl_ret.pack(side="left", padx=4)
        ttk.Label(bl_row, text="% per year, confidence:").pack(side="left")
        self.cb_bl_conf = ttk.Combobox(bl_row, state="readonly", width=8,
                                       values=["low", "medium", "high"])
        self.cb_bl_conf.current(1)
        self.cb_bl_conf.pack(side="left", padx=4)
        ttk.Button(bl_row, text="Recompute with my view",
                   command=lambda: self._opt_run(with_view=True)
                   ).pack(side="left", padx=8)
        ttk.Button(bl_row, text="Reset",
                   command=lambda: self._opt_run(with_view=False)
                   ).pack(side="left")
        self.tree_opt = make_tree(self.content, height=8)
        self.chart_opt = ttk.Frame(self.content)
        self.chart_opt.pack(fill="both", expand=True)
        self._opt_run(with_view=False)

    def _opt_run(self, with_view=False):
        view, confidence = None, 0.5
        if with_view:
            tk_ = self.e_bl_ticker.get().strip().upper()
            if not tk_:
                messagebox.showinfo("LFT", "Enter a ticker for your view "
                                    "(one you hold, e.g. AAPL).")
                return
            try:
                view = {tk_: float(self.e_bl_ret.get().replace(",", ".")) / 100}
            except ValueError:
                messagebox.showwarning("LFT", "Expected return must be a number.")
                return
            confidence = {"low": 0.15, "medium": 0.5, "high": 0.9}[
                self.cb_bl_conf.get()]

        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            strategies, mu, cov = self.backend._target_strategies(asset_returns)
            if view:
                unknown = [t for t in view if t not in cov.index]
                if unknown:
                    raise ValueError(f"View on {unknown[0]}: not in your "
                                     f"portfolio universe.")
                mkt_w = weights.reindex(cov.index).fillna(0.0)
                if mkt_w.sum() <= 0:
                    mkt_w = opt.equal_weights(list(cov.index))
                mu_bl, cov_bl = opt.black_litterman(cov, mkt_w, views=view,
                                                    view_confidence=confidence)
                n = len(mu_bl)
                cap = max(s["max_weight"], 1.0 / n + 0.01)
                strategies["BL MaxSharpe (my view)"] = opt.max_sharpe_weights(
                    mu_bl, cov_bl, s["risk_free"],
                    opt.weight_bounds(n, cap))
            alloc = pd.DataFrame(strategies)
            alloc["Current"] = weights.reindex(alloc.index).fillna(0.0)
            frontier = opt.efficient_frontier(mu, cov, n_points=25)
            cloud = opt.random_portfolios(mu, cov, n=2500)
            pts = {n: (opt.portfolio_volatility(w.values, cov),
                       opt.portfolio_return(w.values, mu))
                   for n, w in strategies.items()}
            wv = weights.reindex(mu.index).fillna(0.0).values
            pts["Current"] = (opt.portfolio_volatility(wv, cov),
                              opt.portfolio_return(wv, mu))
            return alloc.round(4), frontier, cloud, pts

        def done(payload):
            alloc, frontier, cloud, pts = payload
            df_to_tree(self.tree_opt, alloc.T, "strategy")
            fig = Figure(figsize=(7.5, 3.8), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(cloud["volatility"], cloud["return"], s=6, alpha=0.3,
                       color="#9db4c8")
            ax.plot(frontier["volatility"], frontier["return"], color="crimson",
                    lw=2, label="Efficient frontier")
            markers = {"Current": ("o", "black")}
            for name, (v, r) in pts.items():
                mk, col = markers.get(name, ("D", None))
                ax.scatter([v], [r], marker=mk, s=60, label=name,
                           color=col, zorder=5, edgecolors="black")
            ax.set_xlabel("Annualized volatility")
            ax.set_ylabel("Expected annualized return")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            for w in self.chart_opt.winfo_children():
                w.destroy()
            self.embed_chart(fig, self.chart_opt)

        self.run_async(work, done, "Optimizing...")

    # ---------------------------------------------------------- 5 rebalance

    def page_rebalance(self):
        if self.need_positions():
            return
        row = ttk.Frame(self.content)
        row.pack(fill="x")
        ttk.Label(row, text="Target strategy:").pack(side="left")
        self.cb_target = ttk.Combobox(row, state="readonly", width=24, values=[
            "MaxSharpe (LW + cap)", "MinVol (LW + cap)", "RiskParity", "EqualWeight"])
        self.cb_target.current(0)
        self.cb_target.pack(side="left", padx=6)
        ttk.Label(row, text="Ignore trades below:").pack(side="left", padx=(12, 2))
        self.e_min_trade = ttk.Entry(row, width=8)
        self.e_min_trade.insert(0, "100")
        self.e_min_trade.pack(side="left")
        ttk.Button(row, text="Propose trades", command=self._rebalance_run
                   ).pack(side="left", padx=10)
        ttk.Button(row, text="Export CSV", command=self._rebalance_export
                   ).pack(side="left")
        self.lbl_rebal = ttk.Label(self.content, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_rebal.pack(anchor="w", pady=4)
        ttk.Label(self.content, foreground="#a33",
                  text="Nothing is executed automatically - review the list and "
                       "place the orders yourself with your broker."
                  ).pack(anchor="w")
        self.tree_rebal = make_tree(self.content, height=12)
        self._last_proposal = None

    def _rebalance_run(self):
        target_name = self.cb_target.get()
        try:
            min_trade = float(self.e_min_trade.get().replace(",", "."))
        except ValueError:
            min_trade = 100.0

        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            strategies, mu, cov = self.backend._target_strategies(asset_returns)
            target = strategies[target_name]
            mv = snap.market_value(last)
            prop = advisor.propose_rebalance(weights, target, last, mv,
                                             tc_bps=s["tc_bps"],
                                             min_trade_value=min_trade)
            return prop, advisor.turnover(prop, mv)

        def done(payload):
            prop, to = payload
            self._last_proposal = prop
            if prop.empty:
                self.lbl_rebal.config(text="Already aligned with the target.")
                self.tree_rebal.delete(*self.tree_rebal.get_children())
                return
            df_to_tree(self.tree_rebal, prop, "ticker")
            self.lbl_rebal.config(
                text=f"Turnover {to:.1%} | estimated costs "
                     f"{prop['est_cost'].sum():,.2f}")

        self.run_async(work, done, "Building trade list...")

    def _rebalance_export(self):
        if self._last_proposal is None or self._last_proposal.empty:
            messagebox.showinfo("LFT", "Run 'Propose trades' first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            initialfile="rebalance_proposal.csv")
        if path:
            self._last_proposal.to_csv(path)
            self.status.set(f"Saved: {path}")

    # --------------------------------------------------------- 6 projection

    def page_projection(self):
        if self.need_positions():
            return
        row = ttk.Frame(self.content)
        row.pack(fill="x")
        ttk.Label(row, text="Horizon (years):").pack(side="left")
        self.e_horizon = ttk.Entry(row, width=6)
        self.e_horizon.insert(0, "3")
        self.e_horizon.pack(side="left", padx=4)
        ttk.Button(row, text="Run simulation", command=self._projection_run
                   ).pack(side="left", padx=8)
        self.tree_mc = make_tree(self.content, height=10)
        self.chart_mc = ttk.Frame(self.content)
        self.chart_mc.pack(fill="both", expand=True)

    def _projection_run(self):
        try:
            years = float(self.e_horizon.get().replace(",", "."))
        except ValueError:
            years = 3.0

        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            w = weights.reindex(asset_returns.columns).fillna(0.0)
            horizon = int(years * 252)
            rf = s["risk_free"]; fee = s.get("fee_annual", 0.005); infl = s.get("inflation", 0.025)
            gbm = mc.simulate_gbm(asset_returns, w, value, horizon, n_sims=20000,
                                  risk_free=rf, cost_annual=fee, inflation=infl)
            boot = mc.simulate_bootstrap(asset_returns, w, value, horizon, n_sims=20000,
                                         risk_free=rf, cost_annual=fee, inflation=infl)
            summary = pd.concat([gbm.summary(), boot.summary()], axis=1)
            summary.columns = ["GBM", "Bootstrap"]
            import numpy as np
            days = np.arange(gbm.paths.shape[0])
            bands = {p: np.percentile(gbm.paths, p, axis=1) for p in (5, 50, 95)}
            return summary, days, bands, value

        def done(payload):
            summary, days, bands, value = payload
            df_to_tree(self.tree_mc, summary, "measure")
            fig = Figure(figsize=(7.5, 3.4), dpi=100)
            ax = fig.add_subplot(111)
            ax.fill_between(days, bands[5], bands[95], alpha=0.15, color="#2a78d6")
            ax.plot(days, bands[50], color="#2a78d6", lw=2, label="Median")
            ax.plot(days, bands[5], color="#2a78d6", lw=1, ls=":", label="P5 / P95")
            ax.plot(days, bands[95], color="#2a78d6", lw=1, ls=":")
            ax.axhline(value, color="black", lw=1, ls="--")
            ax.set_xlabel("Trading days")
            ax.set_ylabel("Portfolio value")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            for w in self.chart_mc.winfo_children():
                w.destroy()
            self.embed_chart(fig, self.chart_mc)

        self.run_async(work, done, "Simulating 40,000 realistic paths...")

    # ----------------------------------------------------------- 7 backtest

    def page_backtest(self):
        if self.need_positions():
            return
        ttk.Label(self.content,
                  text="Honest walk-forward comparison on your tickers "
                       "(can take 30-60 seconds).").pack(anchor="w")
        ttk.Button(self.content, text="Run backtest",
                   command=self._backtest_run).pack(anchor="w", pady=4)
        self.tree_bt = make_tree(self.content, height=6)
        self.chart_bt = ttk.Frame(self.content)
        self.chart_bt.pack(fill="both", expand=True)

    def _backtest_run(self):
        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            held = list(asset_returns.columns)
            cap = max(s["max_weight"], 1 / len(held) + 0.01)
            strategies = {
                "MaxSharpe WF": wf.make_max_sharpe(s["risk_free"], cap),
                "MinVol WF": wf.make_min_vol(cap),
                "EqualWeight 1/N": wf.make_equal_weight(),
            }
            curves, _ = wf.compare_walk_forward(prices[held], strategies,
                                                lookback=s["lookback"],
                                                tc_bps=s["tc_bps"])
            curves[s["benchmark"]] = (10_000 * (1 + bench.reindex(curves.index))
                                      .cumprod())
            stats = metrics.summary(curves.pct_change().dropna(),
                                    bench.reindex(curves.index), s["risk_free"])
            rows = ["CAGR", "Annualized volatility", "Sharpe", "Max drawdown", "Calmar"]
            return stats.loc[rows].round(4), curves

        def done(payload):
            stats, curves = payload
            df_to_tree(self.tree_bt, stats, "metric")
            fig = Figure(figsize=(7.5, 3.4), dpi=100)
            ax = fig.add_subplot(111)
            for col in curves.columns:
                ax.plot(curves.index, curves[col], lw=1.3, label=col)
            ax.set_yscale("log")
            ax.set_ylabel("Value (log scale)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            for w in self.chart_bt.winfo_children():
                w.destroy()
            self.embed_chart(fig, self.chart_bt)

        self.run_async(work, done, "Running walk-forward backtest (be patient)...")

    # ---------------------------------------------------------- performance

    def page_performance(self):
        ttk.Label(self.content, wraplength=880, justify="left", text=(
            "Your PERSONAL return - not the market's. Import your transaction "
            "history (CSV with columns: date,ticker,quantity,price,fees - "
            "negative quantity = sell) and get your money-weighted annual "
            "return (XIRR) plus realized/unrealized P&L per position."
        )).pack(anchor="w")
        row = ttk.Frame(self.content)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="Import transactions CSV",
                   command=self._perf_load).pack(side="left")
        ttk.Button(row, text="Create template CSV",
                   command=self._perf_template).pack(side="left", padx=8)
        self.lbl_perf = ttk.Label(self.content, text="",
                                  font=("Segoe UI", 10, "bold"))
        self.lbl_perf.pack(anchor="w", pady=4)
        self.tree_perf_sum = make_tree(self.content, height=7)
        self.tree_perf = make_tree(self.content, height=9)
        last_path = self.backend.state.get("transactions_csv")
        if last_path:
            self._perf_run(last_path)

    def _perf_template(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="transactions.csv")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,ticker,quantity,price,fees\n"
                    "2024-01-15,AAPL,10,185.50,1.00\n"
                    "2024-03-02,MSFT,5,410.00,1.00\n"
                    "2024-09-10,AAPL,-4,228.00,1.00\n")
        self.status.set(f"Template written: {path} - fill it with your real "
                        f"trades, then use Import.")

    def _perf_load(self):
        path = filedialog.askopenfilename(
            title="Transactions CSV (date,ticker,quantity,price,fees)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._perf_run(path)

    def _perf_run(self, path):
        def work():
            tx = perf.load_transactions_csv(path)
            tickers = sorted(tx["ticker"].unique())
            s = self.backend.state["settings"]
            prices, source = self.backend.store.get_prices(
                tickers, s["start"], s["end"])
            last = prices.iloc[-1]
            summary = perf.performance_summary(tx, last)
            pnl = perf.pnl_report(tx, last)
            return summary, pnl, source

        def done(payload):
            summary, pnl, source = payload
            self.backend.state["transactions_csv"] = path
            self.backend.save()
            df_to_tree(self.tree_perf_sum, summary.to_frame("value"), "measure")
            df_to_tree(self.tree_perf, pnl, "ticker")
            mwr = summary["Money-weighted return (ann.)"]
            note = " [SYNTHETIC PRICES - offline]" if source == "synthetic" else ""
            self.lbl_perf.config(
                text=f"Your personal annual return (XIRR): {mwr:.1%}{note}")

        self.run_async(work, done, "Computing your P&L...")

    # ---------------------------------------------------------- 8 AI analyst

    def page_ai_analyst(self):
        if self.need_positions():
            return
        analyst = AIAnalyst()
        mode = ("Claude API" if analyst.api_key
                else "offline rule engine (set ANTHROPIC_API_KEY for the full AI)")
        ttk.Label(self.content, text=f"Mode: {mode}").pack(anchor="w")
        ttk.Button(self.content, text="Analyze my portfolio",
                   command=self._ai_run).pack(anchor="w", pady=4)
        self.txt_ai = tk.Text(self.content, wrap="word", font=("Segoe UI", 10))
        self.txt_ai.pack(fill="both", expand=True, pady=6)

    def _ai_run(self):
        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            port_r = (asset_returns * weights.reindex(asset_returns.columns)
                      .fillna(0)).sum(axis=1)
            pstats = metrics.summary(port_r.rename("MyPortfolio"), bench, s["risk_free"])
            table, flags = advisor.health_check(weights, asset_returns, bench,
                                                s["risk_free"])
            return AIAnalyst().explain(
                pstats, table,
                context=f"Current portfolio: {dict(weights.round(3))} | "
                        f"flags: {'; '.join(flags)}")

        def done(text):
            self.txt_ai.delete("1.0", "end")
            self.txt_ai.insert("1.0", text)

        self.run_async(work, done, "Analyzing...")

    # ----------------------------------------------------- IBKR connection

    def page_ibkr_connection(self):
        s = self.backend.state["settings"]
        port = int(s.get("ibkr_port", 7497))
        mode = "paper" if port == 7497 else "live" if port == 7496 else "custom"
        watch = int(s.get("ibkr_watch_minutes", 0))

        self.var_ibkr_mode = tk.StringVar(value=mode)
        self.var_ibkr_host = tk.StringVar(value=str(s.get("ibkr_host", "127.0.0.1")))
        self.var_ibkr_port = tk.StringVar(value=str(port))
        self.var_ibkr_client_id = tk.StringVar(value=str(s.get("ibkr_client_id", 42)))
        self.var_ibkr_autosync = tk.BooleanVar(
            value=bool(int(s.get("ibkr_autosync", 0))))
        self.var_ibkr_watch = tk.BooleanVar(value=watch > 0)
        self.var_ibkr_watch_minutes = tk.StringVar(value=str(watch or 5))
        self.var_ibkr_status = tk.StringVar(value="Not tested in this session.")

        grid = ttk.Frame(self.content)
        grid.pack(anchor="w", fill="x")
        ttk.Label(grid, text="Account", width=22).grid(
            row=0, column=0, sticky="w", pady=4)
        modes = ttk.Frame(grid)
        modes.grid(row=0, column=1, sticky="w", pady=4)
        ttk.Radiobutton(
            modes, text="Paper", value="paper", variable=self.var_ibkr_mode,
            command=self._ibkr_apply_account_mode,
        ).pack(side="left")
        ttk.Radiobutton(
            modes, text="Live", value="live", variable=self.var_ibkr_mode,
            command=self._ibkr_apply_account_mode,
        ).pack(side="left", padx=(12, 0))

        fields = [
            ("Host", self.var_ibkr_host),
            ("API port", self.var_ibkr_port),
            ("Client ID", self.var_ibkr_client_id),
        ]
        for row, (label, variable) in enumerate(fields, start=1):
            ttk.Label(grid, text=label, width=22).grid(
                row=row, column=0, sticky="w", pady=4)
            ttk.Entry(grid, textvariable=variable, width=18).grid(
                row=row, column=1, sticky="w", pady=4)

        ttk.Checkbutton(
            grid, text="Sync when LFT starts", variable=self.var_ibkr_autosync,
        ).grid(row=4, column=1, sticky="w", pady=(10, 4))
        ttk.Checkbutton(
            grid, text="Monitor positions while LFT is open",
            variable=self.var_ibkr_watch,
        ).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(grid, text="Refresh interval (minutes)", width=22).grid(
            row=6, column=0, sticky="w", pady=4)
        ttk.Entry(grid, textvariable=self.var_ibkr_watch_minutes, width=18).grid(
            row=6, column=1, sticky="w", pady=4)

        actions = ttk.Frame(self.content)
        actions.pack(anchor="w", pady=(14, 8))
        ttk.Button(actions, text="Save connection",
                   command=self._ibkr_save_connection).pack(side="left")
        ttk.Button(actions, text="Connect and sync now",
                   command=self._ibkr_sync_now).pack(side="left", padx=8)

        ttk.Separator(self.content, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(self.content, textvariable=self.var_ibkr_status,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=4)
        ttk.Label(
            self.content, foreground="#666", justify="left",
            text=("TWS: Global Configuration > API > Settings > Enable ActiveX "
                  "and Socket Clients.\nLFT opens read-only sessions and cannot "
                  "place orders."),
        ).pack(anchor="w", pady=4)

    def _ibkr_apply_account_mode(self):
        port = 7497 if self.var_ibkr_mode.get() == "paper" else 7496
        self.var_ibkr_port.set(str(port))

    def _ibkr_save_connection(self, notify=True):
        try:
            host = self.var_ibkr_host.get().strip()
            port = int(self.var_ibkr_port.get().strip())
            client_id = int(self.var_ibkr_client_id.get().strip())
            watch = int(self.var_ibkr_watch_minutes.get().strip())
            if not host:
                raise ValueError("host is required")
            if not 1 <= port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            if client_id < 0:
                raise ValueError("client ID must be zero or greater")
            if self.var_ibkr_watch.get() and watch < 1:
                raise ValueError("refresh interval must be at least 1 minute")
        except ValueError as exc:
            messagebox.showwarning("LFT", f"Invalid IBKR setting: {exc}")
            return False

        s = self.backend.state["settings"]
        s.update({
            "ibkr_host": host,
            "ibkr_port": port,
            "ibkr_client_id": client_id,
            "ibkr_autosync": int(self.var_ibkr_autosync.get()),
            "ibkr_watch_minutes": watch if self.var_ibkr_watch.get() else 0,
        })
        self.backend.save()
        self._schedule_ibkr_watch()
        if notify:
            self.var_ibkr_status.set("Connection settings saved.")
            self.status.set("IBKR connection settings saved.")
        return True

    def _ibkr_sync_now(self):
        if not self._ibkr_save_connection(notify=False):
            return
        s = self.backend.state["settings"]
        endpoint = f"{s['ibkr_host']}:{s['ibkr_port']}"
        self.var_ibkr_status.set(f"Connecting to {endpoint}...")

        def work():
            try:
                return self._ibkr_client().fetch(), None
            except Exception as exc:  # noqa: BLE001
                return None, exc

        def done(payload):
            snap, error = payload
            if error is not None:
                self.var_ibkr_status.set(f"Connection failed: {error}")
                messagebox.showerror("LFT", str(error))
                self.after_idle(
                    lambda: self.status.set("IBKR connection failed."))
                return
            self.backend._adopt_snapshot(snap)
            self.var_ibkr_status.set(
                f"Sync complete: {len(snap.positions)} positions, "
                f"cash {snap.cash:,.2f}, net liquidation "
                f"{snap.net_liquidation:,.2f}.")
            self.after_idle(lambda: self.status.set("IBKR sync complete."))

        self.run_async(work, done,
                       f"Connecting to IBKR on {endpoint} (read-only)...")

    # ----------------------------------------------------------- settings

    def page_settings(self):
        s = self.backend.state["settings"]
        self._setting_entries = {}
        grid = ttk.Frame(self.content)
        grid.pack(anchor="w")
        for i, (k, v) in enumerate(s.items()):
            ttk.Label(grid, text=k, width=14).grid(row=i, column=0, sticky="w", pady=3)
            e = ttk.Entry(grid, width=16)
            e.insert(0, str(v))
            e.grid(row=i, column=1, pady=3)
            self._setting_entries[k] = e
        ttk.Button(grid, text="Save settings", command=self._settings_save
                   ).grid(row=len(s), column=1, pady=10, sticky="e")
        ttk.Label(self.content, foreground="#666", justify="left", text=(
            "benchmark: comparison index (e.g. SPY)\n"
            "start / end: analysis period (YYYY-MM-DD)\n"
            "risk_free: annual risk-free rate (0.03 = 3%)\n"
            "tc_bps: transaction costs in basis points\n"
            "lookback: walk-forward estimation window (trading days)\n"
            "max_weight: per-asset cap for optimized strategies\n"
            "ibkr_port: 7497 paper account, 7496 live account")).pack(anchor="w", pady=8)

    def _settings_save(self):
        s = self.backend.state["settings"]
        casts = {"benchmark": str, "start": str, "end": str,
                 "risk_free": float, "tc_bps": float, "lookback": int,
                 "max_weight": float, "ibkr_host": str, "ibkr_port": int,
                 "ibkr_client_id": int, "ibkr_autosync": int,
                 "ibkr_watch_minutes": int}
        try:
            for k, e in self._setting_entries.items():
                raw = e.get().strip()
                s[k] = casts.get(k, str)(raw.upper() if k == "benchmark" else raw)
        except ValueError as exc:
            messagebox.showwarning("LFT", f"Invalid value: {exc}")
            return
        self.backend.save()
        self.backend._prices = None
        self._schedule_ibkr_watch()
        self.status.set("Settings saved.")


if __name__ == "__main__":
    Gui().mainloop()
