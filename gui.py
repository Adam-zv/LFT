"""
LFT - desktop application (windowed edition).

A real standalone window: sidebar navigation on the left, pages on the
right, tables, input forms and charts embedded in the window.

Launch:  python gui.py        (or double-click LFT.bat)

Reuses the exact same engine and the same saved state (app_state.json)
as the console edition (app.py) - you can switch between the two freely.
"""

from __future__ import annotations

import ctypes


def enable_windows_dpi_awareness():
    """Prevent Windows from bitmap-scaling the Tk window into a blur."""
    try:
        per_monitor_v2 = ctypes.c_void_p(-4)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(per_monitor_v2):
            return
    except (AttributeError, OSError, ValueError):
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError, ValueError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError, ValueError):
        pass


# DPI context must be fixed before importing Tkinter or a GUI backend.
enable_windows_dpi_awareness()

from concurrent.futures import CancelledError
import itertools
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from app import App                      # tested backend: state, data, strategies
from quantfolio import data, metrics, optimization as opt
from quantfolio import montecarlo as mc, walkforward as wf, advisor, broker
from quantfolio.ai_analyst import AIAnalyst
from quantfolio import performance as perf

PAGES = ["Portfolio", "Market data", "Health check", "Optimization",
         "Rebalance", "Projection", "Backtest", "Performance", "AI analyst",
         "IBKR connection", "Settings"]

APP_BG = "#e2e3e5"
PLOT_BG = "#f8f9fb"
SIDEBAR_BG = "#1f2732"
SIDEBAR_ACTIVE = "#324153"
TEXT = "#17202b"
MUTED = "#596575"
ACCENT = "#2a78d6"
BORDER = "#aeb5bd"

# ------------------------------------------------------------ small helpers

def df_to_tree(tree: ttk.Treeview, df: pd.DataFrame, index_name: str = ""):
    """Fill a Treeview from a DataFrame (index shown as first column)."""
    tree.delete(*tree.get_children())
    cols = [index_name or (df.index.name or "")] + [str(c) for c in df.columns]
    tree.configure(columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        # stretch=False: columns keep a readable fixed width instead of
        # spreading across the whole window (the "giant gap" effect).
        tree.column(c, width=max(70, min(160, 11 * len(c))), anchor="e",
                    stretch=False)
    tree.column(cols[0], anchor="w", width=max(150, min(260, 9 * max(
        (len(str(i)) for i in df.index), default=12))), stretch=False)
    for idx, row in df.iterrows():
        vals = [idx] + [("" if pd.isna(v) else round(v, 4) if isinstance(v, float) else v)
                        for v in row]
        tree.insert("", "end", values=vals)


def make_tree(parent, height=12) -> ttk.Treeview:
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True, pady=4)
    tree = ttk.Treeview(frame, show="headings", height=height)
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree


class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LFT - LeFort - Portfolio Management")
        self.update_idletasks()
        self._display_dpi = max(96.0, min(240.0, self.winfo_fpixels("1i")))
        self._ui_scale = self._display_dpi / 96.0
        self.tk.call("tk", "scaling", self._display_dpi / 72.0)
        width = min(int(self.winfo_screenwidth() * 0.92),
                    round(1180 * self._ui_scale))
        height = min(int(self.winfo_screenheight() * 0.90),
                     round(760 * self._ui_scale))
        self.geometry(f"{width}x{height}")
        self.minsize(min(width, round(980 * self._ui_scale)),
                     min(height, round(640 * self._ui_scale)))
        # Keep figure pixel sizes close to the LOGICAL UI size: at high
        # display DPI a large plot_dpi produces figures bigger than the
        # window itself, so charts looked cropped / "too zoomed".
        self._plot_dpi = max(96, min(125, round(self._display_dpi * 0.85)))
        self.configure(bg=APP_BG)
        self._configure_style()
        self._set_lft_icon()

        self.backend = App()             # engine + persisted state
        self._queue: queue.Queue = queue.Queue()
        self._tasks: dict[str, dict | None] = {
            "compute": None,
            "ibkr": None,
            "background": None,
        }
        self._current_page = None
        self._task_ids = itertools.count(1)
        self._ibkr_watch_job = None
        self._result_cache = {}

        # ---- layout: sidebar | content
        sidebar = tk.Frame(self, bg=SIDEBAR_BG,
                           width=round(180 * self._ui_scale))
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        brand = tk.Frame(sidebar, bg=SIDEBAR_BG)
        brand.pack(fill="x", pady=(18, 16))
        tk.Label(brand, text="LFT", bg=SIDEBAR_BG, fg="white",
                 font=("Segoe UI", 19, "bold")).pack(anchor="w", padx=18)
        self._nav_buttons = {}
        for name in PAGES:
            b = tk.Button(sidebar, text=name, relief="flat", anchor="w",
                          bg=SIDEBAR_BG, fg="#c9d4e0", activebackground=SIDEBAR_ACTIVE,
                          activeforeground="white", padx=18, pady=8,
                          font=("Segoe UI", 10),
                          command=lambda n=name: self.show_page(n))
            b.pack(fill="x")
            self._nav_buttons[name] = b
        tk.Label(sidebar, text="Educational tool.\nNot investment advice.",
                 bg=SIDEBAR_BG, fg="#8290a0", font=("Segoe UI", 8),
                 justify="left", padx=18, pady=14).pack(side="bottom", fill="x")

        self.status = tk.StringVar(value="Ready.")
        statusbar = tk.Label(self, textvariable=self.status, anchor="w",
                             bg="#f4f5f6", fg=TEXT, bd=1, relief="solid",
                             highlightthickness=0, padx=8, pady=3,
                             font=("Segoe UI", 9))
        statusbar.pack(side="bottom", fill="x")

        self.content = ttk.Frame(self, padding=14)
        self.content.pack(side="right", fill="both", expand=True)

        self.after(100, self._poll)
        self.show_page("Portfolio")
        if int(self.backend.state["settings"].get("ibkr_autosync", 0)):
            self.after(800, self._autosync_ibkr)
        self._schedule_ibkr_watch()
        self.after(1500, self._show_data_freshness)

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(family="Segoe UI", size=10)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkHeadingFont").configure(
                family="Segoe UI", size=10, weight="bold")
        except tk.TclError:
            pass

        style.configure(".", background=APP_BG, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("TFrame", background=APP_BG)
        style.configure("TLabel", background=APP_BG, foreground=TEXT)
        style.configure("TButton", background="#f1f2f4", foreground=TEXT,
                        bordercolor=BORDER, lightcolor="#ffffff",
                        darkcolor="#8e969f", borderwidth=1, padding=(10, 6),
                        relief="raised")
        style.map("TButton",
                  background=[("active", "#ffffff"), ("pressed", "#d8dadd"),
                              ("disabled", "#d7d9dc")],
                  foreground=[("disabled", "#808891")])
        style.configure("TEntry", fieldbackground="#ffffff", foreground=TEXT,
                        bordercolor=BORDER, insertcolor=TEXT, padding=4)
        style.configure("TCombobox", fieldbackground="#ffffff", foreground=TEXT,
                        bordercolor=BORDER, padding=4)
        style.configure("Treeview", background="#fbfcfd",
                        fieldbackground="#fbfcfd", foreground=TEXT,
                        bordercolor=BORDER, borderwidth=1,
                        rowheight=round(24 * self._ui_scale))
        style.configure("Treeview.Heading", background="#d5d9de",
                        foreground=TEXT, bordercolor=BORDER, padding=(6, 5),
                        font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", SIDEBAR_ACTIVE)],
                  foreground=[("selected", "white")])
        style.configure("TSeparator", background=BORDER)
        style.configure("Horizontal.TProgressbar", troughcolor="#c9cdd2",
                        background=ACCENT, bordercolor=BORDER,
                        lightcolor=ACCENT, darkcolor=ACCENT)

        mpl.rcParams.update({
            "font.family": "Segoe UI",
            "font.size": 9.5,
            "figure.facecolor": APP_BG,
            "axes.facecolor": PLOT_BG,
            "axes.edgecolor": "#7d8792",
            "axes.labelcolor": TEXT,
            "xtick.color": "#3f4a56",
            "ytick.color": "#3f4a56",
            "grid.color": "#b7bec6",
            "grid.linewidth": 0.7,
            "savefig.dpi": self._plot_dpi,
        })

    def _set_lft_icon(self):
        icon = tk.PhotoImage(width=32, height=32)
        icon.put(SIDEBAR_BG, to=(0, 0, 32, 32))
        patterns = {
            "L": ("100", "100", "100", "100", "111"),
            "F": ("111", "100", "110", "100", "100"),
            "T": ("111", "010", "010", "010", "010"),
        }
        x0, y0, pixel = 4, 10, 2
        for char in "LFT":
            for row, bits in enumerate(patterns[char]):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        x = x0 + col * pixel
                        y = y0 + row * pixel
                        icon.put("#ffffff", to=(x, y, x + pixel, y + pixel))
            x0 += 9
        icon.put(ACCENT, to=(0, 0, 32, 2))
        self.iconphoto(True, icon)
        self._lft_icon = icon

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
                       silent_errors=True, channel="background",
                       skip_if_busy=True)

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
                       silent_errors=True, channel="ibkr",
                       skip_if_busy=True)
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
                       silent_errors=True, channel="ibkr",
                       skip_if_busy=True)

    # ------------------------------------------------------ infrastructure

    def _poll(self):
        try:
            while True:
                event = self._queue.get_nowait()
                channel = event["channel"]
                current = self._tasks.get(channel)
                if current is None or current["id"] != event["task_id"]:
                    continue
                if event["type"] == "progress":
                    fraction = event["fraction"]
                    eta = event["eta"]
                    message = event["message"] or current["busy_msg"]
                    suffix = f" | ETA {eta:.0f}s" if eta is not None else ""
                    self.status.set(f"{message} ({fraction:.0%}){suffix}")
                    if current["on_progress"] is not None:
                        current["on_progress"](fraction, message, eta)
                    continue

                self._tasks[channel] = None
                status, payload = event["status"], event.get("payload")
                if status == "cancelled":
                    self.status.set("Computation cancelled.")
                    if current["on_cancel"] is not None:
                        current["on_cancel"]()
                elif status == "silent_err":
                    if current["on_error"] is not None:
                        current["on_error"](payload)
                    self.status.set(f"(background task failed: {payload})")
                elif status == "err":
                    if current["on_error"] is not None:
                        current["on_error"](payload)
                    self.status.set(f"Error: {payload}")
                    messagebox.showerror("LFT", str(payload))
                else:
                    self.status.set("Done.")
                    event["on_done"](payload)
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def run_async(self, work, on_done, busy_msg="Computing...",
                  silent_errors=False, *, channel="compute",
                  contextual=False, cancellable=False, on_progress=None,
                  on_cancel=None, on_error=None, skip_if_busy=False):
        self._tasks.setdefault(channel, None)
        if self._tasks[channel] is not None:
            if not skip_if_busy:
                label = "IBKR synchronization" if channel == "ibkr" else "computation"
                self.status.set(f"Please wait, a {label} is already running...")
            return None

        task_id = next(self._task_ids)
        cancel_event = threading.Event()
        started = time.perf_counter()
        self._tasks[channel] = {
            "id": task_id,
            "cancel_event": cancel_event,
            "busy_msg": busy_msg,
            "on_progress": on_progress,
            "on_cancel": on_cancel,
            "on_error": on_error,
            "cancellable": cancellable,
        }
        self.status.set(busy_msg)

        def progress(fraction, message=None):
            fraction = max(0.0, min(1.0, float(fraction)))
            elapsed = time.perf_counter() - started
            eta = (elapsed * (1.0 - fraction) / fraction
                   if 0.02 <= fraction < 1.0 else None)
            self._queue.put({
                "type": "progress", "channel": channel, "task_id": task_id,
                "fraction": fraction, "message": message, "eta": eta,
            })

        def worker():
            try:
                payload = (work(progress, cancel_event)
                           if contextual else work())
                if cancel_event.is_set():
                    raise CancelledError()
                status = "ok"
            except CancelledError:
                status, payload = "cancelled", None
            except Exception as exc:  # noqa: BLE001
                status = "silent_err" if silent_errors else "err"
                payload = exc
            self._queue.put({
                "type": "result", "channel": channel, "task_id": task_id,
                "status": status, "payload": payload, "on_done": on_done,
            })
        threading.Thread(target=worker, daemon=True).start()
        return cancel_event

    def cancel_task(self, channel="compute"):
        task = self._tasks.get(channel)
        if task is not None and task["cancellable"]:
            task["cancel_event"].set()
            self.status.set("Cancelling after the current calculation step...")

    def clear_content(self):
        def release_figures(widget):
            figure = getattr(widget, "_lft_figure", None)
            if figure is not None:
                try:
                    figure.set_layout_engine(None)
                except (AttributeError, ValueError):
                    pass
            for child in widget.winfo_children():
                release_figures(child)

        for w in self.content.winfo_children():
            release_figures(w)
            w.destroy()

    def show_page(self, name: str):
        # A page owns its computation. Navigating away requests cancellation
        # and immediately releases the UI slot; any late result is discarded
        # by the task-id check in _poll.
        current = self._tasks.get("compute")
        if current is not None:
            current["cancel_event"].set()
            self._tasks["compute"] = None
        self._current_page = name
        for n, b in self._nav_buttons.items():
            b.configure(bg=SIDEBAR_ACTIVE if n == name else SIDEBAR_BG)
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

    def new_figure(self, figsize):
        return Figure(figsize=figsize, dpi=self._plot_dpi, facecolor=APP_BG,
                      layout="constrained")

    def _calculation_signature(self, prices, feature, *extra):
        s = self.backend.state["settings"]
        positions = tuple(sorted(
            (p["ticker"], float(p["quantity"]), float(p.get("avg_cost", 0.0)))
            for p in self.backend.state["positions"]))
        market = (tuple(prices.columns), len(prices),
                  str(prices.index[0]), str(prices.index[-1]))
        settings = tuple(sorted((key, str(value)) for key, value in s.items()))
        return (feature, market, positions,
                float(self.backend.state.get("cash", 0.0)), settings, extra)

    def embed_chart(self, fig: Figure, parent=None, *, toolbar=False):
        host = ttk.Frame(parent or self.content)
        host._lft_figure = fig
        host.pack(fill="both", expand=True, pady=6)
        canvas = FigureCanvasTkAgg(fig, master=host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=6)
        if toolbar:
            nav = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
            nav.update()
            nav.pack(fill="x")
            canvas._lft_toolbar = nav
        return canvas

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

        self.run_async(work, done, f"Connecting to IBKR on port {port} (read-only)...",
                       channel="ibkr")

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
        controls = ttk.Frame(self.content)
        controls.pack(fill="x")
        ttk.Button(controls, text="Compute metrics",
                   command=self._market_run).pack(side="left")
        self.var_corr_cluster = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Cluster similar assets",
                        variable=self.var_corr_cluster,
                        command=self._market_run).pack(side="left", padx=12)
        ttk.Label(controls, text="Find ticker or pair:").pack(side="left")
        self.var_corr_search = tk.StringVar()
        search = ttk.Entry(controls, textvariable=self.var_corr_search, width=18)
        search.pack(side="left", padx=4)
        search.bind("<Return>", lambda _event: self._correlation_search())
        ttk.Button(controls, text="Find",
                   command=self._correlation_search).pack(side="left")
        self.var_data_provenance = tk.StringVar(value="Loading data provenance...")
        self.lbl_data_provenance = ttk.Label(
            self.content, textvariable=self.var_data_provenance, foreground=MUTED)
        self.lbl_data_provenance.pack(anchor="w", pady=(6, 0))
        self.tree_market = make_tree(self.content, height=8)
        ttk.Label(self.content, text="Correlation matrix - the heart of "
                  "diversification (red = moves together, blue = opposite):"
                  ).pack(anchor="w", pady=(8, 0))
        body = ttk.Panedwindow(self.content, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=(12, 0, 0, 0))
        body.add(left, weight=4)
        body.add(right, weight=1)
        self.chart_corr = ttk.Frame(left)
        self.chart_corr.pack(fill="both", expand=True)
        ttk.Label(right, text="Strongest relationships",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.var_corr_hover = tk.StringVar(
            value="Hover over a cell to inspect the pair.")
        ttk.Label(right, textvariable=self.var_corr_hover, foreground=MUTED,
                  wraplength=280, justify="left").pack(anchor="w", pady=(4, 6))
        self.tree_corr_pairs = make_tree(right, height=12)
        self._corr_search_artists = []
        self._market_run()

    def _market_run(self):
        clustered = bool(self.var_corr_cluster.get())

        def work():
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            stats = metrics.summary(asset_returns, bench, s["risk_free"])
            port_r = (asset_returns * weights.reindex(asset_returns.columns)
                      .fillna(0)).sum(axis=1)
            stats["MyPortfolio"] = metrics.summary(
                port_r.rename("MyPortfolio"), bench, s["risk_free"]).iloc[:, 0]
            corr = asset_returns.corr()
            store_stats = self.backend.store.stats()
            source = getattr(self.backend, "_source", "unknown")
            last_date = asset_returns.index.max().date().isoformat()
            if clustered and len(corr) > 2:
                from scipy.cluster.hierarchy import leaves_list, linkage
                from scipy.spatial.distance import squareform
                clean = corr.fillna(0.0).clip(-1.0, 1.0)
                distance = np.sqrt(np.maximum(0.0, (1.0 - clean.values) / 2.0))
                np.fill_diagonal(distance, 0.0)
                order = leaves_list(linkage(squareform(
                    distance, checks=False), method="average"))
                corr = corr.iloc[order, order]
            pairs = []
            for i in range(len(corr)):
                for j in range(i + 1, len(corr)):
                    value = float(corr.iat[i, j])
                    if np.isfinite(value):
                        pairs.append((corr.index[i], corr.columns[j], value))
            pairs.sort(key=lambda item: abs(item[2]), reverse=True)
            pair_df = pd.DataFrame(
                {"Correlation": [p[2] for p in pairs[:25]]},
                index=[f"{p[0]} / {p[1]}" for p in pairs[:25]])
            provenance = {
                "source": source, "last_date": last_date,
                "rows": store_stats["price_rows"],
                "flags": store_stats["quality_flags"],
            }
            return stats.round(4), corr, pair_df, provenance

        def done(payload):
            if not self.chart_corr.winfo_exists():
                return
            stats, corr, pair_df, provenance = payload
            df_to_tree(self.tree_market, stats, "metric")
            df_to_tree(self.tree_corr_pairs, pair_df.round(3), "pair")
            self.var_data_provenance.set(
                f"Source: {provenance['source']} | adjusted daily prices through "
                f"{provenance['last_date']} | {provenance['rows']:,} cached rows | "
                f"{provenance['flags']} quality flag(s)")
            self.lbl_data_provenance.configure(
                foreground="#9a2f2f" if provenance["source"] == "synthetic" else MUTED)
            n = len(corr)
            fig = self.new_figure((7.6, 5.6))
            ax = fig.add_subplot(111)
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            step = max(1, int(np.ceil(n / 28)))
            ticks = np.arange(0, n, step)
            ax.set_xticks(ticks, corr.columns[ticks], rotation=45,
                          ha="right", fontsize=7 if n > 30 else 8)
            ax.set_yticks(ticks, corr.index[ticks], fontsize=7 if n > 30 else 8)
            if n <= 20:
                for i in range(n):
                    for j in range(n):
                        value = corr.values[i, j]
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                                fontsize=7,
                                color="white" if abs(value) > 0.6 else "black")
            ax.set_title(f"Return correlations ({n} assets"
                         + (", clustered)" if clustered else ")"), fontsize=10)
            fig.colorbar(im, shrink=0.85)
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
                bbox={"boxstyle": "square,pad=0.35", "fc": "white",
                      "ec": BORDER, "alpha": 0.96}, fontsize=8,
                visible=False, zorder=10)
            for w_ in self.chart_corr.winfo_children():
                w_.destroy()
            canvas = self.embed_chart(fig, self.chart_corr, toolbar=True)
            self._corr_df = corr
            self._corr_ax = ax
            self._corr_canvas = canvas
            self._corr_search_artists = []

            def hover(event):
                if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                    annotation.set_visible(False)
                    canvas.draw_idle()
                    return
                j, i = int(round(event.xdata)), int(round(event.ydata))
                if not (0 <= i < n and 0 <= j < n):
                    return
                left_name, top_name = corr.index[i], corr.columns[j]
                value = float(corr.iat[i, j])
                annotation.xy = (j, i)
                annotation.set_text(f"{left_name} / {top_name}\nCorrelation: {value:+.3f}")
                annotation.set_visible(True)
                self.var_corr_hover.set(
                    f"{left_name} / {top_name}: correlation {value:+.3f}")
                canvas.draw_idle()

            canvas.mpl_connect("motion_notify_event", hover)

        self.run_async(work, done)

    def _correlation_search(self):
        if not hasattr(self, "_corr_df"):
            return
        names = [part.strip().upper() for part in
                 self.var_corr_search.get().replace(",", " ").split() if part.strip()]
        if not names:
            return
        missing = [name for name in names[:2] if name not in self._corr_df.index]
        if missing:
            self.var_corr_hover.set(f"Ticker not found: {missing[0]}")
            return
        for artist in self._corr_search_artists:
            artist.remove()
        self._corr_search_artists = []
        indices = [self._corr_df.index.get_loc(name) for name in names[:2]]
        if len(indices) == 1:
            i = indices[0]
            self._corr_search_artists.extend([
                self._corr_ax.add_patch(Rectangle(
                    (-0.5, i - 0.5), len(self._corr_df), 1, fill=False,
                    ec="#111111", lw=1.8, zorder=8)),
                self._corr_ax.add_patch(Rectangle(
                    (i - 0.5, -0.5), 1, len(self._corr_df), fill=False,
                    ec="#111111", lw=1.8, zorder=8)),
            ])
            self.var_corr_hover.set(f"Highlighted correlations for {names[0]}")
        else:
            i, j = indices
            value = float(self._corr_df.iat[i, j])
            for x, y in ((j, i), (i, j)):
                self._corr_search_artists.append(self._corr_ax.add_patch(
                    Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                              ec="#111111", lw=2.2, zorder=8)))
            self.var_corr_hover.set(
                f"{names[0]} / {names[1]}: correlation {value:+.3f}")
        self._corr_canvas.draw_idle()

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

            fig = self.new_figure((8.6, 4.4))
            ax1 = fig.add_subplot(121)
            rc_plot = rc[rc["weight"] > 0].iloc[::-1]
            y = range(len(rc_plot))
            ax1.barh([i + 0.2 for i in y], rc_plot["weight"], height=0.38,
                     label="Weight (money)", color="#9db4c8")
            ax1.barh([i - 0.2 for i in y], rc_plot["risk_contribution"],
                     height=0.38, label="Risk contribution", color="#d85a30")
            ax1.set_yticks(list(y), rc_plot.index)
            ax1.set_title("Who carries the risk?", fontsize=10)
            ax1.legend(fontsize=7, loc="lower right", framealpha=0.95)
            ax1.grid(alpha=0.3, axis="x")
            ax1.set_xlim(0, max(rc_plot["risk_contribution"].max(),
                                rc_plot["weight"].max()) * 1.18)

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
            ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
            ax2.margins(x=0.01)
            ax2.grid(alpha=0.3)
            handles = [Patch(facecolor=c, alpha=0.4, label=l)
                       for l, c in REGIME_COLORS.items()]
            # Legend BELOW the plot: inside the axes it covered the regime
            # bands and the price curve on wide screens.
            ax2.legend(handles=handles, fontsize=7, loc="upper center",
                       bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)

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
        self.var_opt_hover = tk.StringVar(
            value="Adaptive 80-120 point frontier with bootstrap input uncertainty.")
        ttk.Label(self.content, textvariable=self.var_opt_hover,
                  foreground=MUTED).pack(anchor="w", pady=(2, 0))
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

        def work(progress, cancel_event):
            progress(0.01, "Preparing optimization inputs")
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            signature = self._calculation_signature(
                prices, "optimization", tuple(sorted((view or {}).items())),
                confidence)
            cached = self._result_cache.get("optimization")
            if cached is not None and cached[0] == signature:
                progress(1.0, "Using cached optimization")
                return cached[1]
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
            n = len(mu)
            cap = max(s["max_weight"], 1.0 / n + 0.01)
            bounds = opt.weight_bounds(n, cap)
            frontier_points = 120 if n <= 30 else 100 if n <= 100 else 80
            frontier = opt.efficient_frontier(
                mu, cov, n_points=frontier_points, bounds=bounds,
                progress_callback=lambda f: progress(
                    0.04 + 0.61 * f, "Computing constrained frontier"),
                cancel_event=cancel_event)
            progress(0.68, "Sampling feasible portfolios")
            cloud = opt.random_portfolios(
                mu, cov, n=2500, bounds=bounds)
            n_boot = 30 if n <= 30 else 12 if n <= 100 else 6
            band = opt.frontier_uncertainty_band(
                asset_returns, frontier, n_boot=n_boot,
                mean_shrinkage=0.5,
                progress_callback=lambda f: progress(
                    0.70 + 0.28 * f, "Bootstrapping uncertainty band"),
                cancel_event=cancel_event)
            pts = {n: (opt.portfolio_volatility(w.values, cov),
                       opt.portfolio_return(w.values, mu))
                   for n, w in strategies.items()}
            wv = weights.reindex(mu.index).fillna(0.0).values
            pts["Current"] = (opt.portfolio_volatility(wv, cov),
                               opt.portfolio_return(wv, mu))
            progress(1.0, "Optimization ready")
            result = (alloc.round(4), frontier, cloud, band, pts, cap,
                      n_boot, frontier_points)
            self._result_cache["optimization"] = (signature, result)
            return result

        def done(payload):
            if not self.chart_opt.winfo_exists():
                return
            (alloc, frontier, cloud, band, pts, cap, n_boot,
             frontier_points) = payload
            df_to_tree(self.tree_opt, alloc.T, "strategy")
            fig = self.new_figure((8.6, 4.6))
            ax = fig.add_subplot(111)
            ax.scatter(cloud["volatility"], cloud["return"], s=6, alpha=0.3,
                       color="#9db4c8")
            order = np.argsort(frontier["volatility"].to_numpy())
            band_ordered = band.iloc[order]
            ax.fill_between(
                band_ordered["volatility"], band_ordered["return_p10"],
                band_ordered["return_p90"], color="#cf6679", alpha=0.16,
                label=f"Bootstrap P10-P90 ({n_boot} resamples)")
            frontier_line, = ax.plot(
                frontier["volatility"], frontier["return"], color="crimson",
                lw=2.2, label="Efficient frontier", picker=7)
            markers = {"Current": ("o", "black")}
            for name, (v, r) in pts.items():
                mk, col = markers.get(name, ("D", None))
                ax.scatter([v], [r], marker=mk, s=46, label=name,
                           color=col, zorder=5, edgecolors="black")
            ax.set_xlabel("Annualized volatility")
            ax.set_ylabel("Expected annualized return")
            ax.set_title(f"Constrained frontier | {frontier_points} targets | "
                         f"{cap:.0%} max per asset",
                         fontsize=10)
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
            # Legend OUTSIDE the axes (right): inside, it covered half of the
            # frontier and the strategy markers.
            ax.legend(fontsize=7, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0,
                      framealpha=0.95)
            ax.grid(alpha=0.3)
            focus, = ax.plot([], [], "o", ms=8, mfc="none", mec="black",
                             mew=1.5, visible=False, zorder=9)
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
                bbox={"boxstyle": "square,pad=0.35", "fc": "white",
                      "ec": BORDER, "alpha": 0.96}, fontsize=8,
                visible=False, zorder=10)
            for w in self.chart_opt.winfo_children():
                w.destroy()
            canvas = self.embed_chart(fig, self.chart_opt, toolbar=True)
            xy = np.column_stack((frontier["volatility"], frontier["return"]))

            def hover(event):
                if event.inaxes is not ax:
                    focus.set_visible(False)
                    annotation.set_visible(False)
                    canvas.draw_idle()
                    return
                display = ax.transData.transform(xy)
                distance = np.hypot(display[:, 0] - event.x,
                                    display[:, 1] - event.y)
                idx = int(np.argmin(distance))
                if distance[idx] > 14:
                    focus.set_visible(False)
                    annotation.set_visible(False)
                    canvas.draw_idle()
                    return
                row = frontier.iloc[idx]
                weights = pd.Series({
                    col[2:]: row[col] for col in frontier.columns
                    if col.startswith("w_")}).sort_values(ascending=False)
                top = ", ".join(f"{name} {weight:.0%}"
                                for name, weight in weights.head(5).items())
                x, y = float(row["volatility"]), float(row["return"])
                focus.set_data([x], [y])
                focus.set_visible(True)
                annotation.xy = (x, y)
                annotation.set_text(
                    f"Return {y:.2%} | Volatility {x:.2%}\n{top}")
                annotation.set_visible(True)
                self.var_opt_hover.set(
                    f"Selected frontier portfolio: return {y:.2%}, "
                    f"volatility {x:.2%} | {top}")
                canvas.draw_idle()

            canvas.mpl_connect("motion_notify_event", hover)

        self.run_async(work, done, "Optimizing...", contextual=True,
                       cancellable=True)

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
        self.btn_projection_run = ttk.Button(
            row, text="Run simulation", command=self._projection_run)
        self.btn_projection_run.pack(side="left", padx=8)
        self.btn_projection_cancel = ttk.Button(
            row, text="Cancel", command=lambda: self.cancel_task("compute"),
            state="disabled")
        self.btn_projection_cancel.pack(side="left")

        progress_row = ttk.Frame(self.content)
        progress_row.pack(fill="x", pady=(7, 4))
        self.var_projection_progress = tk.DoubleVar(value=0.0)
        ttk.Progressbar(progress_row, variable=self.var_projection_progress,
                        maximum=100).pack(side="left", fill="x", expand=True)
        self.var_projection_status = tk.StringVar(value="Ready")
        ttk.Label(progress_row, textvariable=self.var_projection_status,
                  width=34, anchor="e").pack(side="right", padx=(10, 0))

        body = ttk.Panedwindow(self.content, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(4, 0))
        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=(12, 0, 0, 0))
        body.add(left, weight=4)
        body.add(right, weight=2)
        self.chart_mc = ttk.Frame(left)
        self.chart_mc.pack(fill="both", expand=True)

        ttk.Label(right, text="Projection snapshot",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self._mc_stat_vars = {}
        stats_grid = ttk.Frame(right)
        stats_grid.pack(fill="x", pady=(6, 8))
        stat_labels = [
            ("P5 (pessimistic)", "p5"), ("Median", "median"),
            ("P95 (optimistic)", "p95"), ("Probability of loss", "prob_loss"),
            ("Horizon VaR 95%", "var"), ("Horizon CVaR 95%", "cvar"),
            ("Minimum observed*", "minimum"), ("Maximum observed*", "maximum"),
        ]
        for i, (label, key) in enumerate(stat_labels):
            ttk.Label(stats_grid, text=label).grid(row=i, column=0, sticky="w", pady=2)
            variable = tk.StringVar(value="-")
            self._mc_stat_vars[key] = variable
            ttk.Label(stats_grid, textvariable=variable,
                      font=("Segoe UI", 10, "bold")).grid(
                          row=i, column=1, sticky="e", padx=(12, 0), pady=2)
        stats_grid.columnconfigure(1, weight=1)
        ttk.Label(right, text="*Raw extremes depend on the simulation sample.",
                  foreground=MUTED, wraplength=310, justify="left").pack(anchor="w")
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        self.var_mc_hover = tk.StringVar(
            value="Move the pointer over a trajectory to inspect it.")
        ttk.Label(right, textvariable=self.var_mc_hover, foreground=MUTED,
                  wraplength=310, justify="left").pack(anchor="w", pady=(0, 6))
        self.tree_mc = make_tree(right, height=10)

    def _projection_run(self):
        try:
            years = float(self.e_horizon.get().replace(",", "."))
        except ValueError:
            years = 3.0

        if years <= 0 or years > 50:
            messagebox.showwarning("LFT", "Horizon must be between 0 and 50 years.")
            return
        started = time.perf_counter()

        def work(progress, cancel_event):
            progress(0.01, "Preparing portfolio inputs")
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            signature = self._calculation_signature(
                prices, "projection", years)
            cached = self._result_cache.get("projection")
            if cached is not None and cached[0] == signature:
                progress(1.0, "Using cached projection")
                return cached[1]
            w = weights.reindex(asset_returns.columns).fillna(0.0)
            horizon = int(years * 252)
            rf = s["risk_free"]; fee = s.get("fee_annual", 0.005); infl = s.get("inflation", 0.025)
            gbm = mc.simulate_gbm(asset_returns, w, value, horizon, n_sims=20000,
                                  risk_free=rf, cost_annual=fee, inflation=infl,
                                  progress_callback=lambda f, m: progress(
                                      0.03 + 0.55 * f, m),
                                  cancel_event=cancel_event)
            boot = mc.simulate_bootstrap(asset_returns, w, value, horizon, n_sims=20000,
                                         risk_free=rf, cost_annual=fee, inflation=infl,
                                         progress_callback=lambda f, m: progress(
                                             0.58 + 0.36 * f, m),
                                         cancel_event=cancel_event)
            progress(0.96, "Computing percentiles and representative paths")
            summary = pd.concat([gbm.summary(), boot.summary()], axis=1)
            summary.columns = ["GBM", "Bootstrap"]
            days = np.arange(gbm.paths.shape[0])
            bands = {p: np.percentile(gbm.paths, p, axis=1) for p in (5, 50, 95)}
            order = np.argsort(gbm.paths[-1])
            positions = np.linspace(0, len(order) - 1,
                                    min(160, len(order)), dtype=int)
            chosen = order[positions]
            representative = gbm.paths[:, chosen]
            path_percentiles = 100 * positions / max(len(order) - 1, 1)
            detail = {
                "p5": gbm.percentiles[5], "median": gbm.percentiles[50],
                "p95": gbm.percentiles[95], "prob_loss": gbm.prob_loss(),
                "var": gbm.var(), "cvar": gbm.cvar(),
                "minimum": float(gbm.terminal_values.min()),
                "maximum": float(gbm.terminal_values.max()),
            }
            progress(1.0, "Projection ready")
            result = (summary, days, bands, value, representative,
                      path_percentiles, detail)
            self._result_cache["projection"] = (signature, result)
            return result

        def done(payload):
            if not self.chart_mc.winfo_exists():
                return
            summary, days, bands, value, paths, path_percentiles, detail = payload
            df_to_tree(self.tree_mc, summary, "measure")
            for key in ("p5", "median", "p95", "minimum", "maximum"):
                self._mc_stat_vars[key].set(f"{detail[key]:,.0f}")
            for key in ("prob_loss", "var", "cvar"):
                self._mc_stat_vars[key].set(f"{detail[key]:.1%}")

            fig = self.new_figure((8.0, 4.7))
            ax = fig.add_subplot(111)
            segments = np.stack([
                np.column_stack((days, paths[:, i])) for i in range(paths.shape[1])])
            collection = LineCollection(segments, colors="#71879f",
                                        linewidths=0.55, alpha=0.20, zorder=1)
            ax.add_collection(collection)
            ax.fill_between(days, bands[5], bands[95], alpha=0.12,
                            color=ACCENT, label="P5-P95 range", zorder=2)
            ax.plot(days, bands[50], color=ACCENT, lw=2.2, label="Median", zorder=4)
            ax.plot(days, bands[5], color=ACCENT, lw=1, ls=":", zorder=3)
            ax.plot(days, bands[95], color=ACCENT, lw=1, ls=":", zorder=3)
            ax.axhline(value, color="black", lw=1, ls="--")
            ax.set_xlim(days[0], days[-1])
            low = min(float(paths.min()), float(bands[5].min()), value)
            high = max(float(paths.max()), float(bands[95].max()), value)
            pad = max((high - low) * 0.04, 1.0)
            ax.set_ylim(low - pad, high + pad)
            ax.set_xlabel("Trading days")
            ax.set_ylabel("Portfolio value")
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.grid(alpha=0.3)
            # Legend below the plot: inside, it sat on top of the fan of paths.
            ax.legend(fontsize=8, loc="upper center",
                      bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
            highlight, = ax.plot([], [], color="#d43f3a", lw=2.1,
                                 zorder=8, visible=False)
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
                bbox={"boxstyle": "square,pad=0.35", "fc": "white",
                      "ec": BORDER, "alpha": 0.96},
                arrowprops={"arrowstyle": "-", "color": BORDER},
                fontsize=8, visible=False, zorder=10)
            for w in self.chart_mc.winfo_children():
                w.destroy()
            canvas = self.embed_chart(fig, self.chart_mc, toolbar=True)

            def hover(event):
                if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                    highlight.set_visible(False)
                    annotation.set_visible(False)
                    canvas.draw_idle()
                    return
                day = int(np.clip(round(event.xdata), 0, len(days) - 1))
                values = paths[day]
                selected = int(np.argmin(np.abs(values - event.ydata)))
                if abs(values[selected] - event.ydata) > 0.055 * (high - low):
                    highlight.set_visible(False)
                    annotation.set_visible(False)
                    canvas.draw_idle()
                    return
                path = paths[:, selected]
                terminal = float(path[-1])
                cagr = (terminal / value) ** (252 / max(days[-1], 1)) - 1
                percentile = path_percentiles[selected]
                highlight.set_data(days, path)
                highlight.set_visible(True)
                annotation.xy = (day, path[day])
                annotation.set_text(
                    f"Path percentile: {percentile:.0f}%\n"
                    f"Day {day}: {path[day]:,.0f}\n"
                    f"Terminal: {terminal:,.0f} | CAGR {cagr:.1%}")
                annotation.set_visible(True)
                self.var_mc_hover.set(
                    f"Selected trajectory: percentile {percentile:.0f}% | "
                    f"terminal value {terminal:,.0f} | CAGR {cagr:.1%}")
                canvas.draw_idle()

            canvas.mpl_connect("motion_notify_event", hover)
            self.var_projection_progress.set(100)
            self.var_projection_status.set(
                f"Completed in {time.perf_counter() - started:.1f}s")
            self.btn_projection_run.configure(state="normal")
            self.btn_projection_cancel.configure(state="disabled")

        def on_progress(fraction, message, eta):
            if (hasattr(self, "btn_projection_run") and
                    self.btn_projection_run.winfo_exists()):
                self.var_projection_progress.set(fraction * 100)
                eta_text = f" | ~{eta:.0f}s left" if eta is not None else ""
                self.var_projection_status.set(f"{message}{eta_text}")

        def on_cancel():
            if hasattr(self, "btn_projection_run") and self.btn_projection_run.winfo_exists():
                self.btn_projection_run.configure(state="normal")
                self.btn_projection_cancel.configure(state="disabled")
                self.var_projection_status.set("Cancelled")

        def on_error(_error):
            if hasattr(self, "btn_projection_run") and self.btn_projection_run.winfo_exists():
                self.btn_projection_run.configure(state="normal")
                self.btn_projection_cancel.configure(state="disabled")
                self.var_projection_status.set("Failed")

        self.btn_projection_run.configure(state="disabled")
        self.btn_projection_cancel.configure(state="normal")
        token = self.run_async(
            work, done, "Simulating 40,000 realistic paths...",
            contextual=True, cancellable=True, on_progress=on_progress,
            on_cancel=on_cancel, on_error=on_error)
        if token is None:
            on_cancel()

    # ----------------------------------------------------------- 7 backtest

    def page_backtest(self):
        if self.need_positions():
            return
        ttk.Label(self.content,
                  text="Honest walk-forward comparison on your tickers "
                       "(can take 30-60 seconds).").pack(anchor="w")
        controls = ttk.Frame(self.content)
        controls.pack(fill="x", pady=4)
        self.btn_backtest_run = ttk.Button(
            controls, text="Run backtest", command=self._backtest_run)
        self.btn_backtest_run.pack(side="left")
        self.btn_backtest_cancel = ttk.Button(
            controls, text="Cancel", command=lambda: self.cancel_task("compute"),
            state="disabled")
        self.btn_backtest_cancel.pack(side="left", padx=8)
        self.var_backtest_progress = tk.DoubleVar(value=0.0)
        ttk.Progressbar(controls, variable=self.var_backtest_progress,
                        maximum=100).pack(side="left", fill="x", expand=True, padx=8)
        self.var_backtest_status = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.var_backtest_status,
                  width=28, anchor="e").pack(side="right")
        self.tree_bt = make_tree(self.content, height=6)
        self.var_bt_hover = tk.StringVar(
            value="Move the pointer over a strategy curve to inspect a date.")
        ttk.Label(self.content, textvariable=self.var_bt_hover,
                  foreground=MUTED).pack(anchor="w", pady=(2, 0))
        self.chart_bt = ttk.Frame(self.content)
        self.chart_bt.pack(fill="both", expand=True)

    def _backtest_run(self):
        started = time.perf_counter()

        def work(progress, cancel_event):
            progress(0.01, "Preparing walk-forward inputs")
            s, prices, asset_returns, bench, last, snap, weights, value = \
                self.backend.portfolio_context()
            signature = self._calculation_signature(prices, "backtest")
            cached = self._result_cache.get("backtest")
            if cached is not None and cached[0] == signature:
                progress(1.0, "Using cached backtest")
                return cached[1]
            held = list(asset_returns.columns)
            cap = max(s["max_weight"], 1 / len(held) + 0.01)
            strategies = {
                "MaxSharpe WF": wf.make_max_sharpe(s["risk_free"], cap),
                "MinVol WF": wf.make_min_vol(cap),
                "EqualWeight 1/N": wf.make_equal_weight(),
            }
            curves, _ = wf.compare_walk_forward(prices[held], strategies,
                                                lookback=s["lookback"],
                                                tc_bps=s["tc_bps"],
                                                progress_callback=lambda f, m: progress(
                                                    0.03 + 0.90 * f, m),
                                                cancel_event=cancel_event)
            progress(0.95, "Computing benchmark statistics")
            curves[s["benchmark"]] = (10_000 * (1 + bench.reindex(curves.index))
                                      .cumprod())
            stats = metrics.summary(curves.pct_change().dropna(),
                                    bench.reindex(curves.index), s["risk_free"])
            rows = ["CAGR", "Annualized volatility", "Sharpe", "Max drawdown", "Calmar"]
            progress(1.0, "Backtest ready")
            result = (stats.loc[rows].round(4), curves)
            self._result_cache["backtest"] = (signature, result)
            return result

        def done(payload):
            if not self.chart_bt.winfo_exists():
                return
            stats, curves = payload
            df_to_tree(self.tree_bt, stats, "metric")
            fig = self.new_figure((8.6, 4.2))
            ax = fig.add_subplot(111)
            lines = {}
            for col in curves.columns:
                line, = ax.plot(curves.index, curves[col], lw=1.4, label=col)
                lines[col] = line
            ax.set_yscale("log")
            ax.set_ylabel("Value (log scale)")
            # Legend outside the axes (right): inside, it overlapped the
            # early part of the equity curves on the log scale.
            ax.legend(fontsize=7, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0,
                      framealpha=0.95)
            ax.grid(alpha=0.3)
            annotation = ax.annotate(
                "", xy=(0, 0), xytext=(10, 12), textcoords="offset points",
                bbox={"boxstyle": "square,pad=0.35", "fc": "white",
                      "ec": BORDER, "alpha": 0.96}, fontsize=8,
                visible=False, zorder=10)
            for w in self.chart_bt.winfo_children():
                w.destroy()
            canvas = self.embed_chart(fig, self.chart_bt, toolbar=True)
            dates_num = mdates.date2num(curves.index.to_pydatetime())

            def hover(event):
                if (event.inaxes is not ax or event.xdata is None or
                        event.ydata is None or event.ydata <= 0):
                    annotation.set_visible(False)
                    for line in lines.values():
                        line.set_linewidth(1.4)
                        line.set_alpha(1.0)
                    canvas.draw_idle()
                    return
                idx = int(np.clip(np.searchsorted(dates_num, event.xdata),
                                  0, len(dates_num) - 1))
                if idx > 0 and abs(dates_num[idx - 1] - event.xdata) < abs(
                        dates_num[idx] - event.xdata):
                    idx -= 1
                values = curves.iloc[idx].dropna()
                if values.empty:
                    return
                distances = np.abs(np.log(values.values) - np.log(event.ydata))
                selected = values.index[int(np.argmin(distances))]
                value_at_date = float(values[selected])
                if float(np.min(distances)) > 0.16:
                    annotation.set_visible(False)
                    canvas.draw_idle()
                    return
                for name, line in lines.items():
                    line.set_linewidth(2.6 if name == selected else 1.0)
                    line.set_alpha(1.0 if name == selected else 0.32)
                first = float(curves[selected].dropna().iloc[0])
                total_return = value_at_date / first - 1
                date = curves.index[idx]
                annotation.xy = (date, value_at_date)
                annotation.set_text(
                    f"{selected}\n{date:%Y-%m-%d}\n"
                    f"Value: {value_at_date:,.0f} | Return: {total_return:.1%}")
                annotation.set_visible(True)
                self.var_bt_hover.set(
                    f"{selected} | {date:%Y-%m-%d} | value {value_at_date:,.0f} | "
                    f"return since inception {total_return:.1%}")
                canvas.draw_idle()

            canvas.mpl_connect("motion_notify_event", hover)
            self.var_backtest_progress.set(100)
            self.var_backtest_status.set(
                f"Completed in {time.perf_counter() - started:.1f}s")
            self.btn_backtest_run.configure(state="normal")
            self.btn_backtest_cancel.configure(state="disabled")

        def on_progress(fraction, message, eta):
            if hasattr(self, "btn_backtest_run") and self.btn_backtest_run.winfo_exists():
                self.var_backtest_progress.set(fraction * 100)
                eta_text = f" | ~{eta:.0f}s left" if eta is not None else ""
                self.var_backtest_status.set(f"{message}{eta_text}")

        def on_cancel():
            if hasattr(self, "btn_backtest_run") and self.btn_backtest_run.winfo_exists():
                self.btn_backtest_run.configure(state="normal")
                self.btn_backtest_cancel.configure(state="disabled")
                self.var_backtest_status.set("Cancelled")

        def on_error(_error):
            if hasattr(self, "btn_backtest_run") and self.btn_backtest_run.winfo_exists():
                self.btn_backtest_run.configure(state="normal")
                self.btn_backtest_cancel.configure(state="disabled")
                self.var_backtest_status.set("Failed")

        self.btn_backtest_run.configure(state="disabled")
        self.btn_backtest_cancel.configure(state="normal")
        token = self.run_async(
            work, done, "Running walk-forward backtest...", contextual=True,
            cancellable=True, on_progress=on_progress, on_cancel=on_cancel,
            on_error=on_error)
        if token is None:
            on_cancel()

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
            start, end = self.backend.analysis_dates()
            prices, source = self.backend.store.get_prices(
                tickers, start, end)
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
                       f"Connecting to IBKR on {endpoint} (read-only)...",
                       channel="ibkr")

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
            "start / end: analysis period (YYYY-MM-DD; end may be 'today')\n"
            "risk_free: annual risk-free rate (0.03 = 3%)\n"
            "tc_bps: transaction costs in basis points\n"
            "lookback: walk-forward estimation window (trading days)\n"
            "max_weight: per-asset cap for optimized strategies\n"
            "ibkr_port: 7497 paper account, 7496 live account")).pack(anchor="w", pady=8)

    def _settings_save(self):
        s = self.backend.state["settings"]
        casts = {"benchmark": str, "start": str, "end": str,
                 "risk_free": float, "tc_bps": float,
                 "fee_annual": float, "inflation": float, "lookback": int,
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
    enable_windows_dpi_awareness()
    Gui().mainloop()
