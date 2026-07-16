"""
dashboard/app.py
CueBot live dashboard — mobile-responsive, dark trading terminal.
Reads positions and trade history directly from MT5 (no local log file).

Run on the Windows machine that has the MT5 terminal:
    python dashboard/app.py

Then access on iPhone / other devices on the same network:
    http://<windows-ip>:8050
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback, Output, Input
import dash_bootstrap_components as dbc

from broker.mt5_client import initialize, resolve_symbol, MT5_AVAILABLE
from broker.executor import MAGIC

if MT5_AVAILABLE:
    import MetaTrader5 as mt5

# ── Config ──────────────────────────────────────────────────────────────
SYMBOLS    = ["US30", "EURUSD"]
REFRESH_MS = 30_000   # auto-refresh every 30 seconds

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1"}
    ],
    title="CueBot",
)
server = app.server


# ════════════════════════════════════════════════════════════════════════
# DATA HELPERS — everything comes from MT5
# ════════════════════════════════════════════════════════════════════════

def get_account_state() -> dict:
    initialize()
    a = mt5.account_info()
    return {
        "equity":   float(a.equity),
        "balance":  float(a.balance),
        "currency": a.currency,
        "mode":     "DEMO" if a.trade_mode == 0 else "LIVE",
        "login":    a.login,
    }


def get_live_prices(symbols: list) -> dict:
    initialize()
    out = {}
    for s in symbols:
        try:
            sym = resolve_symbol(s)
            tick = mt5.symbol_info_tick(sym)
            out[s] = (tick.bid + tick.ask) / 2 if tick else None
        except Exception:
            out[s] = None
    return out


def get_positions() -> list:
    """Open positions opened by this bot (filtered by our magic number)."""
    initialize()
    positions = mt5.positions_get() or []
    mine = [p for p in positions if p.magic == MAGIC]
    out = []
    for p in mine:
        out.append({
            "symbol":    p.symbol,
            "direction": 1 if p.type == mt5.POSITION_TYPE_BUY else -1,
            "volume":    p.volume,
            "entry":     p.price_open,
            "sl":        p.sl,
            "tp":        p.tp,
            "profit":    p.profit,
            "opened":    datetime.fromtimestamp(p.time),
        })
    return out


def get_closed_deals(days: int = 30) -> list:
    """Closed trade history from MT5, filtered by our magic number."""
    initialize()
    frm = datetime.now() - timedelta(days=days)
    to  = datetime.now()
    deals = mt5.history_deals_get(frm, to) or []
    mine = [d for d in deals if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT]
    out = []
    for d in mine:
        out.append({
            "symbol":    d.symbol,
            "profit":    d.profit,
            "price":     d.price,
            "volume":    d.volume,
            "time":      datetime.fromtimestamp(d.time),
            "comment":   d.comment,
        })
    return sorted(out, key=lambda x: x["time"], reverse=True)


def build_equity_curve(start_balance: float, closed: list) -> pd.DataFrame:
    """Reconstruct running equity from closed deals (oldest → newest)."""
    rows = [{"time": "Start", "equity": start_balance}]
    running = start_balance
    for d in reversed(closed):
        running += d["profit"]
        rows.append({"time": d["time"].strftime("%m-%d %H:%M"), "equity": running})
    return pd.DataFrame(rows)


def calc_stats(closed: list) -> dict:
    if not closed:
        return {"trades": 0, "win_rate": 0, "pnl": 0, "avg_win": 0, "avg_loss": 0}
    pnls    = [d["profit"] for d in closed]
    winners = [p for p in pnls if p > 0]
    losers  = [p for p in pnls if p < 0]
    return {
        "trades":   len(closed),
        "win_rate": round(len(winners) / len(closed) * 100, 1),
        "pnl":      round(sum(pnls), 2),
        "avg_win":  round(sum(winners) / len(winners), 2) if winners else 0,
        "avg_loss": round(sum(losers)  / len(losers),  2) if losers  else 0,
    }


# ════════════════════════════════════════════════════════════════════════
# COMPONENT BUILDERS
# ════════════════════════════════════════════════════════════════════════

def _dot(label: str, active: bool, title: str = "") -> html.Span:
    color = "#00e676" if active else "#444"
    return html.Span(
        label, title=title,
        style={
            "background": color,
            "color":      "#000" if active else "#888",
            "borderRadius": "4px",
            "padding":    "2px 6px",
            "fontSize":   "0.7rem",
            "fontWeight": "bold",
        }
    )


def signal_card(symbol: str, position: dict | None, live_price: float | None) -> dbc.Card:
    if position:
        dir_label = "LONG" if position["direction"] == 1 else "SHORT"
        color     = "#00e676" if position["direction"] == 1 else "#ff5252"
        pnl       = position["profit"]
        pnl_col   = "success" if pnl >= 0 else "danger"
        pnl_badge = dbc.Badge(f"{pnl:+.2f}", color=pnl_col, className="ms-2 fs-6")
    else:
        dir_label = "—"
        color     = "#888"
        pnl_badge = html.Span()

    price_str = f"{live_price:.5f}" if live_price else "—"

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H4(symbol, className="d-inline me-3 text-white"),
                html.Span(dir_label,
                          style={"color": color, "fontWeight": "bold", "fontSize": "1.3rem"}),
                pnl_badge,
            ], className="mb-2"),
            html.Div([
                html.Small("Live price: ", className="text-muted"),
                html.Span(price_str, className="text-white"),
            ]),
            html.Small(f"SL {position['sl']:.5f} / TP {position['tp']:.5f}"
                       if position else "No open position",
                       className="text-muted"),
        ])
    ], className="mb-3 h-100",
       style={"background": "#1e1e2e", "border": f"1px solid {color}33"})


def positions_table(positions: list) -> dbc.Table:
    if not positions:
        return html.P("No open positions", className="text-muted text-center py-3")
    rows = []
    for p in positions:
        dir_col   = "#00e676" if p["direction"] == 1 else "#ff5252"
        dir_label = "LONG" if p["direction"] == 1 else "SHORT"
        pnl_col   = "#00e676" if p["profit"] >= 0 else "#ff5252"
        rows.append(html.Tr([
            html.Td(p["symbol"],       style={"color": "#fff", "fontWeight": "bold"}),
            html.Td(dir_label,         style={"color": dir_col}),
            html.Td(f"{p['volume']}",  className="text-muted"),
            html.Td(f"{p['entry']:.5f}", className="text-muted"),
            html.Td(f"{p['profit']:+.2f}", style={"color": pnl_col, "fontWeight": "bold"}),
            html.Td(f"{p['sl']:.5f}",  className="text-danger"),
            html.Td(f"{p['tp']:.5f}",  className="text-success"),
        ]))
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th(x) for x in ["Symbol", "Side", "Vol", "Entry", "P&L", "SL", "TP"]
        ]), style={"fontSize": "0.75rem", "color": "#888"}),
        html.Tbody(rows),
    ], bordered=False, dark=True, hover=True, responsive=True, style={"fontSize": "0.85rem"})


def trades_table(closed: list) -> dbc.Table:
    if not closed:
        return html.P("No completed trades yet", className="text-muted text-center py-3")
    rows = []
    for d in closed[:15]:
        color = "#00e676" if d["profit"] >= 0 else "#ff5252"
        rows.append(html.Tr([
            html.Td(d["symbol"],               style={"color": "#fff"}),
            html.Td(f"{d['price']:.5f}",       className="text-muted"),
            html.Td(f"{d['profit']:+.2f}",
                    style={"color": color, "fontWeight": "bold"}),
            html.Td(d["time"].strftime("%m-%d %H:%M"),
                    className="text-muted", style={"fontSize": "0.72rem"}),
        ]))
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th(x) for x in ["Symbol", "Exit", "P&L", "Time"]
        ]), style={"fontSize": "0.75rem", "color": "#888"}),
        html.Tbody(rows),
    ], bordered=False, dark=True, hover=True, responsive=True, style={"fontSize": "0.85rem"})


def equity_chart(df: pd.DataFrame) -> go.Figure:
    pnl_total = df["equity"].iloc[-1] - df["equity"].iloc[0] if len(df) > 1 else 0
    line_color = "#00e676" if pnl_total >= 0 else "#ff5252"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["equity"],
        mode="lines",
        line=dict(color=line_color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'0,230,118' if pnl_total >= 0 else '255,82,82'},0.08)",
        hovertemplate="%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(gridcolor="#1e1e2e", tickformat=",.0f",
                   tickfont=dict(color="#888", size=10), zeroline=False),
        height=200, showlegend=False,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════

app.layout = dbc.Container([
    dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),

    dbc.Row(dbc.Col(html.Div([
        html.Span("CueBot", style={"fontWeight": "800", "fontSize": "1.4rem", "color": "#00e676"}),
        html.Span(" SMC · MT5", className="text-muted", style={"fontSize": "0.9rem"}),
    ], className="py-3"))),

    dbc.Row(id="kpi-row", className="mb-3 g-2"),
    dbc.Row(id="signal-cards", className="mb-3 g-2"),

    dbc.Card([
        dbc.CardHeader(html.Small("Equity Curve", className="text-muted")),
        dbc.CardBody(dcc.Graph(id="equity-chart", config={"displayModeBar": False}),
                     style={"padding": "8px"}),
    ], style={"background": "#1e1e2e"}, className="mb-3"),

    dbc.Card([
        dbc.CardHeader(html.Small("Open Positions", className="text-muted")),
        dbc.CardBody(html.Div(id="positions-table"), style={"padding": "8px"}),
    ], style={"background": "#1e1e2e"}, className="mb-3"),

    dbc.Card([
        dbc.CardHeader(html.Small("Recent Trades", className="text-muted")),
        dbc.CardBody(html.Div(id="trades-table"), style={"padding": "8px"}),
    ], style={"background": "#1e1e2e"}, className="mb-3"),

    html.P(id="last-updated", className="text-muted text-center pb-3",
           style={"fontSize": "0.7rem"}),

], fluid=True, style={"background": "#0d0d1a", "minHeight": "100vh", "color": "#e0e0e0"})


# ════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════════════

@callback(
    Output("kpi-row",         "children"),
    Output("signal-cards",    "children"),
    Output("equity-chart",    "figure"),
    Output("positions-table", "children"),
    Output("trades-table",    "children"),
    Output("last-updated",    "children"),
    Input("refresh",          "n_intervals"),
)
def refresh_dashboard(n):
    try:
        account     = get_account_state()
        positions   = get_positions()
        closed      = get_closed_deals(days=30)
        live_prices = get_live_prices(SYMBOLS)
    except Exception as e:
        err = html.Div([html.Strong("MT5 connection error: "), str(e)],
                       style={"color": "#ff5252", "padding": "20px"})
        return err, "", go.Figure(), err, err, f"Error at {datetime.now():%H:%M:%S}"

    stats     = calc_stats(closed)
    positions_by_symbol = {p["symbol"]: p for p in positions}

    # Reconstruct starting balance (balance - total realized pnl in window)
    start_balance = account["balance"] - stats["pnl"]
    equity_df     = build_equity_curve(start_balance, closed)

    # ── KPIs ──
    pnl_color = "#00e676" if stats["pnl"] >= 0 else "#ff5252"
    cur       = account["currency"]
    kpi_data = [
        ("Equity",    f"{cur} {account['equity']:,.2f}", "#fff"),
        ("P&L (30d)", f"{stats['pnl']:+,.2f}",           pnl_color),
        ("Trades",    str(stats["trades"]),              "#fff"),
        ("Win Rate",  f"{stats['win_rate']:.0f}%",
                      "#00e676" if stats["win_rate"] >= 50 else "#ff5252"),
    ]
    kpi_row = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Small(label, className="text-muted d-block", style={"fontSize": "0.7rem"}),
            html.Span(value, style={"color": color, "fontWeight": "bold", "fontSize": "1.1rem"}),
        ]), style={"background": "#1e1e2e", "textAlign": "center"}), xs=6, md=3)
        for label, value, color in kpi_data
    ]

    sig_cards = [
        dbc.Col(
            signal_card(sym,
                        next((p for p in positions
                              if resolve_symbol(sym) == p["symbol"]), None),
                        live_prices.get(sym)),
            xs=12, md=6
        )
        for sym in SYMBOLS
    ]

    updated = (f"MT5 {account['mode']} · account {account['login']}  ·  "
               f"Refreshed {datetime.now():%H:%M:%S}")
    return kpi_row, sig_cards, equity_chart(equity_df), \
           positions_table(positions), trades_table(closed), updated


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "="*55)
    print("  CueBot Dashboard (MT5)")
    print("="*55)
    print(f"  Local:    http://127.0.0.1:8050")
    print(f"  Network:  http://{local_ip}:8050")
    print("="*55 + "\n")

    app.run(host="0.0.0.0", port=8050, debug=False)
