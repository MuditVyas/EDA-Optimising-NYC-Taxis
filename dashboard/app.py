"""
NYC Yellow Taxi 2023 — Operations Dashboard

Reads the cleaned trip table and taxi-zone geometry exported by
notebooks/EDA_NYC_Taxi_Operations_Mudit_Vyas.ipynb and serves a single-page
dashboard with three sections: temporal demand, revenue & pricing, geospatial.

    python dashboard/app.py        ->  http://127.0.0.1:8050
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRIPS_PATH = PROCESSED_DIR / "trips_clean.parquet"
ZONES_PATH = PROCESSED_DIR / "taxi_zones.geojson"

# Trips were sampled at 1% of every hour of every day, so counts are multiplied
# by 100 to estimate full-population volumes.
SAMPLE_FRACTION = 0.01
SCALE = int(1 / SAMPLE_FRACTION)

# --------------------------------------------------------------------------- theme
BG = "#0f1419"
PANEL = "#1a2129"
GRID = "#2a3441"
TEXT = "#e6edf3"
MUTED = "#8b949e"
TAXI = "#f5c518"
ACCENT = "#4da3ff"
UP = "#3fb950"
DOWN = "#f85149"
SEQ = ["#0f2b46", "#14496e", "#1b6d94", "#3f9bb5", "#84c3c9", "#d5e8c8", "#f5c518"]

FONT = "Inter, -apple-system, Segoe UI, sans-serif"

# Axis caps for the "Fare against" chart. A few 25-hour meter errors and $500
# fares would otherwise flatten the dense region into an unreadable sliver.
DUR_X_CAP = 100.0   # minutes
DUR_Y_CAP = 150.0   # dollars
PAX_Y_CAP = 100.0   # dollars

# Plotly tints hover text with the trace colour by default, which is unreadable
# against the dark tooltip. Forcing white is applied to every figure.
HOVER = dict(
    bgcolor="#0b0f14",
    bordercolor=GRID,
    font=dict(color="#ffffff", size=12, family=FONT),
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
TOD = ["Morning", "Afternoon", "Evening", "Night"]
PAYMENTS = {1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute"}
VENDORS = {1: "Vendor 1 (Creative Mobile)", 2: "Vendor 2 (VeriFone)"}


def load_data() -> tuple[pd.DataFrame, dict]:
    if not TRIPS_PATH.exists() or not ZONES_PATH.exists():
        raise SystemExit(
            f"Could not find {TRIPS_PATH.name} and {ZONES_PATH.name} in {PROCESSED_DIR}.\n"
            "Run the notebook first — its final section exports both files."
        )

    df = pd.read_parquet(TRIPS_PATH)
    df["month"] = df["tpep_pickup_datetime"].dt.month
    df["day_name"] = df["tpep_pickup_datetime"].dt.day_name()
    df["quarter"] = df["tpep_pickup_datetime"].dt.quarter
    df["duration_min"] = df["trip_duration"] / 60.0
    df["payment_label"] = df["payment_type"].map(PAYMENTS).fillna("Other")

    with open(ZONES_PATH) as fh:
        geo = json.load(fh)
    return df, geo


TRIPS, ZONES_GEO = load_data()
BOROUGHS = sorted(b for b in TRIPS["borough"].dropna().unique() if b != "Unknown")


# --------------------------------------------------------------------------- helpers
def style_fig(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, autosize=True,
        font=dict(color=TEXT, family=FONT, size=12),
        margin=dict(l=62, r=26, t=44, b=48), height=height,
        title_font=dict(size=12, color=MUTED),
        hoverlabel=HOVER,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font_size=11)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font_size=11)
    if legend:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", title_text=""))
    else:
        fig.update_layout(showlegend=False)
    return fig


def empty_fig(msg: str = "No trips match the selected filters", height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_fig(fig, height=height, legend=False)


# Several callbacks need the same filtered frame; caching it keeps a filter
# change to one pass over the data instead of one pass per section.
@lru_cache(maxsize=64)
def _filtered(months: tuple, boroughs: tuple, vendors: tuple, tod: tuple) -> pd.DataFrame:
    df = TRIPS
    if months:
        df = df[df["month"].between(months[0], months[1])]
    if boroughs:
        df = df[df["borough"].isin(boroughs)]
    if vendors:
        df = df[df["VendorID"].isin(vendors)]
    if tod:
        df = df[df["time_of_day"].isin(tod)]
    return df


def apply_filters(months, boroughs, vendors, tod) -> pd.DataFrame:
    """Returns a cached, shared frame — callers must not mutate it in place."""
    return _filtered(
        tuple(months) if months else (1, 12),
        tuple(sorted(boroughs)) if boroughs else (),
        tuple(sorted(vendors)) if vendors else (),
        tuple(sorted(tod)) if tod else (),
    )


def card(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(className="kpi", children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(note, className="kpi-note"),
    ])


def panel(title: str, graph_id: str, flex: str = "1 1 100%", height: int = 340,
          head_extra=None) -> html.Div:
    """`flex` is the complete CSS flex shorthand: '<grow> <shrink> <basis>'."""
    head = (html.Div(className="panel-head", children=[
                html.Div(title, className="panel-title"), head_extra])
            if head_extra is not None
            else html.Div(title, className="panel-title"))
    return html.Div(className="panel", style={"flex": flex, "minWidth": "0"}, children=[
        head,
        dcc.Graph(id=graph_id, config={"displayModeBar": False, "responsive": True},
                  style={"height": f"{height}px", "width": "100%"}),
    ])


def zone_table(table_id: str, title: str, value_header: str) -> html.Div:
    return html.Div(className="panel", style={"flex": "1 1 300px", "minWidth": "0"}, children=[
        html.Div(title, className="panel-title"),
        dash_table.DataTable(
            id=table_id,
            columns=[{"name": "Zone", "id": "zone"}, {"name": value_header, "id": "value"}],
            data=[], style_as_list_view=True, sort_action="native",
            style_header={
                "backgroundColor": PANEL, "color": MUTED, "fontWeight": "600",
                "border": "none", "borderBottom": f"1px solid {GRID}",
                "textTransform": "uppercase", "fontSize": "10px",
                "letterSpacing": "0.8px", "padding": "8px 12px",
            },
            style_cell={
                "backgroundColor": PANEL, "color": TEXT, "border": "none",
                "borderBottom": f"1px solid {GRID}", "fontSize": "12.5px",
                "fontFamily": "Inter, -apple-system, Segoe UI, sans-serif",
                "padding": "8px 12px", "textAlign": "left",
                "whiteSpace": "normal", "height": "auto",
            },
            style_cell_conditional=[{
                "if": {"column_id": "value"}, "textAlign": "right", "width": "38%",
                "color": TAXI, "fontWeight": "600", "fontVariantNumeric": "tabular-nums",
            }],
        ),
    ])


def section(anchor: str, number: str, title: str, subtitle: str, children: list) -> html.Div:
    return html.Div(id=anchor, className="section", children=[
        html.Div(className="section-head", children=[
            html.Span(number, className="section-num"),
            html.Div([html.H2(title), html.P(subtitle)]),
        ]),
        *children,
    ])


def inline_dropdown(dd_id: str, options: list, value: str, width: str = "170px"):
    return dcc.Dropdown(id=dd_id, clearable=False, className="dd dd-inline",
                        options=options, value=value, style={"width": width})


# --------------------------------------------------------------------------- layout
app = Dash(__name__, title="NYC Yellow Taxi 2023 — Operations Dashboard")

frozen = html.Div(className="frozen", children=[
    html.Div(className="controls", children=[
        html.Div([
            html.Label("Months", className="ctrl-label"),
            dcc.RangeSlider(id="f-months", min=1, max=12, step=1, value=[1, 12],
                            marks={i: MONTHS[i - 1][:3] for i in range(1, 13)},
                            tooltip={"placement": "bottom", "always_visible": False}),
        ], style={"flex": "2 1 360px"}),
        html.Div([
            html.Label("Pickup borough", className="ctrl-label"),
            dcc.Dropdown(id="f-borough", options=[{"label": b, "value": b} for b in BOROUGHS],
                         value=[], multi=True, placeholder="All boroughs", className="dd"),
        ], style={"flex": "1 1 200px"}),
        html.Div([
            html.Label("Vendor", className="ctrl-label"),
            dcc.Dropdown(id="f-vendor",
                         options=[{"label": v, "value": k} for k, v in VENDORS.items()],
                         value=[], multi=True, placeholder="Both vendors", className="dd"),
        ], style={"flex": "1 1 200px"}),
        html.Div([
            html.Label("Time of day", className="ctrl-label"),
            dcc.Dropdown(id="f-tod", options=[{"label": t, "value": t} for t in TOD],
                         value=[], multi=True, placeholder="All hours", className="dd"),
        ], style={"flex": "1 1 180px"}),
    ]),
    html.Div(id="kpis", className="kpi-row"),
])

app.layout = html.Div(className="root", children=[
    html.Div(className="header", children=[
        html.Div([
            html.H1("NYC Yellow Taxi — Operations Dashboard"),
            html.P("2023 TLC trip records · 1% hour-stratified sample of ~38M trips · "
                   "Exploratory analysis by Mudit Vyas"),
        ]),
        html.Div("2023", className="year-badge"),
    ]),

    frozen,

    # ---------------------------------------------------------------- 1 temporal
    section("s-temporal", "01", "Temporal Demand",
            "When do New Yorkers hail a cab? Switch the grain to move between "
            "hour of day, day of week and month.", [
        html.Div(className="row", children=[
            panel("Pickups by", "g-temporal", "1 1 100%", 400,
                  head_extra=inline_dropdown("temporal-grain", [
                      {"label": "Hours", "value": "hour"},
                      {"label": "Days", "value": "day"},
                      {"label": "Months", "value": "month"}], "hour", "150px")),
        ]),
        html.Div(className="row", children=[
            panel("Demand heatmap — day of week × hour", "g-heat-demand", "1 1 100%", 330),
        ]),
    ]),

    # ---------------------------------------------------------------- 2 revenue
    section("s-revenue", "02", "Revenue & Pricing",
            "Where the money comes from, how rates move through the week, and how "
            "the two vendors price against each other.", [
        html.Div(className="row", children=[
            panel("Monthly revenue, with change on the previous month",
                  "g-rev-month", "70 1 700px", 400),
            panel("Revenue share by quarter", "g-rev-quarter", "30 1 300px", 400),
        ]),
        html.Div(className="row", children=[
            panel("Average fare per mile — day × hour", "g-heat-fare", "1 1 480px", 360),
            panel("Vendor fare per mile across the day", "g-vendor-hour", "1 1 480px", 360),
        ]),
        html.Div(className="row", children=[
            panel("Fare against", "g-fare-vs", "62 1 620px", 420,
                  head_extra=inline_dropdown("fare-mode", [
                      {"label": "Trip duration", "value": "duration"},
                      {"label": "Passenger count", "value": "passengers"}],
                      "duration", "190px")),
            panel("Payment mix", "g-payment", "38 1 380px", 420),
        ]),
    ]),

    # ---------------------------------------------------------------- 3 geospatial
    section("s-geo", "03", "Geospatial",
            "All 263 TLC taxi zones. The pickup/drop-off ratio is the operational "
            "one — above 1 means a zone sends out more cabs than it receives.", [
        html.Div(className="row", children=[
            html.Div(className="panel", style={"flex": "1 1 100%", "minWidth": "0"}, children=[
                html.Div(className="panel-head", children=[
                    html.Div("Taxi zones — choose a metric", className="panel-title"),
                    dcc.RadioItems(id="geo-metric", className="radio", inline=True, value="trips",
                                   options=[{"label": "Total pickups", "value": "trips"},
                                            {"label": "Pickup / drop-off ratio", "value": "ratio"},
                                            {"label": "Avg passengers", "value": "pax"},
                                            {"label": "Extra charges applied", "value": "extra"}]),
                ]),
                dcc.Graph(id="g-map", config={"displayModeBar": False, "responsive": True},
                          style={"height": "540px", "width": "100%"}),
            ]),
        ]),
        html.Div(className="row", children=[
            zone_table("t-top-pu", "Top 10 pickup zones", "Pickups"),
            zone_table("t-top-do", "Top 10 drop-off zones", "Drop-offs"),
            zone_table("t-ratio-hi", "Highest pickup / drop-off ratio", "Ratio"),
            zone_table("t-ratio-lo", "Lowest pickup / drop-off ratio", "Ratio"),
        ]),
        html.Div("Ratio rankings are limited to zones with at least 30 sampled pickups and "
                 "drop-offs, so a handful of trips cannot produce a spurious extreme.",
                 className="note"),
    ]),

    html.Div(className="footer", children=[
        html.Span("Data: NYC Taxi & Limousine Commission · "),
        html.A("TLC Trip Record Data",
               href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
               target="_blank", rel="noopener noreferrer"),
        html.Span(f" · Counts scaled ×{SCALE} from the 1% sample"),
    ]),
])


# --------------------------------------------------------------------------- KPIs
@app.callback(
    Output("kpis", "children"),
    Input("f-months", "value"), Input("f-borough", "value"),
    Input("f-vendor", "value"), Input("f-tod", "value"),
)
def update_kpis(months, boroughs, vendors, tod):
    df = apply_filters(months, boroughs, vendors, tod)
    if df.empty:
        return [card("No data", "—", "Loosen the filters")]

    busiest = int(df["hour"].value_counts().idxmax())
    return [
        card("Trips", f"{len(df) * SCALE / 1e6:.2f}M", f"{len(df):,} sampled"),
        card("Revenue", f"${df['total_amount'].sum() * SCALE / 1e6:.0f}M", "scaled to population"),
        card("Avg fare", f"${df['fare_amount'].mean():.2f}",
             f"${df['total_amount'].mean():.2f} incl. charges"),
        card("Avg distance", f"{df['trip_distance'].mean():.2f} mi",
             f"median {df['trip_distance'].median():.2f} mi"),
        card("Avg tip", f"{df['tip_percentages'].mean():.1f}%", "card trips only"),
        card("Busiest hour", f"{busiest:02d}:00",
             f"{df['hour'].value_counts().max() * SCALE:,.0f} trips"),
    ]


# --------------------------------------------------------------------------- 1 temporal
@app.callback(
    Output("g-temporal", "figure"), Output("g-heat-demand", "figure"),
    Input("temporal-grain", "value"),
    Input("f-months", "value"), Input("f-borough", "value"),
    Input("f-vendor", "value"), Input("f-tod", "value"),
)
def update_temporal(grain, months, boroughs, vendors, tod):
    df = apply_filters(months, boroughs, vendors, tod)
    if df.empty:
        return empty_fig(height=400), empty_fig(height=330)

    if grain == "hour":
        counts = df["hour"].value_counts().reindex(range(24), fill_value=0)
        labels = [f"{h:02d}" for h in counts.index]
        axis_title = "Hour of day"
    elif grain == "day":
        counts = df["day_name"].value_counts().reindex(DAYS, fill_value=0)
        labels = [d[:3] for d in counts.index]
        axis_title = "Day of week"
    else:
        counts = df["month"].value_counts().reindex(range(1, 13), fill_value=0)
        labels = [MONTHS[m - 1][:3] for m in counts.index]
        axis_title = "Month"

    values = counts.values * SCALE / 1e3
    peak_i = int(np.argmax(values)) if values.size else 0

    f1 = go.Figure(go.Scatter(
        x=labels, y=values, mode="lines+markers+text",
        line=dict(color=TAXI, width=2.5, shape="spline", smoothing=0.5),
        marker=dict(size=8, color=TAXI, line=dict(color=BG, width=1.5)),
        text=[f"{v:,.0f}" for v in values],
        textposition="top center",
        textfont=dict(color="#ffffff", size=10.5),
        fill="tozeroy", fillcolor="rgba(245,197,24,0.07)",
        hovertemplate="%{x} — %{y:,.0f}k trips<extra></extra>",
        cliponaxis=False,
    ))
    f1.update_layout(
        xaxis_title=axis_title, yaxis_title="Trips (thousands)",
        title=f"peak: {labels[peak_i]} · {values[peak_i]:,.0f}k trips",
        yaxis_range=[0, float(values.max()) * 1.22] if values.size else None,
    )
    f1.update_xaxes(type="category", tickangle=0)
    style_fig(f1, height=400, legend=False)

    piv = (df.pivot_table(index="day_name", columns="hour", values="total_amount", aggfunc="size")
             .reindex(DAYS).reindex(columns=range(24)))
    f2 = go.Figure(go.Heatmap(
        z=piv.values * SCALE, x=[f"{h:02d}" for h in piv.columns], y=[d[:3] for d in piv.index],
        colorscale=SEQ, hovertemplate="%{y} %{x}:00 — %{z:,.0f} trips<extra></extra>",
        colorbar=dict(title="Trips", outlinewidth=0, thickness=13),
    ))
    f2.update_layout(xaxis_title="Hour of day")
    style_fig(f2, height=330, legend=False)

    return f1, f2


# --------------------------------------------------------------------------- 2 revenue
@app.callback(
    Output("g-rev-month", "figure"), Output("g-rev-quarter", "figure"),
    Output("g-heat-fare", "figure"), Output("g-vendor-hour", "figure"),
    Output("g-fare-vs", "figure"), Output("g-payment", "figure"),
    Input("fare-mode", "value"),
    Input("f-months", "value"), Input("f-borough", "value"),
    Input("f-vendor", "value"), Input("f-tod", "value"),
)
def update_revenue(fare_mode, months, boroughs, vendors, tod):
    df = apply_filters(months, boroughs, vendors, tod)
    if df.empty:
        return (empty_fig(height=400), empty_fig(height=400), empty_fig(height=360),
                empty_fig(height=360), empty_fig(height=420), empty_fig(height=420))

    # --- monthly revenue, labelled with month-on-month change ---------------
    rev = df.groupby("month")["total_amount"].sum() * SCALE / 1e6
    pct = rev.pct_change() * 100
    text, marker_colours = [], []
    for m in rev.index:
        change = pct[m]
        if pd.isna(change):
            text.append(f"${rev[m]:.1f}M")
            marker_colours.append(TAXI)
        else:
            text.append(f"${rev[m]:.1f}M<br>({change:+.1f}%)")
            marker_colours.append(UP if change >= 0 else DOWN)

    f1 = go.Figure(go.Scatter(
        x=[MONTHS[m - 1][:3] for m in rev.index], y=rev.values,
        mode="lines+markers+text",
        line=dict(color=ACCENT, width=2.5, shape="spline", smoothing=0.5),
        marker=dict(size=10, color=marker_colours, line=dict(color=BG, width=1.5)),
        text=text, textposition="top center",
        textfont=dict(color="#ffffff", size=10),
        fill="tozeroy", fillcolor="rgba(77,163,255,0.07)",
        hovertemplate="%{x}: $%{y:.2f}M<extra></extra>",
        cliponaxis=False,
    ))
    f1.update_layout(
        yaxis_title="Revenue ($M)", xaxis_title="Month",
        title="green marker = up on previous month, red = down",
        yaxis_range=[0, float(rev.max()) * 1.32] if len(rev) else None,
    )
    f1.update_xaxes(type="category")
    style_fig(f1, height=400, legend=False)

    # --- quarterly share ----------------------------------------------------
    q = df.groupby("quarter")["total_amount"].sum() * SCALE / 1e6
    f2 = go.Figure(go.Pie(
        labels=[f"Q{i}" for i in q.index], values=q.values, hole=0.58, sort=False,
        marker=dict(colors=["#14496e", "#1b6d94", "#3f9bb5", TAXI],
                    line=dict(color=PANEL, width=2)),
        texttemplate="<b>%{label}</b> · %{percent}<br>$%{value:,.0f}M",
        textposition="outside", textfont=dict(size=10.5, color=TEXT),
        hovertemplate="%{label}: $%{value:,.1f}M<extra></extra>",
    ))
    f2.add_annotation(text=f"<b>${q.sum():,.0f}M</b><br><span style='font-size:10px'>total</span>",
                      showarrow=False, font=dict(size=16, color=TEXT))
    f2.update_layout(margin=dict(l=60, r=60, t=44, b=40))
    style_fig(f2, height=400, legend=False).update_layout(
        margin=dict(l=60, r=60, t=44, b=40))

    # --- fare per mile heatmap ---------------------------------------------
    priced = df[df["trip_distance"] > 0]
    piv = (priced.pivot_table(index="day_name", columns="hour",
                              values="fare_per_mile", aggfunc="mean")
                 .reindex(DAYS).reindex(columns=range(24)))
    f3 = go.Figure(go.Heatmap(
        z=piv.values, x=[f"{h:02d}" for h in piv.columns], y=[d[:3] for d in piv.index],
        colorscale="RdYlGn_r", zmin=4, zmax=20,
        hovertemplate="%{y} %{x}:00 — $%{z:.2f}/mi<extra></extra>",
        colorbar=dict(title="$/mile", outlinewidth=0, thickness=13),
    ))
    f3.update_layout(xaxis_title="Hour of day")
    f3.update_xaxes(dtick=2)
    style_fig(f3, height=360, legend=False)

    # --- vendor fare per mile ----------------------------------------------
    f4 = go.Figure()
    for vid, colour in [(1, ACCENT), (2, TAXI)]:
        sub = priced[priced["VendorID"] == vid]
        if sub.empty:
            continue
        s = sub.groupby("hour")["fare_per_mile"].mean().reindex(range(24))
        f4.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers",
                                name=VENDORS[vid].split(" (")[0],
                                line=dict(color=colour, width=2.5), marker=dict(size=6)))
    f4.update_layout(xaxis_title="Hour of day", yaxis_title="Fare per mile ($)")
    f4.update_xaxes(dtick=2)
    style_fig(f4, height=360)

    # --- fare against duration / passenger count ----------------------------
    if fare_mode == "duration":
        r = df["duration_min"].corr(df["fare_amount"])
        pts = df[(df["duration_min"] > 0) & (df["fare_amount"] > 0)]

        # Axes are capped so the dense region stays readable; a handful of
        # 25-hour meter errors would otherwise squeeze everything into the
        # left edge. The cap shrinks to fit when a filter selects less.
        x_max = min(DUR_X_CAP, float(pts["duration_min"].max()) * 1.04) if len(pts) else 1.0
        y_max = min(DUR_Y_CAP, float(pts["fare_amount"].max()) * 1.04) if len(pts) else 1.0

        shown = pts[(pts["duration_min"] <= x_max) & (pts["fare_amount"] <= y_max)]
        hidden = len(pts) - len(shown)
        # Downsampled for the browser; the correlation above uses every row.
        s = shown.sample(min(len(shown), 8000), random_state=42)

        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        note = f" · {hidden:,} trip(s) beyond the axes not shown" if hidden else ""
        f5 = go.Figure(go.Scattergl(
            x=s["duration_min"], y=s["fare_amount"], mode="markers",
            marker=dict(size=4.5, color=ACCENT, opacity=0.45, line=dict(width=0)),
            hovertemplate="%{x:,.0f} min — $%{y:.2f}<extra></extra>",
        ))
        f5.update_layout(
            xaxis_title="Trip duration (minutes)", yaxis_title="Fare ($)",
            title=f"correlation r = {r:.2f} — {strength} link between time and fare{note}",
            xaxis_range=[-x_max * 0.02, x_max],
            yaxis_range=[-y_max * 0.02, y_max],
        )
    else:
        # A box per count reads far better than a scatter, since passenger_count
        # only takes six discrete values.
        r = df["passenger_count"].corr(df["fare_amount"])
        f5 = go.Figure()
        pax_df = df[df["passenger_count"].between(1, 6)]
        means = []
        y_max = (min(PAX_Y_CAP, float(pax_df["fare_amount"].max()) * 1.06)
                 if len(pax_df) else 1.0)
        for pc in range(1, 7):
            sub = pax_df[pax_df["passenger_count"] == pc]["fare_amount"]
            if sub.empty:
                continue
            means.append(float(sub.mean()))    # mean over every trip, not just visible ones
            f5.add_trace(go.Box(
                y=sub, name=str(pc), marker_color=TAXI, line_width=1.5,
                boxmean=True,                  # dashed line at the mean
                boxpoints="outliers",
                marker=dict(size=3, opacity=0.45, color=ACCENT),
                fillcolor="rgba(245,197,24,0.14)",
                hovertemplate=f"{pc} passenger(s)<br>$%{{y:,.2f}}<extra></extra>",
            ))

        # Mean printed above each box; the dashed line inside marks the same value.
        # x is the category *position*, not the passenger count — passing the
        # label ("1") makes Plotly read it as index 1 and shifts every label one
        # box to the right. Opaque background so tall outliers don't run through
        # the text.
        for i, mu in enumerate(means):
            f5.add_annotation(
                x=i, y=0.99, yref="paper", yanchor="top",
                text=f"μ ${mu:,.2f}", showarrow=False,
                font=dict(color="#ffffff", size=11),
                bgcolor=PANEL, borderpad=3,
            )
        f5.update_layout(
            xaxis_title="Passengers", yaxis_title="Fare ($)",
            xaxis_type="category",
            title=f"correlation r = {r:.2f} — essentially none · dashed line = mean",
            yaxis_range=[0, y_max],
        )
    style_fig(f5, height=420, legend=False)

    # --- payment mix --------------------------------------------------------
    pay = df["payment_label"].value_counts()
    f6 = go.Figure(go.Pie(
        labels=pay.index, values=pay.values, hole=0.58, sort=True,
        marker=dict(colors=[TAXI, "#14496e", "#3f9bb5", "#84c3c9"],
                    line=dict(color=PANEL, width=2)),
        texttemplate="%{percent}", textposition="inside",
        insidetextorientation="horizontal",
        hovertemplate="%{label}: %{value:,} trips (%{percent})<extra></extra>",
    ))
    f6.add_annotation(text=f"<b>{pay.sum() * SCALE / 1e6:.1f}M</b>"
                           "<br><span style='font-size:10px'>trips</span>",
                      showarrow=False, font=dict(size=16, color=TEXT))
    style_fig(f6, height=380).update_layout(
        # Hide labels on slices too small to hold them, instead of overlapping.
        uniformtext=dict(minsize=11, mode="hide"),
        margin=dict(l=20, r=20, t=44, b=64),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", title_text=""),
    )

    return f1, f2, f3, f4, f5, f6


# --------------------------------------------------------------------------- 3 geospatial
def zone_metrics(df: pd.DataFrame) -> pd.DataFrame:
    pu = df.groupby("PULocationID").agg(
        trips=("PULocationID", "size"),
        pax=("passenger_count", "mean"),
        extra=("extra_applied", "sum"),
    )
    do = df.groupby("DOLocationID").size().rename("dropoffs")
    m = pu.join(do, how="outer").fillna({"dropoffs": 0, "trips": 0, "extra": 0})
    m["ratio"] = (m["trips"] / m["dropoffs"].replace(0, pd.NA)).round(2)
    names = (df[["PULocationID", "zone", "borough"]]
             .drop_duplicates("PULocationID").set_index("PULocationID"))
    return m.join(names).reset_index(names="LocationID")


METRIC_LABEL = {
    "trips": "Pickups",
    "ratio": "Pickup / drop-off ratio",
    "pax": "Avg passengers",
    "extra": "Trips with extra charges",
}


@app.callback(
    Output("g-map", "figure"),
    Output("t-top-pu", "data"), Output("t-top-do", "data"),
    Output("t-ratio-hi", "data"), Output("t-ratio-lo", "data"),
    Input("geo-metric", "value"),
    Input("f-months", "value"), Input("f-borough", "value"),
    Input("f-vendor", "value"), Input("f-tod", "value"),
)
def update_geo(metric, months, boroughs, vendors, tod):
    df = apply_filters(months, boroughs, vendors, tod)
    if df.empty:
        return empty_fig(height=540), [], [], [], []

    m = zone_metrics(df)
    label = METRIC_LABEL[metric]
    plot_m = m.dropna(subset=[metric]) if metric == "ratio" else m

    fmap = px.choropleth_mapbox(
        plot_m, geojson=ZONES_GEO, locations="LocationID",
        featureidkey="properties.LocationID", color=metric,
        color_continuous_scale=SEQ,
        range_color=(0, 3) if metric == "ratio" else (0, plot_m[metric].quantile(0.97)),
        mapbox_style="carto-darkmatter", zoom=8.9,
        center={"lat": 40.72, "lon": -73.94}, opacity=0.82,
        hover_name="zone",
        hover_data={"borough": True, metric: ":.2f", "LocationID": False},
        labels={metric: label},
    )
    fmap.update_layout(
        paper_bgcolor=PANEL, font=dict(color=TEXT, size=12, family=FONT), autosize=True,
        margin=dict(l=0, r=0, t=4, b=0), hoverlabel=HOVER,
        coloraxis_colorbar=dict(title=label, outlinewidth=0, thickness=13),
    )

    def rows(frame, col, fmt, ascending=False):
        d = frame.dropna(subset=[col, "zone"]).sort_values(col, ascending=ascending).head(10)
        return [{"zone": r["zone"], "value": fmt(r[col])} for _, r in d.iterrows()]

    counts = lambda v: f"{int(v) * SCALE:,}"
    ratio = lambda v: f"{v:.2f}"

    do = (df.groupby("DOLocationID").size().rename("dropoffs").reset_index()
            .merge(m[["LocationID", "zone"]], left_on="DOLocationID",
                   right_on="LocationID", how="left"))

    # Enough traffic that the ratio is meaningful rather than an artefact.
    solid = m[(m["trips"] >= 30) & (m["dropoffs"] >= 30)]

    return (fmap,
            rows(m, "trips", counts),
            rows(do, "dropoffs", counts),
            rows(solid, "ratio", ratio),
            rows(solid, "ratio", ratio, ascending=True))


if __name__ == "__main__":
    print(f"Loaded {len(TRIPS):,} cleaned trips and {len(ZONES_GEO['features'])} taxi zones")
    print("Dashboard running at http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
