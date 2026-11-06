import base64
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PJM Energy Forecasting",
    page_icon="⚡",
    layout="wide",
)

FORECAST_FILE = "30_day_forecast.csv"
RESULTS_FILE = "model_comparison.csv"
MODEL_FILE = "random_forest_model.pkl"
IMPORTANCE_FILE = "random_forest_feature_importance.csv"

CYAN = "#22D3EE"
VIOLET = "#A78BFA"
AMBER = "#FBBF24"
GREEN = "#34D399"
PINK = "#F472B6"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }

.stApp {
  background:
    linear-gradient(135deg, rgba(6,10,28,.93), rgba(20,10,48,.90), rgba(4,32,48,.92), rgba(6,10,28,.93)),
    url("__BG__") center / cover fixed;
  background-size: 400% 400%, cover;
  animation: gradientShift 22s ease infinite;
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%, center; }
  50%  { background-position: 100% 50%, center; }
  100% { background-position: 0% 50%, center; }
}
[data-testid="stHeader"] { background: transparent; }
.block-container { position: relative; z-index: 1; padding-top: 2rem; }

.orb { position: fixed; border-radius: 50%; filter: blur(90px); z-index: -1; pointer-events: none; opacity: .35; }
.orb.o1 { width: 420px; height: 420px; background: #22D3EE; top: -80px; left: -80px; animation: drift1 18s ease-in-out infinite; }
.orb.o2 { width: 480px; height: 480px; background: #A78BFA; bottom: -120px; right: -100px; animation: drift2 22s ease-in-out infinite; }
.orb.o3 { width: 300px; height: 300px; background: #F472B6; top: 45%; left: 55%; opacity: .18; animation: drift1 26s ease-in-out infinite reverse; }
@keyframes drift1 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(120px,90px) scale(1.15); } }
@keyframes drift2 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-140px,-80px) scale(1.2); } }

[data-testid="stSidebar"] {
  background: rgba(10,15,35,.72);
  backdrop-filter: blur(14px);
  border-right: 1px solid rgba(255,255,255,.08);
}

.hero { display: flex; align-items: center; gap: 22px; margin-bottom: 10px; animation: fadeUp .8s ease both; }
.bolt { font-size: 64px; animation: float 3.2s ease-in-out infinite; filter: drop-shadow(0 0 18px rgba(251,191,36,.7)); }
@keyframes float { 0%,100% { transform: translateY(0) rotate(-4deg); } 50% { transform: translateY(-12px) rotate(4deg); } }
.title {
  font-size: 2.5rem; font-weight: 700; margin: 0; line-height: 1.15;
  background: linear-gradient(90deg, #22D3EE, #A78BFA, #F472B6, #22D3EE);
  background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 6s linear infinite;
}
@keyframes shimmer { to { background-position: 300% 0; } }
.subtitle { color: #9FB3C8; margin: 6px 0 10px 0; font-size: 1rem; }
.badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 5px 14px; border-radius: 999px;
  background: rgba(34,211,238,.12); border: 1px solid rgba(34,211,238,.4); color: #7DE9F9; font-size: .8rem;
}
.dot { width: 9px; height: 9px; border-radius: 50%; background: #34D399; animation: pulse 1.6s infinite; }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(52,211,153,.7); }
  70%  { box-shadow: 0 0 0 10px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}

.ticker { overflow: hidden; white-space: nowrap; margin: 14px 0 22px 0; border-radius: 12px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); padding: 9px 0; }
.ticker-track { display: inline-block; animation: scroll 38s linear infinite; }
.ticker-track span { margin: 0 34px; color: #CFE3F5; font-size: .9rem; }
.ticker:hover .ticker-track { animation-play-state: paused; }
@keyframes scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

@keyframes fadeUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }

[data-testid="stMetric"] {
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px; padding: 16px 20px; backdrop-filter: blur(10px);
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  animation: fadeUp .7s ease both;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-6px); border-color: rgba(34,211,238,.7);
  box-shadow: 0 12px 34px rgba(34,211,238,.25);
}
[data-testid="stMetricValue"] { font-weight: 700; text-shadow: 0 0 18px rgba(34,211,238,.45); }
[data-testid="stMetricLabel"] { color: #9FB3C8; }

[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"],
[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { animation-delay: .12s; }
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"],
[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { animation-delay: .24s; }
[data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetric"],
[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { animation-delay: .36s; }

[data-testid="stPlotlyChart"] {
  background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px; padding: 8px; animation: fadeUp .9s ease both;
  transition: box-shadow .3s ease;
}
[data-testid="stPlotlyChart"]:hover { box-shadow: 0 10px 40px rgba(167,139,250,.22); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; animation: fadeUp 1s ease both; }
[data-testid="stAlert"] { border-radius: 14px; animation: fadeUp .8s ease both; }

button[data-baseweb="tab"] { font-weight: 600; transition: color .2s ease, transform .2s ease; }
button[data-baseweb="tab"]:hover { transform: translateY(-2px); color: #22D3EE; }
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #22D3EE, #A78BFA); height: 3px; }

[data-testid="stDownloadButton"] button {
  background: linear-gradient(90deg, #22D3EE, #A78BFA); color: #0B1020; font-weight: 600;
  border: none; border-radius: 12px; padding: 8px 20px; transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stDownloadButton"] button:hover { transform: translateY(-3px) scale(1.03); box-shadow: 0 8px 26px rgba(34,211,238,.45); color: #0B1020; }

.stApp { color: #F4F8FF; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color: #F1F6FF; font-weight: 500; font-size: 1.02rem; line-height: 1.65; }
[data-testid="stMarkdownContainer"] strong { color: #67E8F9; font-weight: 700; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #BFD0E6 !important; font-weight: 500; }

h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; }
[data-testid="stHeading"] h3 {
  border-left: 5px solid #22D3EE; padding-left: 12px; letter-spacing: .2px;
  text-shadow: 0 0 20px rgba(34,211,238,.35);
}

.title { font-size: 2.9rem; font-weight: 800; letter-spacing: .3px; filter: drop-shadow(0 0 14px rgba(34,211,238,.35)); }
.subtitle { color: #D4E2F4; font-weight: 500; font-size: 1.08rem; }
.badge { color: #A5F3FC; font-weight: 600; background: rgba(34,211,238,.16); border: 1px solid rgba(34,211,238,.6); }

.ticker { background: rgba(6,12,32,.78); border: 1px solid rgba(34,211,238,.35); box-shadow: 0 0 24px rgba(34,211,238,.12); padding: 12px 0; }
.ticker-track span { color: #FFFFFF; font-weight: 700; font-size: 1.02rem; letter-spacing: .3px; }

[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(10,18,46,.88), rgba(14,24,60,.72));
  border: 1px solid rgba(255,255,255,.14); border-top: 4px solid #22D3EE;
  box-shadow: 0 8px 28px rgba(0,0,0,.35);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: #DCE8F8 !important; font-weight: 700 !important; font-size: .92rem !important;
  text-transform: uppercase; letter-spacing: .6px;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
  font-size: 2.3rem !important; font-weight: 800 !important; color: #67E8F9;
  text-shadow: 0 0 22px rgba(34,211,238,.6);
}
[data-testid="stMetricDelta"] { font-weight: 700 !important; font-size: .95rem !important; }

[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"], [data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-top-color: #F472B6; }
[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetricValue"], [data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"],
[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetricValue"] > div, [data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] > div { color: #F9A8D4; text-shadow: 0 0 22px rgba(244,114,182,.6); }

[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"], [data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-top-color: #FBBF24; }
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetricValue"], [data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"],
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetricValue"] > div, [data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] > div { color: #FDE68A; text-shadow: 0 0 22px rgba(251,191,36,.6); }

[data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetric"], [data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-top-color: #34D399; }
[data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetricValue"], [data-testid="column"]:nth-of-type(4) [data-testid="stMetricValue"],
[data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetricValue"] > div, [data-testid="column"]:nth-of-type(4) [data-testid="stMetricValue"] > div { color: #6EE7B7; text-shadow: 0 0 22px rgba(52,211,153,.6); }

div[data-baseweb="tab-panel"] {
  background: rgba(7,12,32,.62); border: 1px solid rgba(255,255,255,.09);
  border-radius: 20px; padding: 22px 24px; margin-top: 12px; backdrop-filter: blur(8px);
}

button[data-baseweb="tab"] p { font-size: 1.05rem !important; font-weight: 700 !important; color: #DCE8F8; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #22D3EE !important; text-shadow: 0 0 14px rgba(34,211,238,.6); }

[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #FFFFFF !important; font-weight: 600 !important; }
[data-testid="stSidebar"] h1 { color: #22D3EE !important; font-weight: 800 !important; }
[data-testid="stSliderThumbValue"] { color: #67E8F9 !important; font-weight: 800 !important; font-size: .95rem !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #BFD0E6 !important; font-weight: 600; }

[data-testid="stAlert"] p { font-weight: 700 !important; color: #FFFFFF !important; }
[data-testid="stDownloadButton"] button { font-size: 1rem; font-weight: 700; padding: 10px 26px; box-shadow: 0 6px 22px rgba(34,211,238,.35); }
</style>
"""


@st.cache_data
def get_background():
    for name in ("bg.jpg", "bg.jpeg", "bg.png", "bg.webp"):
        p = Path(name)
        if p.exists():
            ext = p.suffix.lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">'
        '<defs><pattern id="g" width="60" height="60" patternUnits="userSpaceOnUse">'
        '<path d="M60 0H0V60" fill="none" stroke="#6ea8ff" stroke-opacity="0.10" stroke-width="1"/></pattern></defs>'
        '<rect width="1400" height="900" fill="url(#g)"/>'
        '<g fill="none" stroke="#22D3EE" stroke-opacity="0.35" stroke-width="2">'
        '<path d="M0 620 C200 540 320 700 520 620 S860 520 1040 600 S1300 560 1400 520"/>'
        '<path d="M0 700 C240 640 360 780 580 700 S900 620 1100 690 S1320 650 1400 610" stroke="#A78BFA"/>'
        '<path d="M0 220 C180 160 360 300 560 220 S900 120 1080 200 S1300 160 1400 120" stroke="#F472B6" stroke-opacity="0.25"/>'
        '</g>'
        '<g fill="#22D3EE" fill-opacity="0.55">'
        '<circle cx="520" cy="620" r="5"/><circle cx="1040" cy="600" r="5"/><circle cx="320" cy="680" r="4"/>'
        '<circle cx="860" cy="560" r="4"/><circle cx="560" cy="220" r="5"/><circle cx="1080" cy="200" r="5"/>'
        '</g></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def style_fig(fig, height):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", color="#FFFFFF", size=14),
        hoverlabel=dict(bgcolor="#0B1228", bordercolor=CYAN, font=dict(color="#FFFFFF", size=14)),
        legend=dict(font=dict(color="#FFFFFF", size=13)),
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    axis_style = dict(
        gridcolor="rgba(255,255,255,0.10)",
        tickfont=dict(color="#E6F0FF", size=13),
        title_font=dict(color="#FFFFFF", size=14),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def animated_line(x, y):
    frames = [
        go.Frame(data=[go.Scatter(x=x[:k], y=y[:k])], name=str(k))
        for k in range(2, len(x) + 1)
    ]
    fig = go.Figure(
        data=[go.Scatter(
            x=x, y=y, mode="lines+markers", name="Forecast",
            line=dict(color=CYAN, width=3, shape="spline"),
            marker=dict(size=7, color=VIOLET),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} MW<extra></extra>",
        )],
        frames=frames,
    )
    pad = (max(y) - min(y)) * 0.15 or 100
    fig.update_layout(
        xaxis=dict(range=[x.min() - pd.Timedelta(days=1), x.max() + pd.Timedelta(days=1)]),
        yaxis=dict(range=[min(y) - pad, max(y) + pad], title="Demand (MW)"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0, y=1.14, xanchor="left",
            buttons=[dict(
                label="▶ Play animation", method="animate",
                args=[None, dict(frame=dict(duration=140, redraw=True),
                                 fromcurrent=False, transition=dict(duration=0))],
            )],
        )],
    )
    return fig


@st.cache_data
def load_forecast(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date").sort_index()


@st.cache_data
def load_results(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_importance(model, csv_path):
    if model is not None and hasattr(model, "feature_importances_"):
        names = getattr(model, "feature_names_in_",
                        [f"feature_{i}" for i in range(len(model.feature_importances_))])
        imp = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_})
    elif Path(csv_path).exists():
        imp = pd.read_csv(csv_path)
    else:
        return None
    return imp.sort_values("Importance", ascending=False)

st.markdown(CSS.replace("__BG__", get_background()), unsafe_allow_html=True)
st.markdown('<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>',
            unsafe_allow_html=True)

missing = [f for f in (FORECAST_FILE, RESULTS_FILE) if not Path(f).exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Keep them in the same folder as app.py.")
    st.stop()

forecast = load_forecast(FORECAST_FILE)
results = load_results(RESULTS_FILE)
model = load_model(MODEL_FILE) if Path(MODEL_FILE).exists() else None

n_days = len(forecast)

st.sidebar.title("⚡ Controls")

horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=min(7, n_days),
    max_value=n_days,
    value=n_days,
    help="Show only the first N days of the forecast.",
)

st.sidebar.markdown("**Chart options**")
chart_type = st.sidebar.radio("Chart type", ["Line", "Bar"], horizontal=True)
show_weekends = st.sidebar.checkbox("Highlight weekends", value=True)
animate = st.sidebar.checkbox("Animated reveal (use ▶ Play)", value=False)

df = forecast.head(horizon).copy()
df["Forecast_MW"] = df["Predicted_MW"]
df["Weekday"] = df.index.day_name()
df["Is_Weekend"] = df.index.dayofweek >= 5
df["Change_%"] = df["Forecast_MW"].pct_change() * 100

st.markdown(
    """
<div class="hero">
  <div class="bolt">⚡</div>
  <div>
    <h1 class="title">PJM Energy Consumption Forecasting</h1>
    <p class="subtitle">Machine learning based electricity demand forecasting for the PJM West region</p>
    <span class="badge"><span class="dot"></span> Random Forest model · Trained on 2002–2018 hourly data</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

_peak_d = df["Forecast_MW"].idxmax()
_low_d = df["Forecast_MW"].idxmin()
_items = (
    f"<span>⚡ Peak {df['Forecast_MW'].max():,.0f} MW on {_peak_d:%a %d %b}</span>"
    f"<span>🔻 Lowest {df['Forecast_MW'].min():,.0f} MW on {_low_d:%a %d %b}</span>"
    f"<span>📊 Average {df['Forecast_MW'].mean():,.0f} MW</span>"
    f"<span>📅 {horizon}-day outlook: {df.index.min():%d %b} to {df.index.max():%d %b %Y}</span>"
)
st.markdown(
    f'<div class="ticker"><div class="ticker-track">{_items}{_items}</div></div>',
    unsafe_allow_html=True,
)

peak_date = df["Forecast_MW"].idxmax()
low_date = df["Forecast_MW"].idxmin()
weekday_avg = df.loc[~df["Is_Weekend"], "Forecast_MW"].mean()
weekend_avg = df.loc[df["Is_Weekend"], "Forecast_MW"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Peak demand ({peak_date:%a, %d %b})", f"{df['Forecast_MW'].max():,.0f} MW")
k2.metric(f"Lowest demand ({low_date:%a, %d %b})", f"{df['Forecast_MW'].min():,.0f} MW")
k3.metric("Average demand", f"{df['Forecast_MW'].mean():,.0f} MW")
if np.isnan(weekend_avg) or np.isnan(weekday_avg):
    k4.metric("Weekend vs weekday", "n/a")
else:
    diff = (weekend_avg / weekday_avg - 1) * 100
    k4.metric("Weekend vs weekday", f"{weekend_avg:,.0f} MW", f"{diff:+.1f}% vs weekdays", delta_color="off")

tab_fc, tab_models, tab_insights, tab_about = st.tabs(
    ["📈 Forecast", "🏆 Model comparison", "🔍 Insights", "ℹ️ About"]
)

with tab_fc:
    st.subheader(f"{horizon}-day energy demand forecast")

    if animate:
        fig = animated_line(df.index, df["Forecast_MW"].to_numpy())
    else:
        fig = go.Figure()

    if show_weekends:
        for d in df.index[df["Is_Weekend"]]:
            fig.add_vrect(
                x0=d - pd.Timedelta(hours=12),
                x1=d + pd.Timedelta(hours=12),
                fillcolor="white",
                opacity=0.06,
                line_width=0,
            )

    if not animate:
        if chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Forecast_MW"], mode="lines+markers",
                name="Forecast", line=dict(color=CYAN, width=3, shape="spline"),
                marker=dict(size=7, color=VIOLET),
                customdata=df["Weekday"],
                hovertemplate="%{x|%d %b %Y} (%{customdata})<br>%{y:,.0f} MW<extra></extra>",
            ))
        else:
            fig.add_trace(go.Bar(
                x=df.index, y=df["Forecast_MW"], name="Forecast",
                marker=dict(color=df["Forecast_MW"], colorscale="Plasma"),
                customdata=df["Weekday"],
                hovertemplate="%{x|%d %b %Y} (%{customdata})<br>%{y:,.0f} MW<extra></extra>",
            ))

    fig.add_trace(go.Scatter(
        x=[peak_date], y=[df.loc[peak_date, "Forecast_MW"]],
        mode="markers", name="Peak",
        marker=dict(symbol="star", size=18, color=AMBER, line=dict(color="white", width=1)),
        hoverinfo="skip",
    ))

    fig.add_annotation(
        x=peak_date, y=df.loc[peak_date, "Forecast_MW"],
        text=f"<b>Peak {df['Forecast_MW'].max():,.0f} MW</b>",
        showarrow=True, arrowhead=2, arrowcolor=AMBER, ax=0, ay=-48,
        font=dict(color=AMBER, size=15), bgcolor="rgba(8,14,36,0.9)",
        bordercolor=AMBER, borderwidth=1.5, borderpad=6,
    )

    style_fig(fig, 500)
    fig.update_layout(
        yaxis_title="Demand (MW)",
        xaxis_title=None,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.35),
    )
    if not animate:
        fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)

    st.subheader("Forecast values")
    table = df[["Weekday", "Forecast_MW", "Change_%"]].copy()
    table.index = table.index.strftime("%d %b %Y")

    lo, hi = table["Forecast_MW"].min(), table["Forecast_MW"].max()

    def shade_demand(col):
        cells = []
        for v in col:
            t = (v - lo) / (hi - lo) if hi > lo else 0.5
            r, g, b = int(15 + 19 * t), int(45 + 166 * t), int(100 + 138 * t)
            fg = "#05101F" if t > 0.6 else "#FFFFFF"
            cells.append(f"background-color: rgb({r},{g},{b}); color: {fg}; font-weight: 700")
        return cells

    def colour_change(col):
        return [
            "color: #FF8FC7; font-weight: 700" if (pd.notna(v) and v > 0)
            else "color: #6EE7B7; font-weight: 700"
            for v in col
        ]

    styled_fc = (
        table.style
        .apply(shade_demand, subset=["Forecast_MW"])
        .apply(colour_change, subset=["Change_%"])
        .format({"Forecast_MW": "{:,.0f} MW", "Change_%": "{:+.1f}%"}, na_rep="–")
    )
    st.dataframe(styled_fc)

    st.download_button(
        "⬇️ Download forecast (CSV)",
        data=df[["Forecast_MW"]].round(2).to_csv(),
        file_name=f"energy_forecast_{horizon}d.csv",
        mime="text/csv",
    )

with tab_models:
    st.subheader("How the models performed on the held-out final year")

    metrics = [c for c in ["MAE", "RMSE", "R2", "MAPE"] if c in results.columns]
    lower_better = {"MAE", "RMSE", "MAPE"}

    metric = st.selectbox("Metric", metrics, index=metrics.index("MAPE") if "MAPE" in metrics else 0)

    best_idx = results[metric].idxmin() if metric in lower_better else results[metric].idxmax()
    best_model = results.loc[best_idx, "Model"]
    colors = [VIOLET if i != best_idx else GREEN for i in results.index]

    bar = go.Figure(go.Bar(
        x=results["Model"], y=results[metric], marker_color=colors,
        text=results[metric].round(2), textposition="outside", cliponaxis=False,
        textfont=dict(size=15, color="#FFFFFF"),
    ))
    note = "lower is better" if metric in lower_better else "higher is better"
    style_fig(bar, 420)
    bar.update_layout(yaxis_title=f"{metric} ({note})", xaxis_title=None)
    st.plotly_chart(bar)

    st.success(f"Best model for {metric}: **{best_model}**")

    styled = results.style.format({c: "{:,.2f}" for c in metrics})
    styled = styled.highlight_min(subset=[c for c in metrics if c in lower_better], color="#14532d")
    if "R2" in metrics:
        styled = styled.highlight_max(subset=["R2"], color="#14532d")
    styled = styled.set_properties(**{"color": "#FFFFFF", "font-weight": "700"})
    st.dataframe(styled, hide_index=True)

    st.caption(
        "Note: ARIMA and SARIMA forecast the whole test year in one go, while XGBoost, "
        "Random Forest, LSTM and GRU use the previous day's actual value. "
        "The two groups are therefore not directly comparable."
    )

with tab_insights:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Average demand by weekday")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = df.groupby("Weekday")["Forecast_MW"].mean().reindex(order).dropna().reset_index()
        fig_wd = px.bar(wd, x="Weekday", y="Forecast_MW", color="Forecast_MW",
                        color_continuous_scale="Plasma")
        style_fig(fig_wd, 380)
        fig_wd.update_layout(yaxis_title="Avg demand (MW)", xaxis_title=None,
                             coloraxis_showscale=False)
        st.plotly_chart(fig_wd)

    with c2:
        st.subheader("Day-over-day change (%)")
        chg = df.dropna(subset=["Change_%"]).reset_index()
        fig_chg = px.bar(chg, x="Date", y="Change_%",
                         color=chg["Change_%"] > 0,
                         color_discrete_map={True: PINK, False: GREEN})
        style_fig(fig_chg, 380)
        fig_chg.update_layout(yaxis_title="Change (%)", xaxis_title=None, showlegend=False)
        st.plotly_chart(fig_chg)

    st.subheader("What drives the model's predictions?")
    imp = load_importance(model, IMPORTANCE_FILE)
    if imp is not None:
        top_n = st.slider("Number of features to show", 5, len(imp), min(10, len(imp)))
        top = imp.head(top_n).iloc[::-1]
        fig_imp = px.bar(top, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis")
        style_fig(fig_imp, 40 * top_n + 120)
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp)
        st.caption("Yesterday's demand (lag_1) is usually the strongest signal, "
                   "followed by day-of-week patterns.")
    else:
        st.info("Add random_forest_feature_importance.csv next to app.py to see feature importance.")

with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**Goal:** forecast electricity demand for the PJM West region so that demand patterns
can be understood and planned for.

**Data:** about 143,000 hourly records (2002 to 2018). Duplicate timestamps were
resolved and missing hours were filled using time-based interpolation.

**Approach**
1. Exploratory analysis of hourly, weekly, seasonal and holiday patterns
2. Feature engineering: calendar features, lags, rolling statistics and cyclical encodings
3. Benchmarked six models: ARIMA, SARIMA, XGBoost, Random Forest, LSTM and GRU
4. Chronological split with the final year held out for testing
5. Selected Random Forest as the best model and generated the 30-day forecast

**Tech stack:** Python, Pandas, Scikit-learn, XGBoost, PyTorch, Statsmodels, Plotly, Streamlit

**Limitations:** reported accuracy is for next-day predictions on daily average demand.
Multi-day forecasts accumulate error, and weather data is not used.
        """
    )