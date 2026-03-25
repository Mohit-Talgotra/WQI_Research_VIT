import os
import json
import joblib
import numpy as np
import pandas as pd
import folium
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()

# Page config
st.set_page_config(
    page_title="WQI Forecast — Salem District",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #0f1117;
        color: #e8e8e8;
    }
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: #7ecfff;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 0.8rem;
        color: #666;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2a3040;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 500;
        color: #7ecfff;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .wqi-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    .confidence-high   { color: #4ade80; }
    .confidence-medium { color: #facc15; }
    .confidence-low    { color: #f87171; }
    .stSlider > div > div > div { background: #7ecfff !important; }
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2433;
    }
    .block-header {
        font-size: 0.7rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #1e2433;
    }
</style>
""", unsafe_allow_html=True)

# Paths
MODEL_PATH     = Path(os.environ.get("FORECAST_MODEL_PATH",      "data/wqi_forecaster.joblib"))
DATASET_PATH   = Path(os.environ.get("FORECAST_DATASET_PATH",    "data/forecast_dataset.csv"))
GEOCODED_PATH  = Path(os.environ.get("GEOCODED_LOCATIONS_PATH",  "data/locations_geocoded.csv"))
MONTHLY_PATH   = Path(os.environ.get("MONTHLY_WQI_PATH",         "data/monthly_wqi_dataset.csv"))
CV_RESULTS_DIR = Path("forecasting_results")

FEATURE_COLS = [
    "Block_encoded", "Location_encoded",
    "Year", "Month_sin", "Month_cos",
    "WQI_lag_1", "WQI_lag_2", "WQI_lag_3",
    "WQI_rolling_mean_2", "WQI_rolling_mean_3",
]

# WQI helpers
def wqi_category(wqi: float) -> tuple:
    if wqi <= 25:   return "Excellent",  "#22c55e"
    if wqi <= 50:   return "Good",       "#84cc16"
    if wqi <= 75:   return "Poor",       "#f59e0b"
    if wqi <= 100:  return "Very Poor",  "#ef4444"
    return              "Unsuitable",   "#7c3aed"

def confidence_label(rmse: float) -> tuple:
    if rmse <= 10:  return "High",   "confidence-high"
    if rmse <= 20:  return "Medium", "confidence-medium"
    return                "Low",    "confidence-low"

# Load resources
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df       = pd.read_csv(DATASET_PATH)
    geo      = pd.read_csv(GEOCODED_PATH)
    monthly  = pd.read_csv(MONTHLY_PATH)
    monthly["Date"] = pd.to_datetime(monthly["Date"])

    enc_block    = pd.read_csv(DATASET_PATH.parent / "encoder_block.csv")
    enc_location = pd.read_csv(DATASET_PATH.parent / "encoder_location.csv")
    block_to_enc    = dict(zip(enc_block["Block"],    enc_block["Block_encoded"]))
    location_to_enc = dict(zip(enc_location["Location"], enc_location["Location_encoded"]))

    # Load CV RMSE per location (use best model results)
    rmse_map = {}
    for fname in ["cv_results_GradientBoosting.csv", "cv_results_RandomForest.csv"]:
        fpath = CV_RESULTS_DIR / fname
        if fpath.exists():
            cv = pd.read_csv(fpath)
            rmse_map = dict(zip(cv["location"], cv["rmse"]))
            break

    return df, geo, monthly, block_to_enc, location_to_enc, rmse_map

@st.cache_data
def get_prediction(location, block, target_date_str, _model, _df,
                   _block_to_enc, _location_to_enc):
    """Cached prediction — recomputes only when inputs change."""
    target_date = pd.Timestamp(target_date_str).replace(day=1)

    loc_history = (
        _df[_df["Location"] == location]
        .sort_values("Date")[["Date", "WQI_target"]]
        .assign(Date=lambda x: pd.to_datetime(x["Date"]))
        .set_index("Date")["WQI_target"]
    )

    def get_lag(months_back):
        lag_date  = target_date - pd.DateOffset(months=months_back)
        available = loc_history[loc_history.index <= lag_date]
        if available.empty:
            return np.nan
        return float(available.iloc[-1])

    lag1, lag2, lag3 = get_lag(1), get_lag(2), get_lag(3)
    rolling2 = np.nanmean([v for v in [lag1, lag2]       if not np.isnan(v)]) if not all(np.isnan(v) for v in [lag1, lag2])       else np.nan
    rolling3 = np.nanmean([v for v in [lag1, lag2, lag3] if not np.isnan(v)]) if not all(np.isnan(v) for v in [lag1, lag2, lag3]) else np.nan

    month     = target_date.month
    features  = pd.DataFrame([{
        "Block_encoded"      : _block_to_enc.get(block, 0),
        "Location_encoded"   : _location_to_enc.get(location, 0),
        "Year"               : target_date.year,
        "Month_sin"          : np.sin(2 * np.pi * month / 12),
        "Month_cos"          : np.cos(2 * np.pi * month / 12),
        "WQI_lag_1"          : lag1,
        "WQI_lag_2"          : lag2,
        "WQI_lag_3"          : lag3,
        "WQI_rolling_mean_2" : rolling2,
        "WQI_rolling_mean_3" : rolling3,
    }])[FEATURE_COLS]

    return float(_model.predict(features)[0])

# Load everything
with st.spinner("Loading model and data..."):
    model = load_model()
    df, geo, monthly, block_to_enc, location_to_enc, rmse_map = load_data()

df["Date"] = pd.to_datetime(df["Date"])

# Sidebar
with st.sidebar:
    st.markdown('<p class="main-title">💧 WQI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Salem District Forecast</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="block-header">Forecast Date</p>', unsafe_allow_html=True)

    min_date = date(2025, 1, 1)
    max_date = date(2026, 12, 1)

    selected_date = st.date_input(
        "Select month",
        value=date(2025, 6, 1),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed"
    )
    # Normalize to first of month
    selected_date = selected_date.replace(day=1)

    st.markdown("---")
    st.markdown('<p class="block-header">Filter by Block</p>', unsafe_allow_html=True)

    all_blocks   = sorted(df["Block"].unique())
    selected_blocks = st.multiselect(
        "Blocks",
        options=all_blocks,
        default=all_blocks,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<p class="block-header">WQI Legend</p>', unsafe_allow_html=True)
    for cat, color, rng in [
        ("Excellent",  "#22c55e", "≤ 25"),
        ("Good",       "#84cc16", "26–50"),
        ("Poor",       "#f59e0b", "51–75"),
        ("Very Poor",  "#ef4444", "76–100"),
        ("Unsuitable", "#7c3aed", "> 100"),
    ]:
        st.markdown(
            f'<span style="color:{color}">■</span> &nbsp;'
            f'<span style="font-size:0.8rem">{cat} ({rng})</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<p class="block-header">Confidence</p>', unsafe_allow_html=True)
    st.markdown(
        '<span class="confidence-high">● High</span> RMSE ≤ 10<br>'
        '<span class="confidence-medium">● Medium</span> RMSE 10–20<br>'
        '<span class="confidence-low">● Low</span> RMSE > 20',
        unsafe_allow_html=True
    )

# Main layout
st.markdown(
    f'<p class="main-title">Water Quality Index Forecast</p>'
    f'<p class="sub-title">Salem District, Tamil Nadu &nbsp;·&nbsp; '
    f'Predicting for {selected_date.strftime("%B %Y")}</p>',
    unsafe_allow_html=True
)


target_str = selected_date.strftime("%Y-%m-%d")

loc_data = (
    df[df["Block"].isin(selected_blocks)][["Block", "Location"]]
    .drop_duplicates()
    .merge(geo[["Block", "Location", "Latitude", "Longitude"]], on=["Block", "Location"], how="left")
    .dropna(subset=["Latitude", "Longitude"])
    .reset_index(drop=True)
)

predictions = []
for _, row in loc_data.iterrows():
    wqi_pred = get_prediction(
        row["Location"], row["Block"], target_str,
        model, df, block_to_enc, location_to_enc
    )
    cat, color = wqi_category(wqi_pred)
    rmse       = rmse_map.get(row["Location"], None)
    conf_label, conf_class = confidence_label(rmse) if rmse else ("Unknown", "confidence-medium")

    predictions.append({
        "Block"       : row["Block"],
        "Location"    : row["Location"],
        "Latitude"    : row["Latitude"],
        "Longitude"   : row["Longitude"],
        "WQI"         : round(wqi_pred, 1),
        "Category"    : cat,
        "Color"       : color,
        "RMSE"        : round(rmse, 1) if rmse else None,
        "Confidence"  : conf_label,
        "ConfClass"   : conf_class,
    })

pred_df = pd.DataFrame(predictions)


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{pred_df["WQI"].mean():.1f}</div>
        <div class="metric-label">Mean WQI</div>
    </div>''', unsafe_allow_html=True)
with col2:
    poor_count = (pred_df["WQI"] > 75).sum()
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value" style="color:#ef4444">{poor_count}</div>
        <div class="metric-label">Very Poor / Unsuitable</div>
    </div>''', unsafe_allow_html=True)
with col3:
    good_count = (pred_df["WQI"] <= 50).sum()
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value" style="color:#84cc16">{good_count}</div>
        <div class="metric-label">Good / Excellent</div>
    </div>''', unsafe_allow_html=True)
with col4:
    high_conf = (pred_df["Confidence"] == "High").sum()
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value" style="color:#4ade80">{high_conf}</div>
        <div class="metric-label">High Confidence</div>
    </div>''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


map_col, detail_col = st.columns([3, 1])

with map_col:
    # Build folium map centered on Salem district
    m = folium.Map(
        location=[11.65, 78.15],
        zoom_start=10,
        tiles="CartoDB dark_matter"
    )

    for _, row in pred_df.iterrows():
        # Circle marker sized by WQI, colored by category
        radius    = max(6, min(18, row["WQI"] / 8))
        conf_icon = "★" if row["Confidence"] == "High" else ("◆" if row["Confidence"] == "Medium" else "●")

        popup_html = f"""
        <div style="font-family: monospace; font-size: 12px; min-width: 180px;
                    background:#1a1f2e; color:#e8e8e8; padding:12px; border-radius:6px;
                    border: 1px solid {row['Color']}">
            <b style="color:{row['Color']}; font-size:14px">{row['Location']}</b><br>
            <span style="color:#888; font-size:10px">{row['Block']}</span><br><br>
            <span style="font-size:22px; color:{row['Color']}">{row['WQI']}</span>
            <span style="color:#888"> WQI</span><br>
            <span style="color:{row['Color']}">{row['Category']}</span><br><br>
            <span style="color:#888; font-size:10px">CONFIDENCE</span><br>
            <span style="color:{'#4ade80' if row['Confidence']=='High' else '#facc15' if row['Confidence']=='Medium' else '#f87171'}">
                {conf_icon} {row['Confidence']}
                {f"(RMSE {row['RMSE']})" if row['RMSE'] else ""}
            </span>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            color=row["Color"],
            fill=True,
            fill_color=row["Color"],
            fill_opacity=0.75,
            weight=2,
            tooltip=f"{row['Location']}: WQI {row['WQI']} ({row['Category']})",
            popup=folium.Popup(popup_html, max_width=220),
        ).add_to(m)

    map_data = st_folium(m, width=None, height=520, returned_objects=["last_object_clicked_popup"])

with detail_col:
    st.markdown('<p class="block-header">Location Detail</p>', unsafe_allow_html=True)

    # Extract clicked location name from popup
    clicked_loc = None
    if map_data and map_data.get("last_object_clicked_popup"):
        popup_text = map_data["last_object_clicked_popup"]
        # Match location name from pred_df
        for loc in pred_df["Location"].values:
            if loc in str(popup_text):
                clicked_loc = loc
                break

    if clicked_loc:
        row = pred_df[pred_df["Location"] == clicked_loc].iloc[0]
        cat, color = wqi_category(row["WQI"])

        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.1rem; color:{color}; font-weight:500">{clicked_loc}</div>
            <div style="font-size:0.7rem; color:#888">{row['Block']}</div>
            <br>
            <div class="metric-value" style="color:{color}">{row['WQI']}</div>
            <div class="metric-label">Predicted WQI — {selected_date.strftime('%b %Y')}</div>
            <br>
            <div style="background:{color}22; border:1px solid {color}; border-radius:4px;
                        padding:4px 10px; display:inline-block; font-size:0.75rem; color:{color}">
                {cat}
            </div>
            <br><br>
            <div class="metric-label">Forecast Confidence</div>
            <div class="{row['ConfClass']}" style="font-size:0.85rem">
                {row['Confidence']} {"· RMSE " + str(row['RMSE']) if row['RMSE'] else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Historical WQI trend chart
        st.markdown('<p class="block-header" style="margin-top:16px">Historical Trend</p>',
                    unsafe_allow_html=True)

        hist = (
            monthly[monthly["Location"] == clicked_loc]
            .sort_values("Date")[["Date", "WQI_mean"]]
            .dropna()
        )

        if not hist.empty:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=hist["Date"],
                y=hist["WQI_mean"],
                mode="lines+markers",
                name="Historical",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
            ))

            # Add predicted point
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(target_str)],
                y=[row["WQI"]],
                mode="markers",
                name="Predicted",
                marker=dict(size=12, color="#7ecfff", symbol="star",
                            line=dict(color="white", width=1)),
            ))

            # WQI category bands
            for lo, hi, cat_color, cat_name in [
                (0,   25,  "#22c55e22", "Excellent"),
                (25,  50,  "#84cc1622", "Good"),
                (50,  75,  "#f59e0b22", "Poor"),
                (75,  100, "#ef444422", "Very Poor"),
                (100, 150, "#7c3aed22", "Unsuitable"),
            ]:
                fig.add_hrect(y0=lo, y1=hi, fillcolor=cat_color,
                              line_width=0, annotation_text=cat_name,
                              annotation_position="right",
                              annotation_font_size=8,
                              annotation_font_color="#666")

            fig.update_layout(
                height=280,
                margin=dict(l=0, r=40, t=10, b=0),
                paper_bgcolor="#0f1117",
                plot_bgcolor="#0f1117",
                font=dict(family="DM Mono, monospace", color="#888", size=10),
                xaxis=dict(gridcolor="#1e2433", showgrid=True),
                yaxis=dict(gridcolor="#1e2433", showgrid=True, title="WQI"),
                legend=dict(
                    font=dict(size=9),
                    bgcolor="#0f1117",
                    bordercolor="#1e2433",
                    borderwidth=1,
                    x=0, y=1,
                ),
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available for this location.")
    else:
        st.markdown("""
        <div style="color:#444; font-size:0.85rem; padding:20px 0; text-align:center;
                    border:1px dashed #2a3040; border-radius:8px">
            Click any marker<br>on the map to see<br>location details
        </div>
        """, unsafe_allow_html=True)


with st.expander("📋 All Predictions Table"):
    display_df = pred_df[["Block", "Location", "WQI", "Category", "Confidence", "RMSE"]].copy()
    display_df = display_df.sort_values("WQI", ascending=False).reset_index(drop=True)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=300,
        column_config={
            "WQI"        : st.column_config.NumberColumn("WQI", format="%.1f"),
            "RMSE"       : st.column_config.NumberColumn("RMSE (CV)", format="%.1f"),
            "Category"   : st.column_config.TextColumn("Category"),
            "Confidence" : st.column_config.TextColumn("Confidence"),
        }
    )

st.markdown(
    '<p style="color:#333; font-size:0.7rem; text-align:center; margin-top:24px">'
    'WQI Forecasting Research · VIT · Salem District Groundwater Monitoring'
    '</p>',
    unsafe_allow_html=True
)