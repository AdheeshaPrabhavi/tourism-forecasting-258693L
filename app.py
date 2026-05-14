# app.py
# Tourism Demand Forecasting — Streamlit Web Application
# Student ID: 258693L

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import calendar
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Tourism Forecaster",
    page_icon="🌴🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F3864;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #595959;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F0F4FA;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #2E75B6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F3864;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #595959;
    }
    .alert-high {
        background: #FFE0E0;
        border-left: 5px solid #CC0000;
        padding: 1rem;
        border-radius: 8px;
    }
    .alert-medium {
        background: #FFF0D9;
        border-left: 5px solid #FF8C00;
        padding: 1rem;
        border-radius: 8px;
    }
    .alert-low {
        background: #FFFBD0;
        border-left: 5px solid #FFD700;
        padding: 1rem;
        border-radius: 8px;
    }
    .alert-normal {
        background: #E0F4E0;
        border-left: 5px solid #228B22;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL AND DATA ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/rf_pipeline.pkl')

@st.cache_data
def load_history():
    df = pd.read_csv('dashboard_data/history.csv',
                     parse_dates=['date'])
    return df

@st.cache_data
def load_forecast():
    return pd.read_csv('dashboard_data/forecast.csv')

@st.cache_data
def load_alert():
    return pd.read_csv('dashboard_data/early_warning.csv')

@st.cache_data
def load_trends():
    df = pd.read_csv('dashboard_data/trends_signal.csv',
                     parse_dates=['date'])
    return df

@st.cache_data
def load_performance():
    return pd.read_csv('dashboard_data/model_performance.csv')

rf_model    = load_model()
df_history  = load_history()
df_forecast = load_forecast()
df_alert    = load_alert()
df_trends   = load_trends()
df_perf     = load_performance()

# ── SEASONAL AVERAGES (normal months only) ────────────────────────────
SEASONAL_AVG = {
    1: 242203, 2: 239487, 3: 188544, 4: 142991,
    5: 109452, 6: 117366, 7: 161698, 8: 160709,
    9: 125406, 10: 132258, 11: 181386, 12: 239884
}

FEATURE_COLS = [
    'exchange_rate', 'inflation', 'google_trends', 'disruption_score',
    'arrivals_lag1', 'arrivals_lag2', 'arrivals_lag3', 'arrivals_lag12',
    'exchange_rate_lag1', 'exchange_rate_lag2',
    'inflation_lag1', 'inflation_lag2',
    'trends_lag1', 'trends_lag2', 'disruption_lag1',
    'month_sin', 'month_cos', 'is_peak_season', 'quarter'
]


# ── SIDEBAR ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/"
             "thumb/1/11/Flag_of_Sri_Lanka.svg/320px-Flag_of_Sri_Lanka.svg.png",
             width=80)
    st.markdown("##  Tourism Forecaster")
    st.markdown("**Sri Lanka — Real-Time Demand Prediction**")
    st.divider()

    st.markdown("### Manual Prediction")
    st.markdown("Enter current conditions to generate a forecast:")

    prev_arrivals    = st.number_input(
        "Previous month's official arrivals (SLTDA)",
        min_value=0, max_value=500000,
        value=135643, step=1000)

    google_trends_val = st.slider(
        "Google Trends index (0-100)",
        min_value=0.0, max_value=100.0,
        value=47.5, step=0.5)

    exchange_rate_val = st.number_input(
        "USD/LKR exchange rate",
        min_value=100.0, max_value=500.0,
        value=321.08, step=0.5)

    inflation_val = st.number_input(
        "Inflation rate (CCPI %)",
        min_value=0.0, max_value=100.0,
        value=5.4, step=0.1)

    disruption_val = st.slider(
        "Disruption severity score (0-10)",
        min_value=0.0, max_value=10.0,
        value=1.0, step=0.5,
        help="0=normal, 5=moderate disruption, 10=severe")

    predict_button = st.button("Generate Forecast", type="primary",
                               use_container_width=True)
    st.divider()
    st.markdown("**Model:** Random Forest (tuned)")
    st.markdown("**Test RMSE:** 40,166 arrivals")
    st.markdown("**Test MAPE:** 16.11%")
    st.markdown("**Data:** Jan 2018 – Mar 2026")


# ── HEADER ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🌴 Sri Lanka Tourism Demand Forecasting</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Machine Learning-Driven Early Warning System | '
    'Student ID: 258693L</div>',
    unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔮 Live Forecast",
    "📈 Historical Analysis",
    "🚦 Early Warning",
    "⚙️ Model Performance"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Current Month Dashboard")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)

    forecast_val  = df_forecast['predicted_rf'].iloc[0]
    ci_lo         = df_forecast['ci_lower_90'].iloc[0]
    ci_hi         = df_forecast['ci_upper_90'].iloc[0]
    alert_level   = df_alert['alert_level'].iloc[0]
    pct_dev       = df_alert['pct_vs_seasonal'].iloc[0]
    season_exp    = df_alert['seasonal_expected'].iloc[0]
    disrupt_score = df_alert['disruption_score'].iloc[0]
    forecast_month= df_forecast['forecast_month'].iloc[0]

    alert_colors = {
        'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟡', 'NORMAL': '🟢'
    }

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Forecast Month</div>
            <div class="metric-value">{forecast_month}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Arrivals</div>
            <div class="metric-value">{forecast_val:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">vs Seasonal Expected</div>
            <div class="metric-value">{pct_dev:+.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Alert Level</div>
            <div class="metric-value">
                {alert_colors.get(alert_level,'🟢')} {alert_level}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Confidence interval bar
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=[ci_lo, ci_hi],
            y=["90% CI", "90% CI"],
            mode='lines',
            line=dict(color='#2E75B6', width=8),
            name='90% Confidence Interval'
        ))
        fig_ci.add_trace(go.Scatter(
            x=[forecast_val],
            y=["90% CI"],
            mode='markers',
            marker=dict(color='#1F3864', size=18, symbol='diamond'),
            name=f'Prediction: {forecast_val:,.0f}'
        ))
        fig_ci.update_layout(
            title=f"Prediction with 90% Confidence Interval — {forecast_month}",
            xaxis_title="Monthly Arrivals",
            height=180,
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_ci, use_container_width=True)

    with col_b:
        st.markdown("**Forecast Summary**")
        st.markdown(f"- **Point estimate:** {forecast_val:,.0f}")
        st.markdown(f"- **90% CI lower:** {ci_lo:,.0f}")
        st.markdown(f"- **90% CI upper:** {ci_hi:,.0f}")
        st.markdown(f"- **Seasonal expected:** {season_exp:,.0f}")
        st.markdown(f"- **Disruption score:** {disrupt_score}")

    st.divider()

    # Recent 12-month chart
    st.subheader("Recent 12-Month Trend")
    df_recent = df_history.tail(12)
    fig_recent = go.Figure()
    fig_recent.add_trace(go.Scatter(
        x=df_recent['date'], y=df_recent['actual'],
        mode='lines+markers', name='Actual arrivals',
        line=dict(color='#2E75B6', width=2),
        marker=dict(size=6)
    ))
    fig_recent.add_trace(go.Scatter(
        x=df_recent['date'], y=df_recent['predicted_rf'],
        mode='lines+markers', name='RF predicted',
        line=dict(color='#FF6B35', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_recent.update_layout(
        xaxis_title="Month",
        yaxis_title="Monthly Arrivals",
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_recent, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE FORECAST
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Generate a Custom Prediction")
    st.markdown("Adjust the inputs in the sidebar and click **Generate Forecast**.")

    if predict_button:
        with st.spinner("Running Random Forest prediction..."):

            # Build feature row from sidebar inputs
            # Use last known values from history for lag features
            df_master = pd.read_csv('data/data.csv',
                                    parse_dates=['date'],
                                    index_col='date')
            df_master = df_master.sort_index()

            # Rebuild lags
            for lag in [1, 2, 3, 12]:
                df_master[f'arrivals_lag{lag}'] = \
                    df_master['arrivals'].shift(lag)
            for lag in [1, 2]:
                df_master[f'exchange_rate_lag{lag}'] = \
                    df_master['exchange_rate'].shift(lag)
                df_master[f'inflation_lag{lag}'] = \
                    df_master['inflation'].shift(lag)
                df_master[f'trends_lag{lag}'] = \
                    df_master['google_trends'].shift(lag)
            df_master['disruption_lag1'] = \
                df_master['disruption_score'].shift(1)
            df_master['month_num']       = df_master.index.month
            df_master['quarter']         = df_master.index.quarter
            df_master['month_sin']       = np.sin(
                2 * np.pi * df_master['month_num'] / 12)
            df_master['month_cos']       = np.cos(
                2 * np.pi * df_master['month_num'] / 12)
            df_master['is_peak_season']  = \
                df_master['month_num'].isin([12,1,2,3]).astype(int)

            df_master = df_master.dropna()
            last_row  = df_master[FEATURE_COLS].iloc[-1].copy()

            # Override with user inputs
            last_row['arrivals_lag1']   = prev_arrivals
            last_row['google_trends']   = google_trends_val
            last_row['trends_lag1']     = df_master['google_trends'].iloc[-1]
            last_row['exchange_rate']   = exchange_rate_val
            last_row['inflation']       = inflation_val
            last_row['disruption_score']= disruption_val

            X_input = last_row.values.reshape(1, -1)
            prediction = rf_model.predict(X_input)[0]

            # Bootstrap CI from individual trees
            tree_preds = np.array([
                tree.predict(
                    rf_model['scaler'].transform(X_input)
                )[0]
                for tree in rf_model['model'].estimators_
            ])
            ci_lo_90 = np.percentile(tree_preds, 5)
            ci_hi_90 = np.percentile(tree_preds, 95)

            # Alert logic
            next_month_num = (date.today().month % 12) + 1
            seasonal_exp   = SEASONAL_AVG.get(next_month_num, prediction)
            pct_dev_live   = (prediction - seasonal_exp) / seasonal_exp * 100

            if pct_dev_live < -40:
                alert_live = 'HIGH';   alert_css = 'alert-high'
            elif pct_dev_live < -25:
                alert_live = 'MEDIUM'; alert_css = 'alert-medium'
            elif pct_dev_live < -10:
                alert_live = 'LOW';    alert_css = 'alert-low'
            else:
                alert_live = 'NORMAL'; alert_css = 'alert-normal'

        # Results
        st.success("Prediction complete!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Arrivals", f"{prediction:,.0f}")
        with col2:
            st.metric("90% CI Range",
                      f"{ci_lo_90:,.0f} – {ci_hi_90:,.0f}")
        with col3:
            st.metric("vs Seasonal Expected",
                      f"{pct_dev_live:+.1f}%",
                      delta=f"{alert_live} alert")

        st.markdown(f"""
        <div class="{alert_css}">
            <strong>🚦 Alert Level: {alert_live}</strong><br>
            Predicted arrivals of {prediction:,.0f} are {pct_dev_live:+.1f}%
            versus the seasonal average of {seasonal_exp:,.0f} for this month.
        </div>""", unsafe_allow_html=True)

        # Gauge chart for disruption score
        st.markdown("")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=disruption_val,
            title={'text': "Disruption Severity Score"},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "#1F3864"},
                'steps': [
                    {'range': [0, 3],   'color': '#E0F4E0'},
                    {'range': [3, 6],   'color': '#FFF0D9'},
                    {'range': [6, 10],  'color': '#FFE0E0'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': disruption_val
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.info("👈 Adjust inputs in the sidebar and click "
                "**Generate Forecast** to run a prediction.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Full Historical Series — Actual vs Predicted")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_history['date'], y=df_history['actual'],
        mode='lines', name='Actual arrivals',
        line=dict(color='#2E75B6', width=2)
    ))
    fig_hist.add_trace(go.Scatter(
        x=df_history['date'], y=df_history['predicted_rf'],
        mode='lines', name='RF predicted',
        line=dict(color='#FF6B35', width=2, dash='dash')
    ))

    # Shock annotations
    shocks = [
    ('2019-04-01', 'Easter attacks'),
    ('2020-04-01', 'COVID-19 peak'),
    ('2022-04-01', 'Economic crisis'),
    ]
    for shock_date, shock_label in shocks:
      fig_hist.add_vline(
        x=pd.Timestamp(shock_date).timestamp() * 1000,
        line_dash='dot',
        line_color='red', line_width=1.5,
        annotation_text=shock_label,
        annotation_position='top'
    )


    fig_hist.update_layout(
        xaxis_title="Month",
        yaxis_title="Monthly Arrivals",
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Two sub-columns
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Google Trends — Leading Indicator")
        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(
            x=df_trends['date'], y=df_trends['google_trends'],
            mode='lines+markers', name='Google Trends',
            line=dict(color='#2E75B6', width=2),
            fill='tozeroy', fillcolor='rgba(46,117,182,0.1)'
        ))
        avg_trends = df_trends['google_trends'].mean()
        fig_trends.add_hline(
            y=avg_trends, line_dash='dash',
            line_color='gray',
            annotation_text=f"12-month avg: {avg_trends:.1f}"
        )
        fig_trends.update_layout(
            xaxis_title="Month",
            yaxis_title="Composite Trends Index",
            height=300
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        st.caption("Keywords: 'sri lanka hotels' + 'visit sri lanka' (averaged). "
                   "Trends index typically leads actual arrivals by 1-2 months.")

    with col_b:
        st.subheader("Prediction Error Over Time")
        df_history['error_pct'] = (
            (df_history['actual'] - df_history['predicted_rf'])
            / df_history['actual'].replace(0, np.nan) * 100
        )
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=df_history['date'],
            y=df_history['error_pct'],
            marker_color=df_history['error_pct'].apply(
                lambda x: '#CC0000' if x < -20
                else '#FF8C00' if x < 0
                else '#228B22'),
            name='Error %'
        ))
        fig_err.add_hline(y=0, line_color='black', line_width=1)
        fig_err.update_layout(
            xaxis_title="Month",
            yaxis_title="Prediction Error (%)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_err, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — EARLY WARNING
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Early Warning System")

    col1, col2 = st.columns([1, 2])

    with col1:
        alert_level_curr = df_alert['alert_level'].iloc[0]
        pct_curr         = df_alert['pct_vs_seasonal'].iloc[0]
        pred_curr        = df_alert['predicted'].iloc[0]
        seas_curr        = df_alert['seasonal_expected'].iloc[0]
        dis_curr         = df_alert['disruption_score'].iloc[0]

        alert_css_map = {
            'HIGH':   'alert-high',
            'MEDIUM': 'alert-medium',
            'LOW':    'alert-low',
            'NORMAL': 'alert-normal'
        }
        css = alert_css_map.get(alert_level_curr, 'alert-normal')

        st.markdown(f"""
        <div class="{css}">
            <h3>🚦 {alert_level_curr}</h3>
            <p>Predicted arrivals are <strong>{pct_curr:+.1f}%</strong>
            versus the seasonal average.</p>
            <hr>
            <p>Predicted: <strong>{pred_curr:,.0f}</strong></p>
            <p>Seasonal expected: <strong>{seas_curr:,.0f}</strong></p>
            <p>Disruption score: <strong>{dis_curr}</strong></p>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Alert Thresholds:**")
        st.markdown("🔴 HIGH: more than -40% below seasonal")
        st.markdown("🟠 MEDIUM: -25% to -40% below seasonal")
        st.markdown("🟡 LOW: -10% to -25% below seasonal")
        st.markdown("🟢 NORMAL: within -10% of seasonal")

    with col2:
        # Seasonal average bar chart
        months     = list(SEASONAL_AVG.keys())
        month_names= [calendar.month_abbr[m] for m in months]
        avgs       = list(SEASONAL_AVG.values())
        curr_month = date.today().month

        colors = ['#FF6B35' if m == curr_month else '#2E75B6'
                  for m in months]

        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Bar(
            x=month_names, y=avgs,
            marker_color=colors,
            name='Seasonal average'
        ))
        fig_seasonal.add_hline(
            y=pred_curr, line_dash='dash',
            line_color='#228B22', line_width=2,
            annotation_text=f"Current prediction: {pred_curr:,.0f}"
        )
        fig_seasonal.update_layout(
            title="Seasonal Baseline vs Current Prediction",
            xaxis_title="Month",
            yaxis_title="Average Monthly Arrivals",
            height=380
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

    st.divider()
    st.subheader("Historical Disruption Score Timeline")

    fig_disrupt = go.Figure()
    fig_disrupt.add_trace(go.Bar(
        x=df_history['date'],
        y=df_history['disruption_score'],
        marker_color=df_history['disruption_score'].apply(
            lambda x: '#CC0000' if x >= 7
            else '#FF8C00' if x >= 4
            else '#FFD700' if x >= 1
            else '#228B22'),
        name='Disruption score'
    ))
    fig_disrupt.update_layout(
        xaxis_title="Month",
        yaxis_title="Disruption Severity (0-10)",
        height=280,
        showlegend=False
    )
    st.plotly_chart(fig_disrupt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Model Performance — Test Set Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test MAE",  "33,177 arrivals")
        st.metric("Test RMSE", "40,166 arrivals")
    with col2:
        st.metric("Test MAPE", "16.11%")
        st.metric("Test R²",   "0.2936")
    with col3:
        st.metric("Train/Test RMSE Ratio", "3.15")
        st.metric("Test Period",
                  "Jan 2024 – Mar 2026")

    st.divider()

    # Model comparison table
    st.subheader("Full Model Comparison")
    comparison = pd.DataFrame({
        'Model': ['OLS Linear Regression', 'Ridge Regression',
                  'RF (default)', 'RF (tuned)', 'LSTM'],
        'Test MAE':  [37229, 39228, 32784, 33177, 82432],
        'Test RMSE': [48254, 47944, 37148, 40166, 89510],
        'Test R²':   [-0.0195, -0.0065, 0.3958, 0.2936, -2.5081],
        'Test MAPE': ['19.98%', '21.46%', '18.08%', '16.11%', '41.64%'],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.divider()

    # SHAP feature importance
    st.subheader("SHAP Feature Importance (Test Set)")
    shap_data = {
        'Feature': ['google_trends', 'arrivals_lag1', 'trends_lag1',
                    'disruption_score', 'disruption_lag1', 'arrivals_lag3',
                    'arrivals_lag12', 'month_cos', 'arrivals_lag2',
                    'trends_lag2'],
        'Mean |SHAP|': [25277, 18652, 10080, 9388, 8407,
                        6812, 4089, 3159, 2331, 2258]
    }
    df_shap = pd.DataFrame(shap_data)

    fig_shap = go.Figure(go.Bar(
        x=df_shap['Mean |SHAP|'],
        y=df_shap['Feature'],
        orientation='h',
        marker_color='#2E75B6'
    ))
    fig_shap.update_layout(
        xaxis_title="Mean |SHAP value|",
        yaxis=dict(autorange='reversed'),
        height=380
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.caption(
        "SHAP values computed using TreeExplainer on 28 test-set predictions. "
        "google_trends is the dominant predictor, confirming that digital "
        "search intent is the primary driver of Sri Lanka's tourism demand "
        "in the post-shock recovery period."
    )
