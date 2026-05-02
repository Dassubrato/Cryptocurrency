import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crypto Predictor", layout="wide")

st.title("🚀 Crypto Price Prediction Dashboard")

# =========================
# LOAD MODELS
# =========================
xgb_model = joblib.load("xgb_model.pkl")
transfermodel = joblib.load("transfermodel.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Transformer", "LightGBM"]
)

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Manual Input", "Sample Data"]
)

# =========================
# INPUT SECTION
# =========================
st.subheader("📊 Input Data")

if input_mode == "Manual Input":
    col1, col2, col3 = st.columns(3)

    with col1:
        open_price = st.number_input("Open", value=100.0)
        high = st.number_input("High", value=105.0)

    with col2:
        low = st.number_input("Low", value=95.0)
        close = st.number_input("Close", value=102.0)

    with col3:
        volume = st.number_input("Volume", value=1000.0)
        taker_buy_base = st.number_input("Taker Buy Base", value=600.0)

else:
    # Sample realistic crypto data
    open_price = 100
    high = 103
    low = 98
    close = 101
    volume = 1200
    taker_buy_base = 700

    st.success("Using Sample Crypto Data")

# =========================
# FEATURE ENGINEERING
# =========================
sell_volume = volume - taker_buy_base
ofi = taker_buy_base - sell_volume
ofi_ratio = ofi / volume if volume != 0 else 0
taker_ratio = taker_buy_base / volume if volume != 0 else 0

price_range = high - low
close_open_diff = close - open_price

input_df = pd.DataFrame([{
    "lag1": 0,
    "lag2": 0,
    "lag5": 0,
    "volatility_10": 0,
    "volatility_20": 0,
    "momentum_5": 0,
    "momentum_10": 0,
    "ofi": ofi,
    "ofi_ratio": ofi_ratio,
    "taker_buy_ratio": taker_ratio,
    "trade_intensity": 0,
    "volume_change": 0,
    "price_range": price_range,
    "close_open_diff": close_open_diff,
    "vwap": close,
    "hour": 0,
    "day_of_week": 0,
    "high_vol_regime": 0
}])

# =========================
# MODEL SELECTION
# =========================
if model_choice == "XGBoost":
    model = xgb_model
elif model_choice == "Transformer":
    model = transfermodel
else:
    model = lgbm_model

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict"):
    pred = model.predict(input_df)[0]

    st.subheader("📈 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Log Return", f"{pred:.6f}")

    with col2:
        direction = "📈 UP" if pred > 0 else "📉 DOWN"
        st.metric("Direction", direction)

    # =========================
    # SIMPLE PLOT
    # =========================
    st.subheader("📉 Simulated Price Movement")

    simulated = np.random.randn(50).cumsum() + close

    plt.figure(figsize=(10,4))
    plt.plot(simulated)
    plt.axhline(close, linestyle="--")
    plt.title("Simulated Future Price Path")

    st.pyplot(plt)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with ML models: XGBoost, Transfomer, LightGBM")
