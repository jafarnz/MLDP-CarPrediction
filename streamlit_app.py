from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path("car_price_model.pkl")
DATA_PATH = Path("SG_usedcar.csv")


st.set_page_config(page_title="SG Car Price Predictor", page_icon="🚗", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 20%, rgba(199, 227, 255, 0.45), transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(255, 221, 184, 0.42), transparent 35%),
            linear-gradient(140deg, #f6fbff 0%, #fdfaf4 100%);
    }
    .block-container {padding-top: 1.8rem; max-width: 900px;}
    .card {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(20, 26, 40, 0.08);
        margin-bottom: 0.9rem;
    }
    .fine {color: #425466; font-size: 0.93rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Singapore Used Car Price Predictor")
st.caption("Consumer-facing estimate using stable vehicle attributes. Market policy fields (e.g., COE) are intentionally excluded from core model.")


@st.cache_data
def load_options():
    df = pd.read_csv(DATA_PATH)
    df = df.replace(["N.A", "N.A.", "NA", "na", "-", ""], np.nan)
    if "Unnamed: 18" in df.columns:
        df = df[df["Unnamed: 18"].isna()].copy()
    valid_transmission = ["Auto", "Manual", "Petrol-Electric", "Electric"]
    df["Transmission"] = df["Transmission"].where(df["Transmission"].isin(valid_transmission), np.nan)
    return {
        "types": sorted(df["Type"].dropna().unique().tolist()),
        "transmissions": sorted(df["Transmission"].dropna().unique().tolist()),
    }


@st.cache_resource
def load_model(_model_mtime):
    return joblib.load(MODEL_PATH)


try:
    model_mtime = MODEL_PATH.stat().st_mtime
    model = load_model(model_mtime)
except FileNotFoundError:
    st.error("Model file not found. Run section 4/5 in `JAFAR.ipynb` to export `car_price_model.pkl` first.")
    st.stop()

options = load_options()
current_year = pd.Timestamp("today").year

type_segment_labels = {
    "PremiummPassneger (Sports car suv luxury)": "Premium_Passenger",
    "Mainstream Passenger (Sedan hatchback mpv wagon)": "Mainstream_Passenger",
    "Commercial (Van truck bus/minibus)": "Commercial",
}

premium_brand_label = (
    "Premium (Mercedes-Benz, BMW, Audi, Lexus, Porsche, Jaguar, Land, Maserati, "
    "Bentley, Ferrari, Lamborghini, Rolls-Royce, Aston, McLaren, Infiniti)"
)

st.markdown('<div class="card">', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    brand_segment = st.selectbox("Brand Segment", [premium_brand_label, "Rest Non-Premium"], index=1)
    type_segment_label = st.selectbox("Type Segment", list(type_segment_labels.keys()), index=1)
    transmission = st.selectbox("Transmission", options["transmissions"], index=0)
    owners = st.slider("Number of Owners", min_value=1, max_value=6, value=2, step=1)
with col_b:
    manufactured = st.slider("Manufactured Year", min_value=1995, max_value=current_year, value=2016, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=800000, value=70000, step=500)
    engine_cap = st.number_input("Engine Capacity (cc)", min_value=600, max_value=7000, value=1600, step=50)
    power = st.number_input("Power (hp)", min_value=40, max_value=1000, value=110, step=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
col_c, col_d = st.columns(2)
with col_c:
    road_tax = st.number_input("Road Tax (SGD/year)", min_value=50, max_value=15000, value=900, step=10)
with col_d:
    curb_weight = st.number_input("Curb Weight (kg)", min_value=700, max_value=25000, value=1400, step=50)
st.markdown('<p class="fine">Future enhancement: add live COE/API inputs and compare baseline estimate vs market-adjusted estimate.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


if st.button("Predict Car Price", type="primary"):
    vehicle_age = max(0.0, current_year - manufactured)
    luxury_brand = 1 if brand_segment == premium_brand_label else 0
    type_group = type_segment_labels[type_segment_label]

    if mileage <= 0:
        st.error("Mileage must be greater than 0.")
        st.stop()
    if power <= 0 or engine_cap <= 0:
        st.error("Engine capacity and power must be greater than 0.")
        st.stop()

    # Build model-aligned row directly to avoid one-row OHE pitfalls.
    row = {col: 0 for col in model.feature_names_in_}

    numeric_inputs = {
        "Mileage": float(mileage),
        "Road Tax": float(road_tax),
        "Engine Cap": float(engine_cap),
        "Curb Weight": float(curb_weight),
        "Manufactured": float(manufactured),
        "Power": float(power),
        "No. of Owners": float(owners),
        "Vehicle_Age": float(vehicle_age),
        "Luxury_Brand": int(luxury_brand),
    }
    for key, value in numeric_inputs.items():
        if key in row:
            row[key] = value

    type_col = f"Type_Group_{type_group}"
    transmission_col = f"Transmission_{transmission}"
    if type_col in row:
        row[type_col] = 1
    if transmission_col in row:
        row[transmission_col] = 1

    df_input = pd.DataFrame([row])

    y_pred = float(model.predict(df_input)[0])
    st.success(f"Predicted Used Car Price: SGD ${y_pred:,.0f}")
