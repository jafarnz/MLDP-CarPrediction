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
    .block-container {padding-top: 1.5rem; max-width: 900px;}
    h1, h2, h3, h4, h5, h6 {color: #1f2937 !important;}
    div[data-testid="stCaptionContainer"] p {color: #475569 !important;}
    div[data-testid="stWidgetLabel"] p {color: #1f2937 !important; font-weight: 600 !important;}
    p, li {color: #334155;}
    .card {
        background: rgba(255, 255, 255, 0.93);
        border: 1px solid rgba(17, 24, 39, 0.08);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 30px rgba(20, 26, 40, 0.08);
        margin-bottom: 0.9rem;
    }
    .fine {color: #475569; font-size: 0.92rem;}
    div[data-testid="stAlert"] p {color: #0f172a !important;}
    @media (max-width: 640px) {
        .block-container {
            padding-top: 0.8rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
        h1 {
            font-size: 1.95rem !important;
            line-height: 1.2 !important;
        }
        .card {
            border-radius: 14px;
            padding: 0.85rem 0.8rem;
        }
        .fine {
            font-size: 0.84rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Singapore Used Car Price Predictor")
st.caption("Consumer-facing estimate using stable vehicle attributes. Market policy fields (e.g., COE) are intentionally excluded from core model.")


def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run section 4/5 in `JAFAR.ipynb` to export `car_price_model.pkl` first.")
    st.stop()

current_year = pd.Timestamp("today").year

type_segment_labels = {
    "Premium Passenger": 1,
    "Mainstream / Commercial": 0,
}
type_group_map = {
    "Sports Car": 1,
    "SUV": 1,
    "Luxury Sedan": 1,
    "Mid-Sized Sedan": 0,
    "Hatchback": 0,
    "MPV": 0,
    "Stationwagon": 0,
    "Others": 0,
    "Van": 0,
    "Truck": 0,
    "Bus/Mini Bus": 0,
}

premium_brand_label = "Premium"
non_premium_brand_label = "Non-Premium"


@st.cache_data
def load_segment_priors():
    df = pd.read_csv(DATA_PATH)
    df = df.replace(["N.A", "N.A.", "NA", "na", "-", ""], np.nan)
    if "Unnamed: 18" in df.columns:
        df = df[df["Unnamed: 18"].isna()].copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"]).copy()
    if "Brand" in df.columns:
        df["Brand_Main"] = df["Brand"].astype(str).str.split().str[0]

    premium_brands = {
        "Mercedes-Benz", "BMW", "Audi", "Lexus", "Porsche", "Jaguar", "Land", "Maserati",
        "Bentley", "Ferrari", "Lamborghini", "Rolls-Royce", "Aston", "McLaren", "Infiniti"
    }
    df["Luxury_Brand"] = df["Brand_Main"].isin(premium_brands).astype(int)
    df["Type_Premium"] = df["Type"].map(type_group_map).fillna(0).astype(int)

    priors = (
        df.groupby(["Type_Premium", "Luxury_Brand"])["Price"]
        .median()
        .to_dict()
    )
    global_median = float(df["Price"].median())
    return priors, global_median


segment_priors, global_median_price = load_segment_priors()

st.markdown('<div class="card">', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    brand_segment = st.selectbox("Brand Segment", [premium_brand_label, non_premium_brand_label], index=1)
    st.caption("Premium brands: Mercedes-Benz, BMW, Audi, Lexus, Porsche, Jaguar, Land Rover, Maserati, Bentley, Ferrari, Lamborghini, Rolls-Royce, Aston Martin, McLaren, Infiniti.")
    type_segment_label = st.selectbox("Type Segment", list(type_segment_labels.keys()), index=1)
    st.caption("Type grouping: Premium Passenger = Sports Car, SUV, Luxury Sedan. Mainstream / Commercial = all other types.")
with col_b:
    vehicle_age = st.slider("Vehicle Age (years)", min_value=0, max_value=30, value=10, step=1)
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

#test
if st.button("Predict Car Price", type="primary"):
    vehicle_age = float(vehicle_age)
    manufactured = float(current_year - vehicle_age)
    luxury_brand = 1 if brand_segment == premium_brand_label else 0
    type_premium = int(type_segment_labels[type_segment_label])

    if mileage <= 0:
        st.error("Mileage must be greater than 0.")
        st.stop()
    if power <= 0 or engine_cap <= 0:
        st.error("Engine capacity and power must be greater than 0.")
        st.stop()

    def predict_with_segments(curr_type_premium, curr_luxury_brand):
        # Build model-aligned row directly to avoid one-row OHE pitfalls.
        row = {col: 0 for col in model.feature_names_in_}
        numeric_inputs = {
            "Mileage": float(mileage),
            "Road Tax": float(road_tax),
            "Engine Cap": float(engine_cap),
            "Curb Weight": float(curb_weight),
            "Manufactured": float(manufactured),
            "Power": float(power),
            "Vehicle_Age": float(vehicle_age),
            "Luxury_Brand": int(curr_luxury_brand),
        }
        for key, value in numeric_inputs.items():
            if key in row:
                row[key] = value

        if "Type_Group_Premium_Passenger" in row:
            row["Type_Group_Premium_Passenger"] = int(curr_type_premium)

        df_input = pd.DataFrame([row])
        y_pred_base = float(model.predict(df_input)[0])
        seg_prior = float(segment_priors.get((int(curr_type_premium), int(curr_luxury_brand)), global_median_price))
        # Stronger prior blend so segment differences are visible and stable in UI.
        y_pred_final = 0.6 * y_pred_base + 0.4 * seg_prior
        return y_pred_base, seg_prior, y_pred_final

    _, _, final_pred = predict_with_segments(type_premium, luxury_brand)

    st.success(f"Predicted Used Car Price: SGD ${final_pred:,.0f}")
