import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Page setup
st.set_page_config(page_title="Demand Forecasting", layout="centered")

# Load model artifacts
model = joblib.load("model/xgb_model.pkl")
encoder = joblib.load("model/encoder.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

st.title("🛒 Walmart Demand Forecasting")
st.markdown("Predict weekly sales based on store, department, and other features.")

# Input form
with st.form("input_form"):
    Store = st.number_input("Store ID", min_value=1, step=1)
    Dept = st.number_input("Department ID", min_value=1, step=1)
    Temperature = st.number_input("Temperature (°F)", value=75.0)
    Fuel_Price = st.number_input("Fuel Price ($)", value=3.0)
    MarkDown1 = st.number_input("MarkDown1", value=0.0)
    MarkDown2 = st.number_input("MarkDown2", value=0.0)
    MarkDown3 = st.number_input("MarkDown3", value=0.0)
    MarkDown4 = st.number_input("MarkDown4", value=0.0)
    MarkDown5 = st.number_input("MarkDown5", value=0.0)
    CPI = st.number_input("CPI", value=220.0)
    Unemployment = st.number_input("Unemployment Rate (%)", value=7.0)
    IsHoliday_y = st.selectbox("Is Holiday", [False, True])
    Type = st.selectbox("Store Type", ["A", "B", "C"])
    Size = st.number_input("Store Size", value=150000)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Step 1: Create DataFrame
    input_data = {
        "Store": [Store],
        "Dept": [Dept],
        "Temperature": [Temperature],
        "Fuel_Price": [Fuel_Price],
        "MarkDown1": [MarkDown1],
        "MarkDown2": [MarkDown2],
        "MarkDown3": [MarkDown3],
        "MarkDown4": [MarkDown4],
        "MarkDown5": [MarkDown5],
        "CPI": [CPI],
        "Unemployment": [Unemployment],
        "IsHoliday_y": [IsHoliday_y],
        "Type": [Type],
        "Size": [Size],
    }

    input_df = pd.DataFrame(input_data)

    # Step 2: Encode categorical features
    cat_cols = ["IsHoliday_y", "Type"]
    encoded = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded, 
        columns=encoder.get_feature_names_out(cat_cols), 
        index=input_df.index
    )
    input_df = input_df.drop(columns=cat_cols).join(encoded_df)

    # Step 3: Align columns with training
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]  # Ensure correct order

    # Step 4: Scale input
    input_scaled = scaler.transform(input_df)

    # Step 5: Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Weekly Sales: **${prediction:,.2f}**")
