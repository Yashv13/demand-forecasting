import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load files
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.title("📈 Walmart Sales Forecasting App")

st.write("Enter the values for a single prediction:")

# Input fields
store = st.selectbox("Store", list(range(1, 46)))
dept = st.selectbox("Department", list(range(1, 100)))
temperature = st.number_input("Temperature", value=60.0)
fuel_price = st.number_input("Fuel Price", value=3.0)
cpi = st.number_input("CPI", value=200.0)
unemployment = st.number_input("Unemployment", value=7.0)
is_holiday = st.selectbox("Is Holiday", [False, True])
store_type = st.selectbox("Store Type", ['A', 'B', 'C'])

# DataFrame for prediction
input_df = pd.DataFrame([{
    "Store": store,
    "Dept": dept,
    "Temperature": temperature,
    "Fuel_Price": fuel_price,
    "CPI": cpi,
    "Unemployment": unemployment,
    "IsHoliday_y": is_holiday,
    "Type": store_type
}])

cat_cols = encoder.feature_names_in_

# Ensure all expected categorical columns are present
for col in cat_cols:
    if col not in input_df.columns:
        input_df[col] = np.nan  # or a default fallback like 'A' or False

# Align column order before encoding
input_df_cat = input_df[cat_cols]
encoded = encoder.transform(input_df_cat)
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=input_df.index)

# Replace original categorical columns
input_df = input_df.drop(columns=cat_cols).join(encoded_df)

# Align columns
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_cols]

# Scale
with open("feature_columns.pkl", "rb") as f:
    expected_columns = joblib.load(f)

# Add missing columns with 0s
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Drop extra columns not in training
input_df = input_df[expected_columns]

# Now scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Weekly Sales"):
    pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Weekly Sales: ${pred:,.2f}")
