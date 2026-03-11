import streamlit as st
import numpy as np
from src.prediction import PetrolPricePredictor

st.title("⛽ Petrol Price Prediction (India ₹)")

st.write("Predict petrol price based on global factors")

# User Inputs
before_price = st.number_input("Before War Price (USD)", min_value=0.0)
change = st.number_input("Amount Change (USD)", min_value=0.0)
oil_import = st.number_input("Oil Import Dependency (%)", min_value=0.0)

region = st.selectbox(
    "Select Region",
    ["Africa","Asia","Europe","MiddleEast","NorthAmerica","SouthAmerica"]
)

# Convert Region to Dummy Variables
region_africa = 1 if region == "Africa" else 0
region_asia = 1 if region == "Asia" else 0
region_europe = 1 if region == "Europe" else 0
region_middleeast = 1 if region == "MiddleEast" else 0
region_northamerica = 1 if region == "NorthAmerica" else 0
region_southamerica = 1 if region == "SouthAmerica" else 0

# Prediction
if st.button("Predict Petrol Price"):

    model = PetrolPricePredictor()

    features = [
        before_price,
        change,
        oil_import,
        region_africa,
        region_asia,
        region_europe,
        region_middleeast,
        region_northamerica,
        region_southamerica
    ]

    features = np.array(features).reshape(1, -1)

    # Model prediction in USD
    result_usd = model.predict(features)[0]

    # Convert to INR
    usd_to_inr = 91
    result_inr = result_usd * usd_to_inr

    # Display only INR
    st.success(f"Predicted Petrol Price in India: ₹ {result_inr:.2f}")

