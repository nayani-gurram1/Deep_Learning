import streamlit as st
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# Load files
model = load_model("car_price_model.keras")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🚗 Car Price Prediction App")

engine_size = st.number_input("Engine Size")
horsepower = st.number_input("Horsepower")
curb_weight = st.number_input("Curb Weight")
highway_mpg = st.number_input("Highway MPG")

if st.button("Predict Price"):

    input_dict = {col:0 for col in feature_columns}

    input_dict["enginesize"] = engine_size
    input_dict["horsepower"] = horsepower
    input_dict["curbweight"] = curb_weight
    input_dict["highwaympg"] = highway_mpg

    input_df = pd.DataFrame([input_dict])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Car Price: ${prediction[0][0]:,.2f}")