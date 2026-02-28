import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = joblib.load(os.path.join(BASE_DIR,"all_models.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR,"scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR,"columns.pkl"))

st.title("🌧 Rainfall Prediction App")
st.write("Enter weather values")

inputs = []

for col in columns:
    val = st.number_input(col, value=0.0)
    inputs.append(val)

if st.button("Predict"):

    df = pd.DataFrame([inputs], columns=columns)

    results = {}

    for name, model in models.items():

        temp_df = df.copy()

        if name in ["Logistic","KNN"]:
            temp_df = scaler.transform(temp_df)

        pred = model.predict(temp_df)[0]
        results[name] = "Yes" if pred==1 else "No"

    st.subheader("Predictions")

    for model, result in results.items():
        st.write(f"{model} → {result}")