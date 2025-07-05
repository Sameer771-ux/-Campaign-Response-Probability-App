import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature list
model = joblib.load("xgboost_campaign_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Campaign Response Predictor", page_icon="📩")
st.title("🧠 Campaign Response Probability App")

st.markdown("Enter customer details below to predict campaign engagement probability:")

# Create input fields for each feature
input_data = {}
for feature in features:
    if "Mnt" in feature or "Income" in feature:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=50.0)
    else:
        input_data[feature] = st.slider(f"{feature}", min_value=0, max_value=20, value=5)

# Convert to dataframe for prediction
input_df = pd.DataFrame([input_data])

# Predict on button click
if st.button("🔍 Predict Response Probability"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"🎯 Response Probability: **{probability:.2%}**")
    
    if probability > 0.6:
        st.balloons()
        st.info("🔥 High likelihood of engagement. Prioritize this customer!")
    else:
        st.warning("📭 Lower probability—consider re-engagement strategies.")
