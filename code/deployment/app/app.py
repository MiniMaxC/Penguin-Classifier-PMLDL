"""A deliberately small browser interface that calls the separate model API."""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")

st.set_page_config(page_title="Penguin Species Classifier", page_icon="🐧", layout="centered")
st.title("🐧 Penguin Species Classifier")
st.write("Enter four measurements to predict whether a penguin is Adelie, Chinstrap, or Gentoo.")

with st.form("prediction"):
    left, right = st.columns(2)
    with left:
        bill_length = st.number_input("Bill length (mm)", min_value=0.1, value=39.1, step=0.1)
        bill_depth = st.number_input("Bill depth (mm)", min_value=0.1, value=18.7, step=0.1)
    with right:
        flipper_length = st.number_input("Flipper length (mm)", min_value=0.1, value=181.0, step=1.0)
        body_mass = st.number_input("Body mass (g)", min_value=0.1, value=3750.0, step=25.0)
    submitted = st.form_submit_button("Predict species", use_container_width=True)

if submitted:
    payload = {"bill_length_mm": bill_length, "bill_depth_mm": bill_depth,
               "flipper_length_mm": flipper_length, "body_mass_g": body_mass}
    try:
        with st.spinner("Getting prediction…"):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        st.success(f"Predicted species: {result['species']}")
        st.caption(f"Model run: {result['run_id']}")
    except requests.exceptions.Timeout:
        st.error("The prediction request timed out. Please try again shortly.")
    except requests.exceptions.RequestException:
        st.error("The prediction service is temporarily unavailable. Please try again shortly.")
    except (ValueError, KeyError):
        st.error("The prediction service returned an unexpected response. Please try again shortly.")

st.caption("Measurements: millimetres and grams · Palmer Penguins dataset · Educational demonstration")
