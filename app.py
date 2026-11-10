
import streamlit as st
import pandas as pd
import joblib

# Load model
bundle = joblib.load("best_model.pkl")
model = bundle["model"]
features = bundle["features"]

st.set_page_config(page_title="Cancer Classifier", page_icon="🔬")

st.title("Breast Cancer Classification")
st.caption("Educational machine learning demonstration")

st.warning("This application is not intended for medical diagnosis.")

st.subheader("Enter feature values")

with st.form("prediction_form"):
    values = {}

    for feature in features:
        values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([values])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.subheader(
        "Prediction: " +
        ("Benign" if prediction == "B" else "Malignant")
    )

    st.write("Class probabilities:")

    for label, probability in zip(
        model.classes_, probabilities
    ):
        name = "Benign" if label == "B" else "Malignant"
        st.write(f"{name}: {probability:.2%}")
