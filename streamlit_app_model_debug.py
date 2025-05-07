
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Phishing Detection", layout="centered")
st.title("🔐 Phishing Website Detection")

st.write("Use the form below to test whether a website is phishing or legitimate based on AI prediction.")

model_choice = st.selectbox(
    "Select Model",
    ["Best Model", "XGBoost", "Random Forest", "Logistic Regression"]
)

model_path = {
    "Best Model": "models/phishing_model_best.pkl",
    "XGBoost": "models/phishing_model_xgb.pkl",
    "Random Forest": "models/phishing_model_rf.pkl",
    "Logistic Regression": "models/phishing_model_lr.pkl"
}[model_choice]

model = joblib.load(model_path)
features = list(model.feature_names_in_)
label_map = {-1: "Suspicious / Malicious", 0: "Neutral / Uncertain", 1: "Legitimate / Safe"}
input_data = {}

with st.form("phishing_form", clear_on_submit=False):
    st.subheader("🔢 Feature Inputs")
    for feature in features:
        input_data[feature] = st.radio(
            feature,
            options=[-1, 0, 1],
            format_func=lambda x: label_map[x],
            key=feature
        )
    submitted = st.form_submit_button("🔎 Predict")

# Run prediction only if form was submitted
if submitted:
    input_df = pd.DataFrame([input_data])[features]

    st.subheader("🧪 Debug Info")
    st.write("✅ Input DataFrame:")
    st.dataframe(input_df)
    st.write("🧬 Data Types:")
    st.write(input_df.dtypes)
    st.write("📋 Model Expected Features:")
    st.write(model.feature_names_in_)

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success("✅ Prediction: **Phishing**")
    else:
        st.success("✅ Prediction: **Legitimate**")

    st.write(f"🔍 Confidence Score (Phishing Probability): **{prediction_proba[0][1]:.2f}**")
