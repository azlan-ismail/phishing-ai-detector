
import streamlit as st
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import numpy as np

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

metrics_path = {
    "Best Model": "metrics/metrics_rf.json",
    "XGBoost": "metrics/metrics_xgb.json",
    "Random Forest": "metrics/metrics_rf.json",
    "Logistic Regression": "metrics/metrics_lr.json"
}[model_choice]

model = joblib.load(model_path)
features = list(model.feature_names_in_)

try:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    st.subheader("📊 Model Evaluation Metrics")
    st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']])
except Exception as e:
    st.warning(f"Could not load metrics: {e}")

label_map = {-1: "Suspicious / Malicious", 0: "Neutral / Uncertain", 1: "Legitimate / Safe"}

input_data = {}
with st.form("phishing_form"):
    st.subheader("🔢 Feature Inputs")
    for feature in features:
        input_data[feature] = st.radio(
            feature,
            options=[-1, 0, 1],
            format_func=lambda x: label_map[x],
            key=feature
        )
    submitted = st.form_submit_button("🔎 Predict")

if submitted:
    input_df = pd.DataFrame([input_data])[features]
    prediction = model.predict(input_df)[0]
    st.success("✅ Prediction: **Phishing**" if prediction == 1 else "✅ Prediction: **Legitimate**")

    with st.expander("🧠 Why this prediction?"):
        bg_data = pd.read_csv("data/phishing.csv")
        bg_data = bg_data.rename(columns={"Result": "Label"})
        bg_data["Label"] = bg_data["Label"].map({-1: 1, 1: 0})
        X_bg = bg_data[features]

        if model_choice in ["XGBoost", "Best Model", "Random Forest"]:
            explainer = shap.Explainer(model, X_bg, algorithm="tree")
        else:
            explainer = shap.Explainer(model, X_bg, algorithm="linear")

        shap_values = explainer(input_df)

        if hasattr(shap_values, "values"):
            val = shap_values.values
            if val.ndim == 3:
                shap_val_row = val[0, :, 1]
            else:
                shap_val_row = val[0]
        else:
            shap_val_row = shap_values[0].values[0]

        top_indices = np.argsort(np.abs(shap_val_row))[::-1][:3]
        top_features = [features[i] for i in top_indices]

        st.subheader("🔍 Feature Influence on Prediction")
        shap.plots.bar(shap.Explanation(values=shap_val_row, feature_names=features), show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        feature_expl_dict = {
            "having_IP_Address": "the URL uses an IP address",
            "URL_Length": "the URL is unusually long",
            "Shortining_Service": "a URL shortening service is used",
            "having_At_Symbol": "the URL contains an '@' symbol",
            "double_slash_redirecting": "the URL contains double slashes (//) beyond the protocol",
            "Prefix_Suffix": "a hyphen is used in the domain",
            "having_Sub_Domain": "the URL contains multiple subdomains",
            "SSLfinal_State": "the SSL certificate is invalid or missing",
            "Domain_registeration_length": "the domain has a short registration duration",
            "Favicon": "the favicon is loaded from an external domain",
            "port": "non-standard port is used in the URL",
            "HTTPS_token": "the word 'HTTPS' is used incorrectly in the URL",
            "Request_URL": "the webpage loads resources from different domains",
            "URL_of_Anchor": "anchor tags point to external or suspicious links",
            "Links_in_tags": "meta/script/link tags point to suspicious resources",
            "SFH": "the Server Form Handler is empty or abnormal",
            "Submitting_to_email": "the form submits to an email address",
            "Abnormal_URL": "the URL does not match whois records or has irregular structure",
            "Redirect": "the website has multiple redirections",
            "on_mouseover": "the page content changes unexpectedly on mouseover",
            "RightClick": "right-click functionality is disabled",
            "popUpWidnow": "pop-up windows are triggered",
            "Iframe": "the webpage uses invisible iframe tags",
            "age_of_domain": "the domain was registered recently",
            "DNSRecord": "no DNS record was found for the domain",
            "web_traffic": "the site receives low web traffic",
            "Page_Rank": "the site has a poor Google PageRank",
            "Google_Index": "the site is not indexed by Google",
            "Links_pointing_to_page": "few external links point to this page",
            "Statistical_report": "the domain or IP address appears in known phishing reports"
        }

        reasons = [feature_expl_dict.get(f, f.replace('_', ' ')) for f in top_features]
        explanation = (
            f"The model classified this input as "
            f"{'phishing' if prediction == 1 else 'legitimate'} primarily because "
            f"{', '.join(reasons[:-1])}, and {reasons[-1]}."
        )
        st.info(explanation)
