
from flask import Flask, request, render_template
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

models = {
    "xgb": joblib.load("models/phishing_model_xgb.pkl"),
    "lr": joblib.load("models/phishing_model_lr.pkl"),
    "best": joblib.load("models/phishing_model_best.pkl"),
    "rf": joblib.load("models/phishing_model_rf.pkl")
}
features = list(models["best"].feature_names_in_)

feature_expl_dict = {
    "having_IP_Address": "the URL uses an IP address",
    "URL_Length": "the URL is unusually long",
    "Shortining_Service": "a URL shortening service is used",
    "having_At_Symbol": "the URL contains an '@' symbol",
    "double_slash_redirecting": "contains double slashes (//) beyond the protocol",
    "Prefix_Suffix": "a hyphen is used in the domain",
    "having_Sub_Domain": "contains multiple subdomains",
    "SSLfinal_State": "SSL certificate is invalid or missing",
    "Domain_registeration_length": "short domain registration duration",
    "Favicon": "favicon is from external domain",
    "port": "non-standard port is used",
    "HTTPS_token": "HTTPS token is present in the URL path",
    "Request_URL": "page loads external resources",
    "URL_of_Anchor": "anchors point to suspicious URLs",
    "Links_in_tags": "tags point to suspicious resources",
    "SFH": "Server Form Handler is abnormal",
    "Submitting_to_email": "form submits to an email address",
    "Abnormal_URL": "URL structure is abnormal",
    "Redirect": "page has multiple redirections",
    "on_mouseover": "content changes on hover",
    "RightClick": "right-click is disabled",
    "popUpWidnow": "pop-up windows are triggered",
    "Iframe": "uses invisible iframe tags",
    "age_of_domain": "domain is newly registered",
    "DNSRecord": "no DNS record found",
    "web_traffic": "low site traffic",
    "Page_Rank": "poor page rank",
    "Google_Index": "not indexed by Google",
    "Links_pointing_to_page": "few backlinks",
    "Statistical_report": "known in phishing reports"
}

@app.route('/')
def home():
    return "✅ Flask API is running. Use /form to access the GUI."

@app.route('/form')
def form():
    return render_template('form.html', features=features)

@app.route('/predict_form', methods=['POST'])
def predict_form():
    model_key = request.form.get("model_choice", "best")
    model = models.get(model_key, models["best"])
    input_data = {feature: int(request.form.get(feature, 0)) for feature in features}
    input_df = pd.DataFrame([input_data])[features]

    prediction = model.predict(input_df)[0]
    label = "Phishing" if prediction == 1 else "Legitimate"

    bg_data = pd.read_csv("data/phishing.csv")
    bg_data = bg_data.rename(columns={"Result": "Label"})
    bg_data["Label"] = bg_data["Label"].map({-1: 1, 1: 0})
    X_bg = bg_data.drop(columns=["Label"])[features]

    if model_key in ["xgb", "best", "rf"]:
        explainer = shap.Explainer(model, X_bg, algorithm="tree")
    else:
        explainer = shap.Explainer(model, X_bg, algorithm="linear")

    shap_values = explainer(input_df)

    # ✅ Safe SHAP value extraction for all models
    if hasattr(shap_values, "values"):
        val = shap_values.values
        if val.ndim == 3:  # shape (1, features, classes)
            shap_val_row = val[0, :, 1]  # class 1
        else:
            shap_val_row = val[0]
    else:
        shap_val_row = shap_values[0].values[0]

    sorted_idx = np.argsort(np.abs(shap_val_row))[::-1]
    top_indices = sorted_idx[:10]
    top_features = [str(f) for f in features]
    top_features = [top_features[i] for i in top_indices]
    top_values = shap_val_row[top_indices]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.xlabel("SHAP value")
    plt.title("Top 10 SHAP Feature Influences")
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/shap_plot.png")
    plt.clf()

    nle_features = [features[i] for i in sorted_idx[:3]]
    reasons = [feature_expl_dict.get(f, f.replace('_', ' ')) for f in nle_features]
    explanation = (
        f"The model classified this input as <b>{label}</b> primarily because "
        f"{', '.join(reasons[:-1])}, and {reasons[-1]}."
    )

    return f"""
        <h2>✅ Prediction: <span style='color:green'>{label}</span></h2>
        <h3>🔍 Feature Influence (SHAP)</h3>
        <img src="/static/shap_plot.png" width="600"/>
        <p style="font-size:16px;margin-top:20px;">🧠 <b>Explanation:</b> {explanation}</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
