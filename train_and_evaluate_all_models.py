import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/phishing.csv")
X = df.drop("Result", axis=1)
y = df["Result"].map({-1: 0, 1: 1})  # Map -1 to 0 for binary classification

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    "lr": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgb": XGBClassifier(eval_metric="logloss")
}

# Train, evaluate, and save models + metrics
for name, model in models.items():
    print(f"🚀 Training {name.upper()} model...")
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"models/phishing_model_{name}.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    # Save metrics
    metrics = {
        "classification_report": report,
        "confusion_matrix": matrix
    }
    with open(f"metrics/metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=4)

print("✅ All models trained and evaluated. Outputs saved to 'models/' and 'metrics/'.")