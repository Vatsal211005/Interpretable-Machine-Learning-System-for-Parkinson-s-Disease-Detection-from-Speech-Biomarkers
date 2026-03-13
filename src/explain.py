import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Loading model and preprocessing...")

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")

print("Loading dataset...")

df = pd.read_csv("data/pd_speech_features.csv", header=1)

X = df.drop(["id","class"], axis=1)

# sample rows for speed
X_sample = X.sample(50, random_state=42)

print("Applying preprocessing...")

X_scaled = scaler.transform(X_sample)
X_selected = selector.transform(X_scaled)

# recover feature names after SelectKBest
selected_features = X.columns[selector.get_support()]

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer.shap_values(X_selected)

# compute mean absolute SHAP values
importance = np.abs(shap_values).mean(axis=0)

feature_importance = pd.DataFrame({
    "feature": selected_features,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

top_features = feature_importance.head(20)

print("Generating bar chart...")

plt.figure(figsize=(10,6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("SHAP Importance")

plt.title("Top 20 Speech Biomarkers for Parkinson Detection")

os.makedirs("static", exist_ok=True)

plt.savefig("static/feature_importance.png", bbox_inches="tight")

plt.close()

print("Saved SHAP importance chart!")