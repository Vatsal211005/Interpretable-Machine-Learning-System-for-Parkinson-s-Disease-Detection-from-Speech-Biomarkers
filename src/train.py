import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import dagshub
import os

from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# --------------------------------
# Initialize DagsHub + MLflow
# --------------------------------
dagshub.init(
    repo_owner="nishnarudkar",
    repo_name="Interpretable-Machine-Learning-System-for-Parkinson-s-Disease-Detection-from-Speech-Biomarkers",
    mlflow=True
)

mlflow.set_experiment("parkinson_detection")


# --------------------------------
# Load dataset
# --------------------------------
df = pd.read_csv("data/pd_speech_features.csv", header=1)

df = df.drop("id", axis=1)

X = df.drop("class", axis=1)
y = df["class"]


# --------------------------------
# Train Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Preprocessing
# --------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(f_classif, k=100)

X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)


# --------------------------------
# Baseline Models
# --------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}


best_acc = 0
best_model = None
best_run_id = None


# --------------------------------
# Train baseline models
# --------------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name) as run:

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        mlflow.log_param("model", name)
        mlflow.log_param("num_features", 100)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, name="model")

        print(f"{name} accuracy:", acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_run_id = run.info.run_id


# --------------------------------
# XGBoost Hyperparameter Tuning
# --------------------------------
depth_values = [3, 5, 7]
learning_rates = [0.05, 0.1]
estimators = [100, 200]

for depth, lr, est in product(depth_values, learning_rates, estimators):

    with mlflow.start_run(run_name="XGBoost_tuning") as run:

        model = XGBClassifier(
            max_depth=depth,
            learning_rate=lr,
            n_estimators=est,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("n_estimators", est)
        mlflow.log_param("num_features", 100)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, name="xgboost_model")

        print(f"XGBoost depth={depth} lr={lr} est={est} accuracy={acc}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_run_id = run.info.run_id


# --------------------------------
# Register Best Model
# --------------------------------
model_uri = f"runs:/{best_run_id}/model"

mlflow.register_model(
    model_uri,
    "parkinson_detection_model"
)


# --------------------------------
# Save best model locally for API
# --------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(selector, "models/selector.pkl")

print("\nBest Model Accuracy:", best_acc)