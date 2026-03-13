from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import pandas as pd

app = FastAPI()

# templates + static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# load model artifacts
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)


# request schema
class FeatureInput(BaseModel):
    features: list


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(data: FeatureInput):

    # convert to numpy
    arr = np.array(data.features).reshape(1, -1)

    # apply preprocessing
    arr_scaled = scaler.transform(arr)
    arr_selected = selector.transform(arr_scaled)

    # prediction
    prediction = int(model.predict(arr_selected)[0])

    # probability
    prob = float(model.predict_proba(arr_selected)[0][1])

    # SHAP explanation
    shap_values = explainer.shap_values(arr_selected)

    shap_values = shap_values[0]

    # top contributing features
    importance = np.abs(shap_values)
    top_indices = np.argsort(importance)[-10:][::-1]

    explanation = [
        {
            "feature_index": int(i),
            "impact": float(shap_values[i])
        }
        for i in top_indices
    ]

    return {
        "prediction": prediction,
        "probability": prob,
        "top_contributions": explanation
    }