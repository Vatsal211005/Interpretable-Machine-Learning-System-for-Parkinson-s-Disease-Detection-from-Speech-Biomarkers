from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

model = joblib.load("models/parkinsons_model.pkl")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: Request, features: str = Form(...)):

    values = list(map(float, features.split(",")))

    arr = np.array(values).reshape(1, -1)

    prediction = model.predict(arr)

    result = "Parkinson Detected" if prediction[0] == 1 else "Healthy"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": result
        }
    )