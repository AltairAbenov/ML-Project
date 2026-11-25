from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(
    title="Iris ML API",
    description="Predicts the Iris flower species from 4 measurements (cm).",
    version="1.0.0",
)

if not MODEL_PATH.exists():
    raise RuntimeError(f"{MODEL_PATH} not found. Run `python train.py` first.")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1], description="Sepal length, cm")
    sepal_width: float = Field(..., gt=0, examples=[3.5], description="Sepal width, cm")
    petal_length: float = Field(..., gt=0, examples=[1.4], description="Petal length, cm")
    petal_width: float = Field(..., gt=0, examples=[0.2], description="Petal width, cm")


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict[str, float]


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    x = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    try:
        pred = int(model.predict(x)[0])
        proba = model.predict_proba(x)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        prediction=pred,
        class_name=target_names[pred],
        probabilities={name: round(float(p), 4) for name, p in zip(target_names, proba)},
    )
