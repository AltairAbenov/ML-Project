# ML Model Deployment with FastAPI and Docker

A Random Forest classifier trained on the classic **Iris** dataset, served as a REST API
with **FastAPI** and packaged in a **Docker** container.

The model predicts the iris species (`setosa`, `versicolor`, `virginica`) from four
flower measurements in centimetres.

## Project structure

```
ml-fastapi-docker/
├── main.py           # FastAPI app (GET /, POST /predict)
├── train.py          # trains and saves the model
├── model.joblib      # saved model artifact
├── requirements.txt  # Python dependencies
├── Dockerfile        # container instructions
└── README.md
```

## 1. Train the model

```bash
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python train.py
```

Output: test accuracy (about 0.90 with random_state=42) and a new `model.joblib` file.

## 2–3. Run the API locally with Uvicorn

```bash
uvicorn main:app --reload
```

- Root: http://127.0.0.1:8000/ → `{"message": "ML API is running"}`
- Swagger docs: http://127.0.0.1:8000/docs
- Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Example response:

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}
}
```

## 4–5. Build the Docker image

```bash
docker build -t ml-fastapi-docker .
```

The image contains the code, the dependencies and the runtime configuration.
The model is re-trained during the build so that it matches the scikit-learn
version inside the container.

## 6. Run the container

```bash
docker run -d -p 8000:8000 --name ml-api ml-fastapi-docker
```

Then test again: http://localhost:8000/, http://localhost:8000/docs and `POST /predict`.

Stop and remove:

```bash
docker stop ml-api && docker rm ml-api
```

## Endpoints

| Method | Path       | Description                         |
|--------|------------|-------------------------------------|
| GET    | `/`        | Health message: API is running      |
| POST   | `/predict` | Returns predicted class and probabilities |
| GET    | `/docs`    | Swagger UI (auto-generated)         |
