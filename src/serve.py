from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if GCS_BUCKET:
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(GCS_MODEL_KEY)
            blob.download_to_filename(MODEL_PATH)
            print("Model da duoc tai xuong tu GCS.")
            return
        except Exception as e:
            print(f"Warning: Failed to download model from GCS: {e}")
    
    if not os.path.exists(MODEL_PATH) and os.path.exists("models/model.pkl"):
        import shutil
        shutil.copy("models/model.pkl", MODEL_PATH)
        print("Model local duplicate copied to MODEL_PATH.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


LABEL_MAPPING = {0: "thap", 1: "trung_binh", 2: "cao"}


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 12 features, got {len(req.features)}"
        )

    pred = int(model.predict([req.features])[0])
    label = LABEL_MAPPING.get(pred, "unknown")
    return {"prediction": pred, "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
