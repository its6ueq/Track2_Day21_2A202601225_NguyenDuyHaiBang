from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Doc ten bucket tu bien moi truong (duoc dat trong systemd service).
# Dung os.environ chu khong phai os.getenv: neu thieu cau hinh, service phai
# dung ngay thay vi khoi dong voi bucket rong.
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).

    Khong bat exception: neu tai model that bai thi service phai dung lai de
    health check trong job Deploy bao do. Neu bo qua loi va tiep tuc dung model
    cu tren dia, /health van tra ve ok trong khi VM dang phuc vu model sai.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print(f"Model da duoc tai xuong tu gs://{GCS_BUCKET}/{GCS_MODEL_KEY}")


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
