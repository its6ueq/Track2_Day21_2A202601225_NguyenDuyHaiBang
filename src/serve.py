from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# Doc ten bucket tu bien moi truong (duoc dat trong systemd service).
# Dung os.environ chu khong phai os.getenv: neu thieu cau hinh, service phai
# dung ngay thay vi khoi dong voi bucket rong.
S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu S3 ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. boto3 tu doc credentials
    tu ~/.aws/credentials hoac bien moi truong AWS_ACCESS_KEY_ID /
    AWS_SECRET_ACCESS_KEY (duoc dat trong systemd service).

    Khong bat exception: neu tai model that bai thi service phai dung lai de
    health check trong job Deploy bao do. Neu bo qua loi va tiep tuc dung model
    cu tren dia, /health van tra ve ok trong khi VM dang phuc vu model sai.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = boto3.client("s3")
    client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Model da duoc tai xuong tu s3://{S3_BUCKET}/{S3_MODEL_KEY}")


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

    Thu tu 12 dac trung (khop voi thu tu cot trong file CSV huan luyen):
        fixed acidity, volatile acidity, citric acid, residual sugar,
        chlorides, free sulfur dioxide, total sulfur dioxide, density,
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
