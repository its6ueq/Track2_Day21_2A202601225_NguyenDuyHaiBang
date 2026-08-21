# BÁO CÁO BÀI LAB K3: CI/CD CHO AI SYSTEMS
**Từ Thí Nghiệm Cục Bộ Đến Triển Khai Liên Tục**

- **Học viên:** Nguyễn Duy Hải Bằng (MSSV: 2A202601225)
- **Khóa học:** AIInAction - VinUni (Khoá K3 - Day 21)
- **Repository GitHub:** `https://github.com/its6ueq/Track2_Day21_2A202601225_NguyenDuyHaiBang`

---

## 1. Kết Quả Thí Nghiệm Cục Bộ & Lựa Chọn Siêu Tham Số (MLflow Tracking)

### 1.1 Quá trình thí nghiệm trên MLflow
Đã tiến hành 8 lượt chạy (experiments) với các bộ siêu tham số khác nhau cho thuật toán **RandomForestClassifier** trên tập dữ liệu Wine Quality (2,998 mẫu huấn luyện, 500 mẫu đánh giá held-out set). Toàn bộ được theo dõi qua MLflow SQLite database (`mlflow.db`).

| Run ID | `n_estimators` | `max_depth` | `min_samples_split` | Accuracy | F1 Score (weighted) | Ghi chú |
|---|---|---|---|---|---|---|
| `009c9b64` | 100 | 5 | 2 | 0.5640 | 0.5534 | Baseline shallow forest |
| `f24a4f3d` | 50 | 3 | 5 | 0.5580 | 0.5185 | Underfitting nhẹ |
| `631abacd` | 200 | 15 | 2 | 0.6640 | 0.6620 | Cải thiện đáng kể độ chính xác |
| `97feeb97` | 300 | 25 | 2 | 0.6760 | 0.6751 | Mô hình học tốt đặc trưng phức tạp |
| **`9dbef5c2`** | **300** | **20** | **2** | **0.6780** | **0.6767** | **Bộ siêu tham số tối ưu (Phase 1)** |

### 1.2 Lý do lựa chọn bộ siêu tham số tối ưu
- Bộ tham số `n_estimators=300`, `max_depth=20`, `min_samples_split=2` giúp mô hình RandomForest capture được các phi tuyến phức tạp giữa 12 chỉ số hóa học của rượu vang mà không bị overfitting quá mức trên tập validation.
- Đạt chỉ số Accuracy = **0.6780** và F1 Score = **0.6767** tốt nhất trong tất cả các thí nghiệm Phase 1.

---

## 2. Kiến Trúc Pipeline CI/CD & Tự Động Hóa (GitHub Actions & DVC)

### 2.1 Cấu trúc 4 Giai Đoạn (Jobs) Trong `mlops.yml`
1. **Unit Test (`test`):** Tự động khởi tạo dữ liệu giả lập (`pytest tests/ -v`) kiểm tra tính đúng đắn của logic hàm `train()`, định dạng outputs `metrics.json` và file model `model.pkl`.
2. **Train (`train`):** Xác thực Google Cloud Credentials (`sa-key.json`), kéo dữ liệu được quản lý phiên bản qua DVC (`dvc pull`), thực hiện huấn luyện mô hình và đẩy `models/latest/model.pkl` lên GCS Bucket.
3. **Eval Gate (`eval`):** Đọc độ chính xác (`accuracy`) từ output job train. Áp dụng Quality Gate `accuracy >= 0.70`.
4. **Deploy (`deploy`):** SSH trực tiếp vào Cloud VM, tự động khởi động lại dịch vụ `mlops-serve` (FastAPI) và kiểm tra sức khỏe endpoint `curl -sf http://localhost:8000/health`.

### 2.2 Mô Phỏng Huấn Luyện Liên Tục (Continuous Retraining - Phase 2)
- Khi có dữ liệu mới (`train_phase2.csv` bổ sung thêm 2,998 mẫu), dữ liệu được kết hợp thành 5,996 mẫu.
- **Kết quả retrain:** Accuracy tăng vọt từ **0.6780** lên **0.7580** (F1 score = **0.7562**).
- Vượt qua ngưỡng Quality Gate (`0.7580 >= 0.70`), tự động kích hoạt tiến trình Deploy mô hình sản xuất thành công lên Cloud VM.

---

## 3. Xác Nhận API Suy Luận (FastAPI Serving Endpoint)

FastAPI Inference Service (`src/serve.py`) triển khai tại cổng 8000:
- **`GET /health`**: Trả về `{"status": "ok"}` (HTTP status code 200).
- **`POST /predict`**:
  - Input: `{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}`
  - Output: `{"prediction": 0, "label": "thap"}` (HTTP status code 200).

---

## 4. Báo Cáo Khó Khăn Gặp Phải & Giải Pháp

1. **Xung đột phiên bản Python 3.14 & Khả năng tương thích Scikit-Learn:**
   - *Khó khăn:* Môi trường mặc định sử dụng Python 3.14 trong khi một số dependency cũ trong `requirements.txt` yêu cầu pre-built wheels cho C-extensions.
   - *Giải pháp:* Đã điều chỉnh cài đặt phiên bản phù hợp tương thích với Python 3.14, đảm bảo build wheel thành công và chạy mượt mà tất cả unit tests.
2. **Quản lý DVC Credentials trong CI/CD Runner:**
   - *Khó khăn:* GitHub Actions Runner cần truy cập an toàn vào GCS Bucket mà không lưu mật khẩu công khai trong repo.
   - *Giải pháp:* Mã hóa Service Account JSON Key vào GitHub Secrets (`CLOUD_CREDENTIALS`), tự động ghi ra file tạm `/tmp/sa-key.json` trong giai đoạn chạy pipeline.

---

## 5. Bằng Chứng Thực Hiện (Screenshot Evidence Summary)

1. **MLflow UI:** Đã ghi nhận 8 lượt chạy thí nghiệm trong bảng điều khiển MLflow với đầy đủ Parameters & Metrics.
2. **GitHub Actions Workflow:** Cả 4 jobs (`test`, `train`, `eval`, `deploy`) đều đạt trạng thái xanh (Passed 100%).
3. **Cloud VM Serving:** Đã verify phản hồi từ cURL terminal cho cả endpoint `/health` và `/predict`.
4. **Cloud Object Storage:** Đã verify việc đồng bộ hóa dữ liệu DVC (`.dvc`) và artifact mô hình tại `gs://<CLOUD_BUCKET>/models/latest/model.pkl`.
