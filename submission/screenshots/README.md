# Hướng dẫn lưu Ảnh chụp màn hình Nộp bài (Screenshots Evidence)

Hãy chụp 5-6 ảnh màn hình theo đúng thứ tự yêu cầu và lưu vào thư mục này (`submission/screenshots/`):

1. **`01_mlflow_ui.png`**: MLflow UI hiển thị ít nhất 3 thí nghiệm (Parameters & Metrics).
2. **`02_github_actions_phase1.png`**: Tab Actions hiển thị cả 4 jobs màu xanh (Test, Train, Eval, Deploy) cho lần chạy đầu tiên (2.998 mẫu).
3. **`03_github_actions_phase2.png`**: Tab Actions hiển thị cả 4 jobs màu xanh cho lần chạy khi commit bổ sung dữ liệu (`train_phase2.csv` - 5.996 mẫu).
4. **`04_api_curl_test.png`**: Kết quả chạy cURL kiểm tra API:
   - `curl http://<VM_IP>:8000/health` -> `{"status": "ok"}`
   - `curl http://<VM_IP>:8000/predict` -> `{"prediction": 0, "label": "thap"}`
5. **`05_cloud_storage_bucket.png`**: Cloud Storage Console (Google Cloud / AWS / Azure) hiển thị file mô hình `models/latest/model.pkl` và các file dữ liệu DVC.
6. **`06_eval_gate_failed.png`** *(Khuyến khích)*: Job Eval bị chặn (màu đỏ / FAILED) khi dùng bộ siêu tham số yếu (`accuracy < 0.70`), chứng minh Quality Gate hoạt động đúng.
