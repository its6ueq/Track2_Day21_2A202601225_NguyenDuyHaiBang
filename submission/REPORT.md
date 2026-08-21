# BÁO CÁO BÀI LAB K3: CI/CD CHO AI SYSTEMS
**Từ Thí Nghiệm Cục Bộ Đến Triển Khai Liên Tục**

- **Học viên:** Nguyễn Duy Hải Bằng (MSSV: 2A202601225)
- **Khóa học:** AIInAction - VinUni (Khoá K3 - Day 21)
- **Repository GitHub:** `https://github.com/its6ueq/Track2_Day21_2A202601225_NguyenDuyHaiBang`

---

## 1. Bước 1 — Thí Nghiệm Cục Bộ & Lựa Chọn Siêu Tham Số (MLflow)

Đã chạy 5 thí nghiệm với 5 bộ siêu tham số khác nhau cho `RandomForestClassifier`, dữ liệu Wine Quality (2998 mẫu train_phase1, 500 mẫu eval held-out). Tracking backend: `sqlite:///mlflow.db`.

| Run ID | n_estimators | max_depth | min_samples_split | Accuracy | F1 (weighted) | Nhận xét |
|---|---|---|---|---|---|---|
| `009c9b64` | 100 | 5 | 2 | 0.5640 | 0.5534 | Baseline, cây quá nông |
| `f24a4f3d` | 50 | 3 | 5 | 0.5580 | 0.5185 | Underfitting rõ |
| `631abacd` | 200 | 15 | 2 | 0.6640 | 0.6620 | Tăng độ sâu → cải thiện mạnh |
| `97feeb97` | 300 | 25 | 2 | 0.6760 | 0.6751 | Sâu hơn nữa không còn lợi |
| **`9dbef5c2`** | **300** | **20** | **2** | **0.6780** | **0.6767** | **Bộ được chọn** |

**Lý do chọn `n_estimators=300, max_depth=20, min_samples_split=2`:** cho accuracy và F1 cao nhất trong 5 lần chạy. So sánh 15 → 20 → 25 cho thấy accuracy đạt đỉnh tại `max_depth=20` rồi đi ngang/giảm nhẹ (0.6640 → 0.6780 → 0.6760), nên 20 là điểm cân bằng giữa khả năng học phi tuyến và overfitting. Tăng `n_estimators` từ 200 lên 300 giúp giảm phương sai của rừng, chi phí huấn luyện vẫn nhỏ. Giá trị này đã được ghi vào `params.yaml`.

Ngoài 5 run trên, `mlflow.db` còn 1 run tái lập lại đúng cấu hình đã chọn để sinh `models/model.pkl` và `outputs/metrics.json` cục bộ (cùng kết quả 0.6780 / 0.6767).

---

## 2. Bước 2 — Pipeline CI/CD (GitHub Actions + DVC)

`.github/workflows/mlops.yml` gồm 4 job tuần tự, trigger khi push vào `main` có thay đổi ở `data/**.dvc`, `src/**.py` hoặc `params.yaml`. Cloud provider được chọn: **AWS** (S3 cho object storage, EC2 cho VM).

1. **Unit Test** — `pytest tests/ -v`, 3 test chạy trên dữ liệu random sinh trong `tmp_path` (không cần cloud). Đã pass cục bộ và pass trên CI.
2. **Train** — đọc secret `CLOUD_CREDENTIALS` (JSON access key), export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` qua `$GITHUB_ENV`, `dvc pull` train_phase1 + eval, `python src/train.py`, dùng boto3 đẩy `models/model.pkl` lên `s3://<bucket>/models/latest/model.pkl`, export `accuracy` làm job output.
3. **Eval** — quality gate: `accuracy >= 0.70`, không đạt thì `SystemExit` và Deploy bị chặn.
4. **Deploy** — `appleboy/ssh-action` vào VM, `systemctl restart mlops-serve`, `curl -sf /health`.

### 2.1 Trạng thái thực tế (chưa hoàn thành)

| Hạng mục | Trạng thái |
|---|---|
| Code 4 file phải tự viết (`train.py`, `serve.py`, `test_train.py`, `mlops.yml`) | Xong |
| Unit test cục bộ (3 passed) | Xong |
| `data/*.dvc` được commit vào git | Xong (sửa ở commit này) |
| S3 bucket + IAM user (quyền chỉ trên bucket) | **Chưa làm** |
| `dvc push` dữ liệu lên bucket | **Chưa làm** |
| 5 GitHub Secrets (`CLOUD_CREDENTIALS`, `CLOUD_BUCKET`, `VM_HOST`, `VM_USER`, `VM_SSH_KEY`) | **Chưa làm** |
| EC2 instance + systemd `mlops-serve` | **Chưa làm** |
| 4 job xanh trên Actions | **Chưa đạt** |

Hai lần chạy CI tới thời điểm viết báo cáo đều **fail** ở job Train, nhưng nguyên nhân khác nhau:

- Run `32470305209`: fail tại `Pull data with DVC` với `data/train_phase1.csv.dvc does not exist` — 3 file con trỏ `.dvc` chưa được commit vào git. Đã sửa.
- Run `32474626418` (sau khi commit `.dvc`): Unit Test pass, fail tại `Pull data with DVC` với `Anonymous caller does not have ... access to bucket 'my-mlops-bucket' (or it may not exist)` — remote vẫn trỏ tới tên bucket ví dụ trong đề, chưa có bucket thật và chưa có secrets.

Nghĩa là phần code/cấu hình trong repo đã hết lỗi; điều kiện còn thiếu là hạ tầng AWS và 5 secrets.

### 2.2 Lưu ý về eval gate

Accuracy tốt nhất khi chỉ dùng train_phase1 là **0.6780 < 0.70**, nên với dữ liệu Bước 2 thì eval gate sẽ **chặn Deploy** — đây là hành vi đúng của quality gate, không phải lỗi. Deploy chỉ có thể xanh sau khi làm Bước 3 (bổ sung train_phase2).

---

## 3. Bước 3 — Huấn Luyện Liên Tục (chưa chạy)

Chưa thực hiện: `data/train_phase1.csv` hiện vẫn 2998 mẫu, chưa chạy `add_new_data.py`, chưa có commit dữ liệu nào kích hoạt pipeline.

Đo trước bằng script cục bộ (chỉ để dự đoán, **không phải kết quả từ pipeline**): huấn luyện cùng bộ siêu tham số trên train_phase1 + train_phase2 = 5996 mẫu cho **accuracy 0.7560 / F1 0.7552** trên cùng tập eval 500 mẫu. Nếu con số này lặp lại trên CI thì vượt ngưỡng 0.70 và Deploy sẽ chạy.

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| accuracy | 0.6780 | *(chờ kết quả CI — đo cục bộ: 0.7560)* |
| f1_score | 0.6767 | *(chờ kết quả CI — đo cục bộ: 0.7552)* |

---

## 4. Khó Khăn Gặp Phải & Cách Xử Lý

1. **CI fail vì thiếu file con trỏ DVC.** `dvc pull` trên runner báo `data/train_phase1.csv.dvc does not exist` vì thư mục `data/` chưa được `git add` (chỉ CSV bị `.gitignore`, còn `.dvc` thì bắt buộc phải vào git). Xử lý: commit 3 file `data/*.dvc`.
2. **`.dvc/cache` và `.dvc/tmp` bị commit vào git.** Thiếu file `.dvc/.gitignore` mà `dvc init` sinh ra, nên blob dữ liệu bị đẩy vào git — ngược với mục đích của DVC. Xử lý: thêm `.dvc/.gitignore` (`/config.local`, `/tmp`, `/cache`) và `git rm -r --cached .dvc/cache .dvc/tmp`.
3. **Cấu hình DVC remote theo provider.** Bản đầu để `credentialpath = sa-key.json` trong `.dvc/config` (cấu hình GCP theo ví dụ mặc định của đề). Đường dẫn tương đối này chỉ tồn tại trên máy cá nhân nên CI runner không xác thực được. Xử lý: chọn AWS làm provider, đổi remote sang `s3://`, bỏ `credentialpath` khỏi file `.dvc/config` được commit — trên CI thì `dvc-s3`/boto3 đọc `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` từ môi trường, trên máy cá nhân đọc `~/.aws/credentials` (hoặc `.dvc/config.local` không commit).
4. **`serve.py` che lỗi tải model.** Bản đầu dùng `os.getenv("GCS_BUCKET", "")` cộng `try/except` bỏ qua lỗi tải model và fallback sang `models/model.pkl` cũ trên đĩa. Hậu quả: VM có thể phục vụ model cũ mà `/health` vẫn trả `ok`, tức là deploy fail nhưng pipeline báo thành công. Xử lý: chuyển sang boto3 với `os.environ["S3_BUCKET"]` và để lỗi tải model làm service dừng, đúng tinh thần skeleton.
5. **Unit test ghi vào đúng đường dẫn artifact thật.** `tests/test_train.py` assert trên `outputs/metrics.json` và `models/model.pkl`, nên chạy `pytest` cục bộ sẽ ghi đè artifact thật bằng model huấn luyện từ dữ liệu random (accuracy 0.275). Đây là hành vi theo skeleton của đề; cần lưu ý chạy lại `python src/train.py` sau khi test nếu muốn dùng artifact cục bộ làm bằng chứng.
6. **Phiên bản thư viện.** Môi trường ảo cục bộ là Python 3.14 nên không cài được đúng các pin trong `requirements.txt` (thực tế đang dùng mlflow 3.15.1, scikit-learn 1.9.0, pandas 2.3.3). `requirements.txt` được giữ nguyên theo đề vì CI dùng Python 3.10 và pin ở đó cài được bình thường.

---

## 5. Bằng Chứng

| Bằng chứng | Trạng thái |
|---|---|
| `submission/screenshots/01_mlflow_ui.png` — MLflow UI với 5 run | Có |
| GitHub Actions 4 job xanh (Bước 2) | **Chưa có** |
| GitHub Actions 4 job xanh, trigger bởi commit dữ liệu (Bước 3) | **Chưa có** |
| `curl /health` và `curl /predict` từ EC2 | **Chưa có** |
| S3 console (dữ liệu `dvc/` + `models/latest/model.pkl`) | **Chưa có** |

## 6. Việc Còn Lại

1. Tạo S3 bucket + IAM user có policy giới hạn trên bucket, lấy access key; `dvc remote modify myremote url s3://<bucket>/dvc` rồi `dvc push`.
2. Thêm 5 GitHub Secrets (`CLOUD_CREDENTIALS` là JSON `{"aws_access_key_id": ..., "aws_secret_access_key": ...}`).
3. Tạo EC2 instance Ubuntu 22.04, security group mở tcp 22 + 8000, cài dependency, upload `serve.py`, đặt credentials ở `~/.aws/credentials`, tạo systemd `mlops-serve` với `Environment="S3_BUCKET=<bucket>"`.
4. Push để chạy pipeline Bước 2 (dự kiến Eval chặn Deploy vì 0.6780 < 0.70).
5. Chạy Bước 3: `add_new_data.py` → `dvc add data/train_phase1.csv` → commit `.dvc` → `dvc push` → `git push`, xác nhận 4 job xanh và cập nhật bảng ở mục 3.
6. Bổ sung 3 screenshot còn thiếu và điền số thật từ artifact của CI.
