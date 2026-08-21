# BÁO CÁO BÀI LAB K3: CI/CD CHO AI SYSTEMS
**Từ Thí Nghiệm Cục Bộ Đến Triển Khai Liên Tục**

- **Học viên:** Nguyễn Duy Hải Bằng (MSSV: 2A202601225)
- **Khóa học:** AIInAction - VinUni (Khoá K3 - Day 21)
- **Repository GitHub:** `https://github.com/its6ueq/Track2_Day21_2A202601225_NguyenDuyHaiBang`
- **Cloud provider:** AWS (S3 cho object storage, EC2 cho VM inference)

---

## 1. Bước 1 — Thí Nghiệm Cục Bộ & Lựa Chọn Siêu Tham Số (MLflow)

Đã chạy 5 thí nghiệm với 5 bộ siêu tham số khác nhau cho `RandomForestClassifier`, dữ liệu Wine Quality (2998 mẫu `train_phase1`, 500 mẫu `eval` held-out). Tracking backend: `sqlite:///mlflow.db`.

| Run ID | Run name (trên MLflow UI) | n_estimators | max_depth | min_samples_split | Accuracy | F1 (weighted) | Nhận xét |
|---|---|---|---|---|---|---|---|
| `009c9b64` | `omniscient-loon-530` | 100 | 5 | 2 | 0.5640 | 0.5534 | Baseline, cây quá nông |
| `f24a4f3d` | `victorious-mare-702` | 50 | 3 | 5 | 0.5580 | 0.5185 | Underfitting rõ |
| `631abacd` | `placid-newt-93` | 200 | 15 | 2 | 0.6640 | 0.6620 | Tăng độ sâu → cải thiện mạnh |
| `97feeb97` | `vaunted-hound-253` | 300 | 25 | 2 | 0.6760 | 0.6751 | Sâu hơn nữa không còn lợi |
| **`9dbef5c2`** | **`capable-perch-860`** | **300** | **20** | **2** | **0.6780** | **0.6767** | **Bộ được chọn** |

**Lý do chọn `n_estimators=300, max_depth=20, min_samples_split=2`:** cho accuracy và F1 cao nhất trong 5 lần chạy. So sánh 15 → 20 → 25 cho thấy accuracy đạt đỉnh tại `max_depth=20` rồi đi ngang/giảm nhẹ (0.6640 → 0.6780 → 0.6760), nên 20 là điểm cân bằng giữa khả năng học phi tuyến và overfitting. Tăng `n_estimators` từ 200 lên 300 giúp giảm phương sai của rừng, chi phí huấn luyện vẫn nhỏ. Giá trị này đã được ghi vào `params.yaml`.

Ngoài 5 run thí nghiệm trên, `mlflow.db` còn 2 run tái lập bằng chính `src/train.py` với bộ tham số đã chọn: `0e8a0ac9` / `auspicious-shad-906` trên dữ liệu Bước 2 (2998 mẫu → 0.6780 / 0.6767) và `34155658` / `righteous-wasp-66` trên dữ liệu Bước 3 (5996 mẫu → 0.7560 / 0.7552). Tổng cộng 7 run active trong ảnh `01_mlflow_ui.png`.

---

## 2. Bước 2 — Pipeline CI/CD (GitHub Actions + DVC)

`.github/workflows/mlops.yml` gồm 4 job tuần tự, trigger khi push vào `main` có thay đổi ở `data/**.dvc`, `src/**.py` hoặc `params.yaml` (thêm `workflow_dispatch` để chạy tay khi cần).

1. **Unit Test** — `pytest tests/ -v`, 3 test chạy trên dữ liệu random sinh trong `tmp_path` (không cần cloud).
2. **Train** — đọc secret `CLOUD_CREDENTIALS` (JSON access key), export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` qua `$GITHUB_ENV`, `dvc pull` `train_phase1` + `eval`, `python src/train.py`, dùng boto3 đẩy `models/model.pkl` lên `s3://<bucket>/models/latest/model.pkl`, export `accuracy` làm job output.
3. **Eval** — quality gate `accuracy >= 0.70`; không đạt thì `SystemExit` và Deploy bị chặn.
4. **Deploy** — `appleboy/ssh-action` vào EC2, `systemctl restart mlops-serve`, `curl -sf http://localhost:8000/health`.

### 2.1 Hạ tầng AWS đã dựng

| Hạng mục | Chi tiết |
|---|---|
| S3 bucket | `mlops-lab-day21-708664` (region `ap-southeast-1`), chứa `dvc/` (DVC remote) và `models/latest/model.pkl` |
| IAM user cho CI | `mlops-ci` + inline policy chỉ cho phép `ListBucket`/`GetBucketLocation` trên bucket và `Get/Put/DeleteObject` trên `bucket/*` |
| GitHub Secrets | `CLOUD_CREDENTIALS`, `CLOUD_BUCKET`, `VM_HOST`, `VM_USER`, `VM_SSH_KEY` |
| EC2 | `i-01c22886abc8dd4e7`, t3.micro, Ubuntu 22.04, public IP `13.212.129.221`, security group mở tcp 22 + 8000 |
| Service trên VM | systemd unit `mlops-serve` chạy `python3 ~/src/serve.py`, `Environment="S3_BUCKET=mlops-lab-day21-708664"`, `Restart=always` |

Trên VM cài `scikit-learn==1.4.2` và `joblib==1.4.2` khớp với pin trong `requirements.txt` của CI, vì file pickle sinh bởi sklearn 1.4.2 không đảm bảo đọc được bằng phiên bản khác.

### 2.2 Quality gate chặn Deploy với dữ liệu Bước 2 (bằng chứng)

Run **`32479269248`** chạy trên dữ liệu Bước 2 (2998 mẫu):

| Job | Kết quả |
|---|---|
| Unit Test | ✅ 3 passed |
| Train | ✅ train xong, upload `s3://.../models/latest/model.pkl` |
| Eval | ❌ `FAILED: accuracy 0.6780 < 0.70. Cancelled deploy.` |
| Deploy | ⏭️ skipped |

Đây là hành vi **đúng** của quality gate, không phải lỗi: model 0.6780 không được phép ra production. Deploy chỉ xanh sau khi Bước 3 bổ sung dữ liệu.

### 2.3 Các lần CI fail trước đó và nguyên nhân

| Run | Fail ở | Nguyên nhân |
|---|---|---|
| `32470305209` | Pull data with DVC | `data/train_phase1.csv.dvc does not exist` — 3 file con trỏ `.dvc` chưa được commit vào git |
| `32474626418` | Pull data with DVC | `Anonymous caller does not have storage.objects.list access to bucket 'my-mlops-bucket'` — remote còn trỏ tới bucket ví dụ trong đề, chưa có bucket thật |
| `32477131211` | Pull data with DVC | `failed to connect to s3 (my-mlops-bucket/...) - Unable to locate credentials` — đã đổi sang provider AWS nhưng chưa có bucket thật và chưa có secrets |

---

## 3. Bước 3 — Huấn Luyện Liên Tục

Quy trình đã chạy: `python add_new_data.py` (2998 → 5996 mẫu) → `dvc add data/train_phase1.csv` → commit `data/train_phase1.csv.dvc` (`cd2fbe5`) → `dvc push` → `git push`. Chỉ commit **file con trỏ 100 byte**, dữ liệu CSV đi lên S3 qua DVC.

Push kích hoạt run **`32480027037`** — **4/4 job xanh:**

| Job | Kết quả |
|---|---|
| Unit Test | ✅ 3 passed |
| Train | ✅ `dvc pull` 5996 mẫu, train, upload model 62,775,265 byte lên S3 |
| Eval | ✅ `Evaluated model accuracy: 0.7560` → `PASSED: Eval threshold met (>= 0.70)` |
| Deploy | ✅ `systemctl restart mlops-serve` → `{"status":"ok"}` `Health check passed.` |

Số liệu lấy từ artifact `metrics` của chính run này (`outputs/metrics.json`):

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) | Thay đổi |
|---|---|---|---|
| accuracy | 0.6780 | **0.7560** | +0.0780 |
| f1_score | 0.6767 | **0.7552** | +0.0785 |

**Kết luận:** gấp đôi dữ liệu huấn luyện đưa accuracy từ dưới ngưỡng lên trên ngưỡng, vượt gate và tự động deploy — đúng vòng lặp continuous training mà bài lab muốn thể hiện. Bộ siêu tham số không đổi, chỉ dữ liệu thay đổi, nên phần cải thiện đến từ dữ liệu.

### 3.1 Kiểm tra API sau deploy

Model trên VM sau khi Deploy: `/home/ubuntu/models/model.pkl`, 62,775,265 byte, timestamp `Aug 21 12:05` — khớp đúng file mà job Train vừa upload lên S3.

```
$ curl http://13.212.129.221:8000/health
{"status":"ok"}

# Vang do (wine_type=0)
$ curl -X POST http://13.212.129.221:8000/predict -H 'Content-Type: application/json' \
    -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
{"prediction":0,"label":"thap"}

# Vang trang (wine_type=1)
$ curl -X POST http://13.212.129.221:8000/predict -H 'Content-Type: application/json' \
    -d '{"features": [6.3, 0.30, 0.34, 1.6, 0.049, 14.0, 132.0, 0.9940, 3.30, 0.49, 9.5, 1]}'
{"prediction":1,"label":"trung_binh"}

# Sai so luong dac trung
$ curl -X POST http://13.212.129.221:8000/predict -H 'Content-Type: application/json' \
    -d '{"features": [1,2,3]}'
{"detail":"Expected 12 features, got 3"}      # HTTP 400
```

---

## 4. Khó Khăn Gặp Phải & Cách Xử Lý

1. **CI fail vì thiếu file con trỏ DVC.** `dvc pull` trên runner báo `data/train_phase1.csv.dvc does not exist` vì các file `.dvc` chưa được `git add` (chỉ CSV bị `.gitignore`, còn `.dvc` thì **bắt buộc** phải vào git). Xử lý: commit 3 file `data/*.dvc`.
2. **`.dvc/cache` và `.dvc/tmp` bị commit vào git.** Thiếu file `.dvc/.gitignore` mà `dvc init` sinh ra, nên blob dữ liệu bị đẩy vào git — ngược hoàn toàn mục đích của DVC. Xử lý: thêm `.dvc/.gitignore` (`/config.local`, `/tmp`, `/cache`) và `git rm -r --cached .dvc/cache .dvc/tmp`.
3. **Cấu hình DVC remote theo provider.** Bản đầu để `credentialpath = sa-key.json` trong `.dvc/config` (mẫu GCP của đề). Đường dẫn tương đối này chỉ tồn tại trên máy cá nhân nên CI runner không xác thực được (`Unable to locate credentials`). Xử lý: chọn AWS, đổi remote sang `s3://mlops-lab-day21-708664/dvc`, bỏ `credentialpath` khỏi `.dvc/config` được commit; trên CI thì `dvc-s3`/boto3 đọc biến môi trường, trên máy cá nhân đọc `~/.aws/credentials` và `.dvc/config.local` (không commit).
4. **`serve.py` che lỗi tải model.** Bản đầu dùng `os.getenv("GCS_BUCKET", "")` cộng `try/except` bỏ qua lỗi tải model rồi fallback sang `models/model.pkl` cũ trên đĩa. Hậu quả: VM có thể phục vụ model cũ mà `/health` vẫn trả `ok` — deploy fail nhưng pipeline báo thành công. Xử lý: dùng boto3 với `os.environ["S3_BUCKET"]`, không bắt exception, để lỗi tải model làm service dừng để health check của job Deploy phát hiện được.
5. **IAM permissions boundary chặn `s3:CreateBucket`.** Lần đầu tạo bucket bị `AccessDenied ... because no permissions boundary allows the s3:CreateBucket action`: policy `IAMFullAccess` bị gắn làm **permissions boundary** của user chứ không chỉ là policy thường, nên boundary trở thành trần quyền và mọi action S3 bị chặn. Xử lý: `aws iam delete-user-permissions-boundary`, đợi vài chục giây cho IAM đồng bộ rồi tạo lại bucket thành công.
6. **Không khớp phiên bản scikit-learn giữa CI và VM.** File `.pkl` do sklearn 1.4.2 (Python 3.10 trên CI) sinh ra không đảm bảo đọc được bằng phiên bản khác. Xử lý: cài đúng `scikit-learn==1.4.2` + `joblib==1.4.2` trên EC2.
7. **Unit test ghi vào đúng đường dẫn artifact thật.** `tests/test_train.py` assert trên `outputs/metrics.json` và `models/model.pkl`, nên chạy `pytest` cục bộ sẽ ghi đè artifact thật bằng model huấn luyện từ dữ liệu random (accuracy ~0.275) và thêm run rác vào `mlflow.db`. Đây là hành vi theo skeleton của đề; cần chạy lại `python src/train.py` sau khi test nếu muốn dùng artifact cục bộ làm bằng chứng.
8. **Phiên bản thư viện cục bộ.** Môi trường ảo cục bộ là Python 3.14 nên không cài được đúng các pin trong `requirements.txt` (đang dùng mlflow 3.15.1, scikit-learn 1.9.0, pandas 2.3.3). `requirements.txt` giữ nguyên theo đề vì CI dùng Python 3.10 và pin ở đó cài bình thường.

---

## 5. Bằng Chứng

| Bằng chứng | Nguồn |
|---|---|
| MLflow UI với 7 run (5 thí nghiệm + 2 tái lập) | `submission/screenshots/01_mlflow_ui.png` |
| 4 job xanh, trigger bởi commit dữ liệu (Bước 3) | `submission/screenshots/02_actions_4_green.png` — Actions run `32480027037` |
| Quality gate chặn Deploy với dữ liệu Bước 2 | `submission/screenshots/03_actions_gate_blocked.png` — Actions run `32479269248` |
| `curl /health` và `curl /predict` tới EC2 | `submission/screenshots/04_curl_api.png` |
| S3 console: `dvc/` + `models/latest/model.pkl` | `submission/screenshots/05_s3_console.png` |
| Metrics từ CI | artifact `metrics` của run `32480027037`: `{"accuracy": 0.756, "f1_score": 0.7552068895839901}` |
| Model artifact trên cloud | `s3://mlops-lab-day21-708664/models/latest/model.pkl` (62,775,265 byte) |
| Dữ liệu trên DVC remote | `s3://mlops-lab-day21-708664/dvc/files/md5/...` (4 object) |
| API hoạt động | `curl /health` và `/predict` tới `http://13.212.129.221:8000` (mục 3.1) |
