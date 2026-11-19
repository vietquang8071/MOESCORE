# MOE-10.2 — Báo cáo notebook Kaggle Full10 Save & Run All

Ngày: 2026-11-20 | Nhánh: `feature/moe-10-m2d-lora-coral` | Commit nền lúc bắt đầu: `1b9b3def32a900bb748f63e1aec967331845943b`

Commit triển khai được pin để Kaggle clone: `867cc84d00d0cf00e53aed183e04f69f9287948e`.

## 1. Kết quả

Notebook Full10 đã được cập nhật để chạy theo thứ tự tự động từ kernel mới: kiểm tra Python/CUDA, checkout source theo SHA, cài dependencies phù hợp, xác minh Kaggle Inputs và checkpoint, khôi phục HF cache v2, preflight offline, smoke hai GPU, khôi phục cache cũ tùy chọn, trích xuất/resume Full10, train gate, đánh giá validation và suy luận test không nhãn.

Notebook dùng lại các script MOESCORE hiện có. Scheduler vẫn chạy một expert mỗi GPU theo hai hàng đợi tuần tự. Không sửa notebook Full9 đã chạy thành công và không đổi toán học của các expert hoặc gate.

**Trạng thái thực thi:**

| Hạng mục | Trạng thái |
|---|---|
| Notebook và orchestration Full10 | Đã triển khai |
| Kiểm thử notebook/helper/cache/scheduler | Đã chạy cục bộ trong `nlp` |
| Xác minh SHA checkpoint Expert 10 cục bộ | Đạt |
| Kaggle GPU smoke | Chưa chạy |
| Full extraction 15.000 mẫu trên Kaggle | Chưa chạy |
| Huấn luyện gate 50 epoch và validation 3.000 mẫu | Chưa chạy |
| Test inference trên Kaggle | Chưa chạy |

Vì không khởi chạy Kaggle job theo yêu cầu, báo cáo không ghi nhận cache reuse, số liệu Full10 hay benchmark mới như kết quả đã đo.

## 2. Khác biệt so với notebook Full9 đã xác minh

Đã đọc toàn bộ code cell, output và metadata của `notebooks/moescore-ensemblesvr-m2d.ipynb`. Notebook đó ghi nhận môi trường Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, hai Tesla T4; dependency gồm `transformers==4.57.6`, `msclap==1.3.4`, `laion-clap==1.1.7`, `timm==1.0.30`, `nnAudio==0.3.4`, NumPy 2.1.3. Nó đã hoàn tất extraction 7.500/3.000/3.000 mẫu, gate 50 epoch, validation và test inference Full9.

Notebook Full10 giữ các quyết định vận hành đó: hai GPU T4, batch size 4 cho chín expert cũ, requirements riêng cho Python 3.12/3.13, installer giữ nguyên PyTorch/CUDA, cài MS-CLAP/LAION-CLAP `--no-deps` trên Python 3.13 và kiểm tra import thật. NumPy 1.x không bị ép cài trên Python 3.13; metadata conflict legacy chỉ được ghi cảnh báo sau khi import/runtime preflight đạt.

Điểm khác biệt là Full10 kiểm tra thêm SHA và vị trí riêng của checkpoint CORAL, khôi phục HF cache v2 đã sửa, đặt batch Expert 10 bằng 1, audit cache identity Full9/Full10, và train gate độc lập 10 expert. Test evaluation sau khi chọn model bằng validation vẫn tắt mặc định.

Full9 reference từ notebook đã chạy thành công (không phải kết quả Full10): SRCC ≈ 0.673605, LCC ≈ 0.687932, KTAU ≈ 0.490364, MSE ≈ 3.156417. Notebook này không tính delta khi chưa có Full10 evaluation tương thích.

## 3. Kaggle Inputs cần gắn

Các đường dẫn được cấu hình tập trung trong code cell đầu tiên:

- XACLE development dataset: `/kaggle/input/datasets/qnv8071/xacle-2027-data/XACLE_Challenge_2027_development_dataset`
- Weights dataset: `/kaggle/input/datasets/qnv8071/xacle-weights`
- Weights root lồng: `/kaggle/input/datasets/qnv8071/xacle-weights/weights/weights`
- HF cache v2: `/kaggle/input/datasets/qnv8071/moescore-hf-text-cache-v2`
- Expert 10: gắn **Kaggle Input riêng** chứa file `m2d-lora-Nam.pt`.

Checkpoint Expert 10 không được coi là có trong `xacle-weights`. Mặc định notebook tìm duy nhất file có tên chính xác dưới `/kaggle/input`; không thấy file thì báo gắn dataset, còn nhiều file trùng tên thì bắt buộc đặt `EXPERT10_CHECKPOINT` thành đường dẫn tuyệt đối. SHA-256 bắt buộc:

`d6edbec6b724be48a18f618c98a2e25d044f06108e8715d86b58c59e97e6cbdb`

Giá trị đã được đối chiếu cục bộ với `weights/m2d-lora-Nam.pt` (14.447.361 byte). Backbone bắt buộc `weights/pretrained/checkpoint-30.pth` cũng hiện diện ở local `weights/` (1.694.677.655 byte). Notebook preflight kiểm tra cả mười checkpoint trước khi bắt đầu GPU extraction.

Ngoài ra, nếu muốn tái sử dụng cache Full9, gắn Kaggle Input cache (không bắt buộc). Notebook chấp nhận `upload_manifest.json + feature_cache.tar` hoặc thư mục `feature_cache/`. Không có cache cũ là chế độ bình thường và không làm notebook dừng.

Metadata notebook đặt GPU accelerator là NVIDIA T4 và bật internet cho clone/cài package. Trong Kaggle Settings chọn cấu hình **hai GPU T4**. Internet không được dùng để tải checkpoint hoặc model HF trong extraction; model load ở chế độ offline sau khi khôi phục cache.

## 4. HF cache v2 và offline preflight

Mặc định `REBUILD_HF_CACHE=False`, `UPLOAD_REPAIRED_HF_CACHE=False`. Notebook tìm `hf_hub_cache.tar` và `manifest.json` trong Input v2, xác thực manifest và SHA-256 trước khi giải nén, chỉ chấp nhận entry dưới `hub/`, từ chối path traversal/entry không hỗ trợ/symlink thoát khỏi `hub/`, và giữ symlink tương đối an toàn. Hub được đặt tại `/kaggle/working/moescore/huggingface/hub`; `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` và offline variables được cấu hình cho tiến trình worker.

Offline preflight xác minh năm model/tokenizer trong manifest và các model ID mà expert dùng. BART được kiểm tra bằng đúng đường gọi đã chạy ở Full9:

```python
from transformers import BartTokenizer
tok = BartTokenizer.from_pretrained("facebook/bart-base", local_files_only=True)
assert tok.vocab_size > 0
import laion_clap
```

Đoạn trên chạy trong subprocess mới. Chế độ rebuild/download chỉ dành cho bảo trì khi bật rõ ràng trong cell cấu hình; chế độ Save & Run All mặc định không tải lại năm model hoặc tạo archive nhiều GB.

## 5. Full10 extraction và tái sử dụng cache

Thứ tự là: restore cache tùy chọn → tạo job Full10 với manifest/sample identity hiện hành → so sánh expert cache identities → validate từng shard/sidecar → trích xuất shard thiếu hoặc sai → assemble store mới Full10. Assembly Full9 không bị gắn nhãn lại thành Full10.

Scheduler dùng `scripts/kaggle_full9_dual_gpu.py --profile full_10`: GPU 0 và GPU 1 có queue riêng, mỗi queue chỉ giữ một expert process tại một thời điểm. Batch size của Expert 1–9 là 4; Expert 10 là 1. Process worker kết thúc sau mỗi expert, giải phóng model/VRAM. Cache reuse trong report lấy từ bộ đếm shard thực sự đã xác thực, không suy ra từ tổng shard dự kiến. Shard lỗi cùng sidecar được chuyển vào thư mục quarantine và tính lại; identity khác checkpoint/code/preprocessing/device/sample tạo đường dẫn cache khác và được báo các trường khác nhau.

Full10 store được xác minh là profile/order 10 expert, feature width 30.472, 7.500 train, 3.000 validation, 3.000 test; test store phải không có nhãn. Validator schema-v2 hiện có xác thực hash shard, checkpoint provenance, sample order và identity.

Notebook mặc định chạy toàn bộ extraction. Nếu không muốn GPU extraction, có thể tắt cờ nhưng chỉ dùng store Full10 đã hoàn tất và khớp provenance hiện tại. Cache Full9 riêng lẻ vẫn cần chạy extraction để bổ sung expert 10 và assemble Full10.

## 6. Gate, validation và test

Gate config được sinh từ cấu hình MOESCORE hiện tại với seed 2027, `cuda:0`, batch 8, 50 epoch, learning rate `1e-5`, weight decay/diversity/loss/scheduler theo cấu hình hiện có. Gate 10 expert được tạo độc lập; checkpoint Full9 không được nạp làm Full10. Chỉ tham số gate được train.

`GATE_RESUME_CHECKPOINT` mặc định `None`. Nếu đặt, script checkpoint manager yêu cầu profile, thứ tự expert, width, feature-store identity, checkpoint provenance, optimizer/scheduler và resume state tương thích. Để bảo toàn checkpoint cũ, resume từ Input/checkpoint khác và đặt `RUNS_ROOT` vào thư mục output mới; notebook không tự chép đè checkpoint đang có.

Validation sử dụng best gate được chọn bằng validation SRCC, xác thực identity 3.000 mẫu và ghi metrics SRCC/LCC/KTAU/MSE/MAE cùng prediction CSV giữ `sample_id` và thứ tự nguồn. Test inference gọi `infer_moe_gating.py` với `require_labels=False`, xác thực 3.000 hàng, đúng schema `wav_file_name,pred_score`, finite score và thứ tự filename. `test_predictions.csv` giữ thêm sample identity. `RUN_TEST_EVALUATION=False` mặc định; nếu bật, pipeline chỉ đọc test labels sau khi chọn checkpoint bằng validation.

## 7. Cấu hình đầu cell và artifact

Các flag mặc định trong cell đầu:

```python
SMOKE_ONLY = False
RUN_GPU_SMOKE = True
RUN_FULL_EXTRACTION = True
RUN_GATE_TRAINING = True
RUN_EVALUATION = True
RUN_TEST_INFERENCE = True
RUN_TEST_EVALUATION = False
UPLOAD_FEATURE_CACHE = False
REBUILD_HF_CACHE = False
UPLOAD_REPAIRED_HF_CACHE = False
```

Artifact quan trọng ghi dưới `/kaggle/working/moescore`: configs trong `configs/`, log/preflight/smoke/extraction/validation trong `logs/`, cache expert/store trong `extraction/feature_cache/`, checkpoint `runs/best_gate.pt` và `runs/latest_gate.pt`, `runs/training_history.json`, validation metrics/predictions, `submission.csv`, `test_predictions.csv`, `test_inference.json`, và `run_summary.json`. Run summary lưu trạng thái mỗi stage, source SHA, dataset/store identity, provenance mười checkpoint, reuse statistics, GPU names, training epoch/step, validation metrics, artifacts, warnings và errors. Notebook không tự upload cache; cờ upload mặc định false.

## 8. Kiểm thử đã chạy

Các lệnh kiểm thử dùng môi trường `nlp`:

```bash
conda run -n nlp python -m pytest -q tests/test_moe10_2_kaggle_notebook.py tests/test_moe10_2_kaggle_support.py
conda run -n nlp python -m pytest -q tests/test_kaggle_full9_dual_gpu.py tests/test_moe09_3_cache_identity.py tests/test_moe10_2_kaggle_notebook.py tests/test_moe10_2_kaggle_support.py
conda run -n nlp python -m pytest -q tests
```

Kết quả thực tế: helper/notebook tests `17 passed`; Full9 scheduler/cache regression cùng các notebook tests `37 passed, 10 subtests passed`; toàn bộ `tests/`: **398 passed, 1 skipped, 278 subtests passed** trong 87.28 giây. Có một `FutureWarning` của `timm.models.layers`; không có test failure. `git diff --check` cũng đạt.

Ngoài pytest, test notebook parse JSON, compile mọi code cell, yêu cầu outputs rỗng và execution count `None`, và kiểm tra cài đặt input/config/stage order. `nbformat` không có trong `nlp`, vì vậy validation tương đương dùng JSON parser và compile từng code cell thay cho `nbformat.validate`. Helper có test checkpoint discovery/SHA, HF manifest/SHA/archive traversal/symlink, stale source pin, RunSummary, cache identity mismatch, shard quarantine và reuse counts. Full9 scheduler/cache regressions chạy trong bộ targeted.

## 9. Cách chạy trên Kaggle

1. Source triển khai đã commit trên nhánh `feature/moe-10-m2d-lora-coral`; `REPO_COMMIT_SHA` trong cell đầu trỏ chính xác tới commit `867cc84d00d0cf00e53aed183e04f69f9287948e`.
2. Tạo notebook Kaggle từ `notebooks/MOESCORE_Full10_Kaggle.ipynb`; attach XACLE dataset, `xacle-weights`, `moescore-hf-text-cache-v2`, và dataset riêng tư có `m2d-lora-Nam.pt`. Optional: attach cache Full9 nếu muốn thử reuse.
3. Nếu Expert 10 Input có duy nhất file đúng tên, để `EXPERT10_CHECKPOINT=None`. Nếu có file trùng tên ở nhiều Input, đặt thành đường dẫn tuyệt đối tới đúng file; SHA vẫn phải khớp.
4. Bật Internet để checkout/cài package, chọn hai GPU T4, giữ các cờ mặc định trong cell đầu.
5. Chọn **Save Version → Save & Run All**. Theo dõi `run_summary.json` và log; chỉ coi một stage thành công khi status PASS và artifact kiểm tra đã tồn tại.

Các code cell không yêu cầu thao tác thủ công hoặc restart giữa chừng. Nếu dependency installer phát hiện phiên bản module đã import bị thay đổi, nó dừng sớm và yêu cầu chạy kernel mới; Save & Run All tạo kernel mới nên không cần xử lý bằng tay giữa cell.

## 10. Giới hạn còn lại

- Không có Kaggle remote job hoặc GPU T4x2 nào được khởi chạy; smoke/full extraction/train/evaluation trên GPU chưa được xác nhận.
- Notebook clone commit implementation `867cc84d00d0cf00e53aed183e04f69f9287948e`, không clone `main`; Kaggle job chưa được chạy để xác nhận hành vi trên runtime T4 hiện tại.
- Người dùng cần gắn Expert 10 private Kaggle Input; đường dẫn và SHA đã ghi trong mục 3. Không upload checkpoint lên Kaggle Dataset công khai.
- Full10 metrics và Full9-vs-Full10 delta chưa được đo. Các số Full9 trong mục 2 chỉ là kết quả tham chiếu từ notebook đã chạy.
- Chưa xác minh notebook thực thi thành công bằng toàn bộ Kaggle private Inputs và phiên bản Kaggle runtime hiện tại.

### Trạng thái mức độ sẵn sàng

**Code và local tests sẵn sàng; implementation commit đã được tạo.** Gắn đúng private Inputs, bật hai GPU và chạy Save & Run All để xác minh trên Kaggle. Chưa thể gọi Full10 là “đã train/đã benchmark” trước khi pipeline hoàn tất.
