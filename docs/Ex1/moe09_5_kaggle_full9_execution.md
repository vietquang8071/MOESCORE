# MOE-09.5 — Notebook Kaggle full_9, hai GPU

## Phạm vi đã triển khai

- Notebook: `notebooks/MOESCORE_Full9_Kaggle.ipynb`, gồm 17 stage từ cấu hình đến artifact cuối.
- `scripts/kaggle_full9_dual_gpu.py` chuẩn bị một job/schema-v2, gán thiết bị riêng cho từng expert, chạy hai queue subprocess song song, mỗi GPU chỉ nạp một expert tại một thời điểm. Chỉ assemble sau khi chín worker hoàn tất và shard được xác thực.
- `scripts/extract_moe_feature_store.py` lưu `worker_devices` trong job và tính cache identity theo thiết bị thật của từng expert. Worker từ chối thiết bị khác với identity. CSV chiếu metadata nằm trong `projections/<expert>/`, tránh ghi đồng thời cùng một file. Giao diện một GPU cũ vẫn dùng `--device` như trước.
- `scripts/kaggle_full9_artifacts.py` xác thực ba split, đóng gói riêng `feature_cache/`, ghi SHA-256, khôi phục với kiểm tra đường dẫn và SHA-256; upload bằng `kagglehub`, sau đó tải lại manifest **và archive** từ dataset để đối chiếu SHA-256. Dataset mới được API Kagglehub tạo ở chế độ private.
- Không sửa phép tính score/feature, checkpoint hay kiến trúc gate.

## Cấu hình Kaggle

Trong Settings: chọn Kaggle environment chạy **Python 3.12** (ưu tiên khi cần `numpy<2`) hoặc Python 3.13, cùng accelerator có **ít nhất hai GPU CUDA** (ưu tiên T4 × 2). Nếu notebook cũ được tạo trên image 3.12, có thể chọn `Settings → Session options → Environment → Original`; Kaggle chỉ cho chọn các image có sẵn, không chọn tùy ý phiên bản Python ([tài liệu Kaggle](https://www.kaggle.com/docs/notebooks)). Bật Internet khi cần clone source, cài wheel hoặc upload dataset. Việc nạp model và trích xuất dùng Hugging Face offline.

Gắn bốn nguồn trong Kaggle Input: XACLE 2027, trọng số MOESCORE, Hugging Face Hub cache đầy đủ, và (nếu riêng) UniLM. Trong stage **00** sửa `DATASET_ROOT`, `WEIGHTS_ROOT`, `HF_CACHE_SOURCE`, `UNILM_SOURCE_DIR`, `KAGGLE_USERNAME`, `FEATURE_DATASET_SLUG`. Có thể chỉ định `REPOSITORY_SOURCE` là snapshot chứa đúng các file MOE-09.5; nếu dùng clone, `REPO_REF` phải trỏ tới commit đã chứa chúng. `CHECKPOINT_OVERRIDES` dùng khóa `expert.field` cho đường dẫn khác mặc định.

Layout trọng số mặc định dưới `WEIGHTS_ROOT`:

| Expert | Trọng số |
|---|---|
| expert1 | `pretrained/laion-clap.pt`, `finetuned_weights/expert1_best.pt` |
| expert2 | `pretrained/mga-clap.pt`, `finetuned_weights/expert2_best.pt` |
| expert3 | `pretrained/checkpoint-30.pth`, `finetuned_weights/expert3_best.pt` |
| expert4 | `pretrained/BEATs_iter3_plus_AS2M.pt`, `pretrained/laion-clap.pt`, `finetuned_weights/expert4_best.pt` |
| top2_m2d | `pretrained/checkpoint-30.pth` |
| top2_msclap | `pretrained/CLAP_weights_2023.pth` |
| top2_laion | `pretrained/laion-clap.pt` |
| top2_mga | `pretrained/mga-clap.pt` |
| m2d_lora | `pretrained/checkpoint-30.pth`, `best.pt` |

HF cache cần đúng Hub snapshot của `roberta-base`, `bert-base-uncased`, `google-bert/bert-base-uncased`, `gpt2`, gồm blobs/refs/snapshots. Stage 05 copy nguyên symlink tương đối, đặt `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1`; stage 07 chạy `prepare_moe_text_cache.py --profile full_9` trong tiến trình mới. Cần gắn nguồn UniLM được `verify_bundled_import_equivalence()` chấp nhận.

## Chạy notebook

1. Upload `notebooks/MOESCORE_Full9_Kaggle.ipynb` vào Kaggle Notebook; gắn Inputs, chọn Python 3.12 nếu image có sẵn và chọn hai GPU. Notebook kiểm tra cả phiên bản Python lẫn số GPU ở stage 01.
2. Sửa duy nhất cell **00**. Để `SMOKE_ONLY=True`, `RUN_GPU_SMOKE=True`, rồi **Run all**. Stage 08 trích xuất 4 mẫu/split qua chín expert; stage 09 chạy ba bước gate trên đặc trưng thật. Hai GPU được chia theo `WORKER_DEVICES`.
3. Sau smoke thành công, đặt `SMOKE_ONLY=False`, `RUN_FULL_EXTRACTION=True`, `UPLOAD_FEATURE_CACHE=True`, `RUN_GATE_TRAINING=True`, `RUN_EVALUATION=True`, `RUN_TEST_INFERENCE=True`; chạy lại các stage từ đầu. Dữ liệu đầy đủ phải có train 7500, validation 3000, test 3000; feature width được xác thực theo registry `full_9` (27399 với paired-cross `4D+1`), scores 9 cột.
4. Có thể đổi GPU mỗi expert trong `WORKER_DEVICES` nhưng phải dùng cả `cuda:0` và `cuda:1`. Đổi thiết bị thực của một expert làm đổi cache identity của expert đó theo chính sách provenance hiện hành.
5. Để resume phiên khác, gắn Kaggle Dataset cache vào `EXISTING_FEATURE_CACHE`. Notebook kiểm tra archive SHA-256 và khi chuẩn bị job sẽ đối chiếu checkpoint, source, preprocessing, sample manifest và assembly identity. Nếu store đầy đủ hợp lệ, extractor không nạp expert; nếu thiếu shard, chỉ chạy lại phần còn thiếu.

Nếu gắn một cache hoàn chỉnh và chỉ muốn train, có thể đặt `RUN_GPU_SMOKE=False`, `RUN_FULL_EXTRACTION=False`, `SMOKE_ONLY=False`; stage 11 vẫn tái lập và xác thực toàn bộ identity trước training. Không dùng cách này để bỏ qua smoke cho một lượt trích xuất mới.

Notebook lưu smoke và full ở hai root riêng. Stage 12 upload **trước** stage training. Nếu upload lỗi, exception dừng notebook; dữ liệu cục bộ vẫn còn để retry. Khi cố ý bỏ upload, đặt `UPLOAD_FEATURE_CACHE=False` trong cell 00. Upload dataset mới private theo [mã nguồn Kagglehub](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/datasets_helpers.py), và `dataset_upload` tạo version mới khi handle đã tồn tại. File upload chỉ gồm `feature_cache.tar` và `upload_manifest.json`, không chứa weights hoặc credential.

Với một handle **đã tồn tại**, phiên bản mới giữ thiết lập chia sẻ của dataset đó. Helper gọi Kaggle API `dataset_view` trước upload và từ chối dataset public hoặc metadata quyền riêng tư không xác minh được. Kaggle API cần quyền truy cập dataset; nếu không có credentials, upload dừng trước khi gửi dữ liệu.

Việc tải lại toàn bộ archive để kiểm tra byte cần thêm dung lượng tạm và băng thông tương ứng với kích thước cache. Nếu Kaggle thiếu quota, stage upload dừng và chưa được tính là đã xác minh; có thể giải phóng dung lượng, tăng quota hoặc gắn cache đã upload để tiếp tục sau khi xác minh thủ công.

Các artifact chính:

```text
/kaggle/working/moescore/smoke/
/kaggle/working/moescore/extraction/feature_cache/experts/
/kaggle/working/moescore/extraction/feature_cache/assemblies/full_9/<assembly-id>/
/kaggle/working/moescore/logs/{smoke_extraction,full_extraction,full_store_validation,upload}.json
/kaggle/working/moescore/runs/{latest_gate,best_gate}.pt
/kaggle/working/moescore/runs/validation_metrics.json
/kaggle/working/moescore/runs/validation_predictions.csv
/kaggle/working/moescore/runs/{submission,test_predictions}.csv
```

Chạy CLI hai GPU riêng khi đã chuẩn bị config:

```bash
python scripts/kaggle_full9_dual_gpu.py \
  --config /kaggle/working/moescore/configs/moe_expert_config.yaml \
  --dataset-root /kaggle/input/<dataset-root> \
  --output-root /kaggle/working/moescore/extraction \
  --worker-devices-json /kaggle/working/moescore/configs/worker_devices.json \
  --batch-sizes 1,1,1,1,1,1,1,1,1 --shard-size 128
```

Sau khi stage 11 ghi đúng `data.feature_store_root` vào gating YAML, các lệnh tương ứng là:

```bash
python scripts/train_moe_gating.py --config /kaggle/working/moescore/configs/moe_gating_config.yaml --device cuda:0
python scripts/evaluate_moe_gating.py --config /kaggle/working/moescore/configs/moe_gating_config.yaml --checkpoint /kaggle/working/moescore/runs/best_gate.pt --device cuda:0 --report-json /kaggle/working/moescore/runs/validation_metrics.json
python scripts/infer_moe_gating.py --config /kaggle/working/moescore/configs/moe_gating_config.yaml --checkpoint /kaggle/working/moescore/runs/best_gate.pt --device cuda:0 --output /kaggle/working/moescore/runs/submission.csv
```

## Python 3.12/3.13 và giới hạn xác minh

`msclap==1.3.4` và `laion-clap==1.1.7` khai báo NumPy `<2`. **Nhánh Python 3.12** chọn `requirements-kaggle-py312.txt`, pin `numpy==1.26.4`, `librosa<0.11`, `transformers<5`, và cài cả hai package model bằng resolver bình thường. Đây là cách phù hợp metadata gốc. NumPy 1.26.4 hỗ trợ Python 3.9–3.12 ([release notes NumPy](https://numpy.org/doc/stable/release/1.26.4-notes.html)).

**Nhánh Python 3.13** giữ `requirements-kaggle-py313.txt` với NumPy 2 và cài hai package model bằng `--no-deps` để giữ nguyên mã/checkpoint; preflight import và **smoke model thật** là cổng bắt buộc. Nếu NumPy 2 hoặc thư viện phụ không tương thích, notebook dừng ở stage 03/07/08. Không có fallback tự đổi phiên bản Python hoặc model. `numba` có wheel CPython 3.13 ([PyPI](https://pypi.org/project/numba/)).

Môi trường phát triển ở đây là Python **3.10.21**, PyTorch **2.14.1+cu130**, và tại thời điểm kiểm tra `torch.cuda.device_count()` trả về **0**. Đây không phải Kaggle Python 3.12/3.13 và không có hai GPU; chưa thể xác nhận khả năng nạp chín model trên đúng runtime đích, tốc độ, VRAM T4, hay upload thật. Một lần smoke cục bộ `full_9` một GPU từ milestone trước không xác nhận pipeline hai GPU này. Gating training dùng một GPU theo `GATE_DEVICE`; không triển khai DDP vì gate nhỏ và việc đổi ngữ nghĩa batch/optimizer không có lợi rõ ràng.

## Kết quả kiểm thử và resume

Kiểm thử cục bộ của MOE-09.5 dùng worker giả để chứng minh hai queue chạy đồng thời, mỗi expert được gán một lần, lỗi dừng scheduling, thiết bị worker phải trùng identity, cache archive khôi phục đúng byte và SHA-256 sai bị từ chối. Kiểm thử riêng cũng từ chối dataset public trước upload. Hồi quy Python 3.12 kiểm tra chọn requirements NumPy 1.26.4 và vẫn yêu cầu hai GPU. `python -m unittest discover -s tests -p 'test*.py' -q`: **349 tests OK, 1 skipped**. `nbformat.validate()` và biên dịch 34 cell dưới Python 3.12: **PASS**. `py_compile` và `git diff --check`: **PASS**. Không diễn giải test giả thành suy luận model thật.

Chạy riêng cell cấu hình trên Python 3.10.21: **PASS**; cell runtime dừng đúng với thông báo cần Python 3.12 hoặc 3.13. Không chạy các cell Kaggle tiếp theo trên môi trường địa phương này.

Mỗi worker ghi `logs/<assembly-job-id>/<expert>.json` và `.log` gồm GPU, batch size, checkpoint provenance, thời gian, số shard, VRAM peak và trạng thái. Shard hợp lệ còn nguyên sau lỗi; chạy lại cùng cấu hình sẽ xác thực rồi dùng lại. Không assemble nếu một worker lỗi. Cache identity thay đổi khi một expert đổi GPU do `requested_device` được lưu trong inference policy; chỉ thay thứ tự scheduling mà giữ cùng mapping không làm đổi identity.
