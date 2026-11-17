# MOE-09.4 — Tích hợp chín expert và Gating Network

## Kết quả triển khai

`top1_4` vẫn là profile mặc định. `full_9` đăng ký thêm bốn expert pretrained và M2D-LoRA E34 theo thứ tự cố định. Không thay đổi mã tính điểm, vector đặc trưng, checkpoint hoặc kiến trúc của bất kỳ expert nào.

| Vị trí | Expert | Số chiều đặc trưng | Đầu vào audio |
|---:|---|---:|---|
| 1 | `expert1` | 2560 | `expert1_wavs` `[B,1,480000]` |
| 2 | `expert2` | 5120 | `expert2_wavs` `[B,1,320000]` |
| 3 | `expert3` | 3840 | `expert3_wavs` `[B,1,160000]` |
| 4 | `expert4` | 1024 | `expert4_wavs` `[B,1,160000]`, `expert4_wav_lens` `[B]` |
| 5 | `top2_m2d` | 3840 | `expert3_wavs` |
| 6 | `top2_msclap` | 5120 | `expert2_wavs`, `wav_paths` đã kiểm tra WAV/SHA-256 |
| 7 | `top2_laion` | 2560 | `expert1_wavs` |
| 8 | `top2_mga` | 5120 | `expert2_wavs` |
| 9 | `m2d_lora` | 3073 | `expert3_wavs` |

Mọi expert nhận `captions_raw`. `top1_4` ghép điểm `[B,4]`, đặc trưng `[B,12544]`; `full_9` ghép điểm `[B,9]`, đặc trưng `[B,32257]`. Bộ ghép kiểm tra tên, thứ tự theo profile, chiều, dtype, thiết bị và giá trị hữu hạn. Cổng dùng `Linear(width,512) → LayerNorm → ReLU → Dropout(0.1) → Linear(512,E) → Softmax`; dự đoán là `(scores * weights).sum(dim=1, keepdim=True)`. Không có hiệu chỉnh hay chuẩn hóa thêm trên chín điểm. Các điểm gốc của expert1–4, bốn cosine Top2 và đầu CORAL/regression E34 được giữ nguyên.

## Cache và feature store

- Trích xuất chọn profile bằng `--profile`; mỗi expert chạy trong subprocess riêng. Shard hợp lệ được xác minh trước factory, vì vậy expert có đủ shard sẽ không được nạp lại. Checkpoint Top2 M2D được đưa vào thư mục có tên mà PortableM2D cần qua symlink tạm thời, không sửa checkpoint gốc.
- Cache schema-v2 giữ identity riêng cho từng expert và identity lắp ghép phụ thuộc thứ tự. Fingerprint tiền xử lý của `_mono_float()` nay bao phủ mọi expert dùng `_expert2_audio` hoặc `_expert4_audio`, gồm Top2 MGA. Snapshot Hugging Face được băm lại mỗi lần tính identity để thay đổi trong cùng tiến trình không bị cache cũ che khuất. Fingerprint không còn băm toàn bộ bộ điều phối tiền xử lý cho mọi expert; điều kiện đầu vào hiệu lực, phương thức tiền xử lý và bộ giải đường dẫn WAV vẫn được ghi riêng.
- Feature store `top1_4` giữ schema-v1 để đọc artifact cũ. Store `full_9` dùng schema-v2, ghi profile, thứ tự, chiều từng expert, width, checkpoint provenance, preprocessing identity, identity cache từng expert, assembly identity, source manifest identity, dtype và hash shard. Validator đối chiếu SHA-256, sample order, array shape/dtype và tính hữu hạn; store sai profile, thứ tự hoặc provenance bị từ chối. Không chuyển nhãn cache cũ sang profile mới.
- Script `build_moe_feature_store.py` nhận `--assembly-id` để lắp ghép lại job per-expert đã chuẩn bị; nhánh nhập artifact bốn expert cũ vẫn còn. `validate_moe_feature_store.py` báo chiều theo manifest thực tế.

## Huấn luyện và checkpoint

Dataset/collator, trainer, evaluator và CLI train/evaluate/infer lấy profile đã xác minh từ manifest. Optimizer chỉ chứa tham số gate; expert không được nạp khi huấn luyện trên feature store. Dự đoán được ép thành `[B]` trước loss hồi quy để tránh broadcast `[B,B]`; phép đổi nhãn `(raw-5)/5`, ClippedMSE, ContrastiveLoss và diversity loss được giữ. Đánh giá trở về thang điểm gốc; test không cần nhãn và giữ thứ tự mẫu.

Checkpoint `top1_4` schema-v1 vẫn được chấp nhận nếu width là 12544. `full_9` dùng schema-v2 với profile, expert order/dimensions, gate width/count, identity store, checkpoint provenance, cấu hình, optimizer/scheduler, bước và RNG. Khôi phục kiểm tra profile, chiều, thứ tự và provenance; cổng 7936 chiều cũ không được hiểu lại thành cổng mới. Checkpoint chỉ chứa gate, không chứa chín expert.

## Cấu hình và Kaggle offline

`configs/moe_full9.example.yaml` dùng đúng tên field của từng factory. Cần thay `expert4.unilm_source_dir`, đường dẫn dataset và HF cache theo máy chạy. Generator Kaggle nhận `profile="full_9"` và `unilm_source_dir`; preflight kiểm tra chín checkpoint, `msclap==1.3.4`, các dependency và cache text. Kiểm tra offline gồm `roberta-base`, `bert-base-uncased`, `google-bert/bert-base-uncased`, `gpt2` (bao gồm trọng số GPT-2 mà MS-CLAP nạp). `HF_HOME` và `HF_HUB_CACHE` nay trỏ đúng cấp thư mục.

Trong preflight ban đầu, kiểm tra đường dẫn checkpoint/YAML/dataset/UniLM và cache Hugging Face offline đều **PASS**; `msclap` là `1.3.4`, `transformers` là `4.57.6`. Lúc preflight CUDA chưa khả dụng. Sau đó người dùng chạy bounded extraction trên `cuda:0`; kết quả bên dưới ghi nhận lần chạy mới nhất.

Preflight `prepare_job(..., profile="full_9", device="cpu", limit=1)` với checkpoint thật, cache HF thật và một bản ghi từ mỗi split XACLE 2027 **PASS**: chín expert có cache identity và mỗi split có assembly identity riêng. Bước này chỉ kiểm tra cấu hình, nội dung tệp và provenance; **không** khởi tạo model hoặc chạy audio qua expert.

## Kiểm thử

`tests/test_moe09_4_full9.py` kiểm tra registry chín factory, thứ tự/chặn đầu ra sai, các khối đặc trưng, gate Softmax và tổng chín điểm, lắp ghép 27 cache shard tổng hợp, phát hiện file hỏng, ghi/đọc ba split, một bước cập nhật gate, checkpoint save/load, đánh giá và suy luận test không nhãn. Bộ test hiện có bao phủ lại `top1_4` và adapter cũ.

Lệnh `python -m unittest discover -s tests -p 'test*.py' -q` hoàn tất **335 test, OK, 1 skipped**. Sau các chỉnh sửa cuối, chạy lại riêng `test_moe09_4_full9.py` (**4 OK**), `test_moe_checkpoint_resume.py` (**4 OK**), `test_moe_feature_store.py` (**6 OK**) và `test_kaggle_notebook_pipeline.py` (**4 OK**). `compileall` và `git diff --check` đều hoàn tất không lỗi. Không chạy trích xuất toàn bộ tập hoặc train gate đầy đủ.

Các tệp đã sửa: `configs/moe_config.py`, `data/moe_feature_store.py`, `data/moe_training_dataset.py`, `model/expert_adapters/registry.py`, `model/moe_forward_integration.py`, `model/moe_gating_core.py`, `scripts/build_moe_feature_store.py`, `scripts/evaluate_moe_gating.py`, `scripts/extract_moe_feature_store.py`, `scripts/infer_moe_gating.py`, `scripts/kaggle_moe_pipeline.py`, `scripts/prepare_moe_text_cache.py`, `scripts/train_moe_gating.py`, `scripts/validate_moe_feature_store.py`, `training/checkpoint_manager.py`, `training/moe_evaluator.py`, `training/moe_trainer.py`. Các tệp mới: `configs/moe_full9.example.yaml`, `tests/test_moe09_4_full9.py` và báo cáo này.

## Lệnh cho bước chạy tiếp theo

Sau khi sửa đường dẫn thật trong bản sao của `configs/moe_full9.example.yaml` và trỏ training config tới store vừa tạo:

```bash
python scripts/prepare_moe_text_cache.py --cache-dir /path/to/huggingface --profile full_9
python scripts/extract_moe_feature_store.py --config /path/to/moe_full9.yaml --dataset-root /path/to/XACLE_dataset --output-root /path/to/output --profile full_9 --device cuda:0 --smoke
python scripts/extract_moe_feature_store.py --config /path/to/moe_full9.yaml --dataset-root /path/to/XACLE_dataset --output-root /path/to/output --profile full_9 --device cuda:0
python scripts/validate_moe_feature_store.py --root /path/to/output/feature_cache/assemblies/full_9/ASSEMBLY_ID --split train
python scripts/train_moe_gating.py --config /path/to/moe_gating_config.yaml
python scripts/evaluate_moe_gating.py --config /path/to/moe_gating_config.yaml --checkpoint /path/to/best_gate.pt
python scripts/infer_moe_gating.py --config /path/to/moe_gating_config.yaml --checkpoint /path/to/best_gate.pt --output /path/to/submission.csv
```

`ASSEMBLY_ID` và `feature_store_root` được lệnh extraction in ra; đặt đúng giá trị đó trong gating config. Chạy lần đầu cần trích xuất lại những expert chưa có cache hợp lệ. Khi mở rộng từ `top1_4` sang `full_9`, cache của expert1–4 có cùng identity đầu ra có thể được tái sử dụng, còn assembly identity mới luôn khác. Việc chỉnh fingerprint tiền xử lý trong đợt này có thể làm đổi identity một lần so với cache trước đó; shard cũ được giữ và không tự đổi nhãn.

Bounded extraction đã chạy xong; các bước huấn luyện/evaluate gate và Kaggle job thật chưa chạy. Không có checkpoint hoặc text resource local nào bị thiếu trong preflight.

## Sửa lỗi cache BERT của Expert 3 sau smoke thử

Lần chạy thật sau khi gỡ symlink M2D vướng `LocalEntryNotFoundError`: PortableM2D truyền `HF_HOME` làm `cache_dir`, trong khi snapshot BERT nằm ở thư mục con `HF_HOME/hub`. Expert 3 nay đặt `HF_HUB_CACHE` đúng thành thư mục `hub` trong phạm vi offline và BertXEncoder dùng đường dẫn đó. Model ID và checkpoint không đổi.

Kiểm tra local `AutoConfig`/`AutoTokenizer` cho `google-bert/bert-base-uncased` với offline cache **PASS**; `test_expert3_adapter.py`: **21 test, OK, 1 skipped**; syntax và `git diff --check` đạt. Thay đổi nguồn làm identity extraction của các expert phụ thuộc PortableM2D thay đổi; lần extraction bên dưới tạo assembly identity mới. Những shard cũ được giữ nguyên.

## Kết quả bounded extraction thật trên GPU

Lệnh extraction trả về `status=PASS`, `smoke=true`, assembly ID `d36f2600898936c875da756afd649a44e56a91fcbd6ffbf33f1e154beddfd544` và store tại `/tmp/moe09_4_smoke/feature_cache/assemblies/full_9/d36f2600898936c875da756afd649a44e56a91fcbd6ffbf33f1e154beddfd544`.

Manifest và runtime report đã được kiểm tra: cả ba split train/validation/test đều PASS với 4 mẫu mỗi split. Train và validation có nhãn; test không có nhãn. Feature store có điểm `[4,9]` và đặc trưng `[4,32257]` cho từng split. Bảy expert (`expert3`, `expert4`, `top2_m2d`, `top2_msclap`, `top2_laion`, `top2_mga`, `m2d_lora`) được khởi tạo tuần tự trên `cuda:0`, ở eval và frozen. `expert1` và `expert2` dùng lại ba shard đã xác minh cho các cache identity tương ứng. Gate chưa được train hoặc chạy trên các đầu ra thật trong lần smoke này.
