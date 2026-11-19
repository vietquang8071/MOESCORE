# Báo cáo triển khai MOE27-Core Full7 (Core.1–Core.5)

Ngày: 2026-11-19. Phạm vi: triển khai offline Full7 và kiểm chứng tổng hợp;
**không chạy Core.6 hoặc huấn luyện trên dữ liệu thật**.

## A. Git và phạm vi thay đổi

- Worktree: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE-Core`
- Branch: `dev/moe27-core`
- HEAD ban đầu và hiện tại: `3409b8a2df8f4ec248eb4bf0db872c2ce50bbc1c`
- `feature/xacle2027-7expert-moe` là tổ tiên của HEAD (`git merge-base --is-ancestor`: exit 0).
- Worktree ban đầu sạch. Không commit, push, merge hoặc cập nhật nhánh từ xa.
- Sửa: `core-overview.md`, `model/expert_adapters/registry.py`,
  `data/moe_feature_store.py`, `training/checkpoint_manager.py`,
  `training/config.py`, `training/moe_evaluator.py`,
  `scripts/train_moe_gating.py`, `scripts/evaluate_moe_gating.py`,
  `scripts/infer_moe_gating.py`.
- Thêm: `data/full7_assembler.py`, `data/full7_score_contract.py`,
  `scripts/build_full7_feature_store.py`, `scripts/audit_full7_scores.py`,
  `configs/moe_full7.example.yaml`, `docs/MOE27-Core-implementation.md`,
  năm file `tests/test_full7_*.py` và báo cáo này.
- Không có file weight, cache, WAV hoặc NumPy mới trong danh sách untracked.

## B. Ma trận triển khai

| Giai đoạn | Trạng thái | Bằng chứng |
|---|---|---|
| Core.1 Registry + Gate | PASS | `profile_contract('full_7')`; test thứ tự/slice, hình dạng, giá trị bất hợp lệ; 29,959→512→7. |
| Core.2 Builder + Store | PASS trên dữ liệu tổng hợp | Builder dùng Full9/Whisper/E5/E6 đã xác minh; store v3 train/validation/test, SHA-256, atomic resume, test không nhãn. |
| Core.3 Training + Checkpoint | PASS trên dữ liệu tổng hợp | Gate-only train/evaluate; checkpoint v3; resume CPU cho cùng tham số sau bước tiếp theo; train rows thực dùng được báo cáo. |
| Core.4 CLI + Inference | PASS trên dữ liệu tổng hợp | Build CLI chạy lại cả ba split; train/evaluate/infer path; CSV hai cột theo yêu cầu Full7. |
| Core.5 Regression | PASS trong `tests/`; toàn repo còn lỗi ngoài phạm vi Core | `461 passed, 1 skipped` trong `tests/`; full repo: `498 passed, 11 failed, 1 skipped`. |

## C. Hợp đồng đã xác minh

`expert1 → expert2 → expert3 → expert4 → expert5 → expert6 → expert7`;
score `float32 [B,7]`; feature `float32 [B,29959]`. E5 lấy
`top2_m2d + top2_msclap + top2_mga` (11267); E6 lấy
`top2_laion + Whisper audio + Whisper text` (4609); E7 lấy đúng Full9
`m2d_lora` (3073). Gate có `Linear(29959,512)`, LayerNorm, ReLU,
Dropout(0.1), `Linear(512,7)`, Softmax. Chỉ gate có tham số trainable trong
optimizer. Score không đi vào MLP; fusion là `(scores * weights).sum(1)`.

Target train dùng `(raw-5)/5`; output raw dùng `5*normalized+5` đúng một lần.
Score contract từng expert ghi nguồn, semantics, target/output scale, conversion,
fingerprint, evidence và audit status; cache thật phải có hash mã nguồn và
checkpoint/score contract khớp Full9. E3 training code tại
`moescore_expert3_dataset.py:100` có phép chuyển raw→normalized; output thật
của checkpoint chưa được kiểm tra. E1/E2 là similarity proxy chưa được xác nhận
hiệu chuẩn theo điểm người chấm. Không thêm scaler, clip hoặc calibrator.
Huấn luyện thật yêu cầu file review ký nhận từng expert, gắn hash store,
contract và audit train-only; giá trị ngoài [-1,1] tự nó không quyết định scale.
Full7 manifest ghi WAV/caption hash từng hàng và membership E5/E6; train và
test inference từ chối WAV trùng giữa split. Checkpoint v3 và run manifest ghi
hash mã nguồn, run manifest ghi trạng thái Git dirty.

## D. Kiểm thử thực tế

Môi trường dùng: `/home/vitquay1708/miniconda3/envs/nlp/bin/python`.

| Lệnh | Kết quả |
|---|---|
| `python -m pytest -q tests/test_moe_gating.py tests/test_moe_feature_store.py tests/test_moe_checkpoint_resume.py tests/test_set_a_residual.py tests/test_set_b_residual.py` | Baseline trước sửa: **48 passed, 14 subtests passed**. |
| `python -m pytest -q tests/test_full7_contract.py tests/test_full7_feature_store.py tests/test_full7_score_audit.py tests/test_full7_checkpoint.py tests/test_full7_pipeline.py` | **23 passed**. |
| `python -m pytest -q tests/test_moe_gating.py tests/test_moe_forward_integration.py tests/test_moe_feature_store.py tests/test_moe_training_pipeline.py tests/test_moe_validation_inference.py tests/test_moe_checkpoint_resume.py tests/test_moe09_4_full9.py tests/test_set_a_residual.py tests/test_set_a_training.py tests/test_set_b_residual.py tests/test_set_b_cached_dataset.py tests/test_set_b_training.py tests/test_whisper_feature_cache.py` | **133 passed, 38 subtests passed**. |
| `python -m pytest -q tests` | **461 passed, 1 skipped, 1 warning, 260 subtests passed**. |
| `PYTHONPATH=/tmp/moe27-py-env python -m pytest -q` | **498 passed, 11 failed, 1 skipped, 2 warnings, 264 subtests passed**. Alias `/tmp/moe27-py-env/MOESCORE` trỏ vào worktree vì test expert cũ import tên gói `MOESCORE`. |
| `git diff --check` | PASS. |

`python` trong bảng là executable môi trường `nlp` nêu trên. Full suite trực
tiếp không có alias cho `MOESCORE` cho kết quả `467 passed, 27 failed,
1 skipped, 4 warnings, 260 subtests passed`, chủ yếu do import; alias giảm
số lỗi còn 11. Thử thêm toàn bộ thư mục expert vào `PYTHONPATH` gây xung đột
gói `model` và 38 lỗi collection; không dùng cách đó làm bằng chứng hồi quy.

11 lỗi còn lại thuộc file không sửa: một test E2 patch import `Train_Expert2`
không có trên module path; sáu test E3 patch import `models` không có trên module
path; ba test E4 notebook đòi
`expert4_finetune/colab/Finetune_MOESCORE_Expert4.ipynb` vắng mặt ở HEAD nền;
một test E4 workflow import `expert4_repro` không có trên module path. Những
test này không được sửa hay ẩn đi. Chưa có full-suite baseline trước sửa để
chứng minh toàn bộ 11 lỗi là tiền tồn tại; các import/asset gây lỗi nằm ngoài
phạm vi file đã thay đổi.

## E. Trạng thái và giới hạn

- **IMPLEMENTED AND SYNTHETICALLY VERIFIED:** Full7 registry, lắp ghép,
  builder, schema v3, score audit/review boundary, gate train/evaluate/resume,
  CLI, kiểm WAV trùng split và test CSV.
- **IMPLEMENTED BUT NOT REAL-DATA VERIFIED:** tương thích checkpoint/cache
  Full9/Whisper/E5/E6 thật; alignment thực; score E3 thực nghiệm; chất lượng
  validation và khả năng tổng quát hóa.
- **NOT IMPLEMENTED:** diagnostic artifact tùy chọn; raw-audio end-to-end
  inference (ngoài phạm vi V1); Core.6 huấn luyện thật.
- **BLOCKED BY EXTERNAL ARTIFACTS:** Full9 và Whisper cache xác minh đủ ba
  split; E5/E6 checkpoint thật với SHA-256, kiến trúc và provenance phù hợp;
  source WAV/manifest thật; audit score và kiểm tra số học các chuyên gia.
- **BLOCKED BY COMPETITION APPROVAL:** tài nguyên pretrained bên ngoài và
  chính sách scheduler theo validation loss, best/early-stop theo validation
  SRCC. Không tuyên bố mô hình đã hợp lệ để nộp.

## F. Điều kiện vào Core.6

1. Cung cấp Full9 store và Whisper large-v2 cache đã xác minh cho train,
   validation, test cùng source manifest và WAV byte gốc; xác minh không trùng
   WAV giữa split.
2. Cung cấp checkpoint E5/E6 thật được tin cậy, kiểm SHA-256, kiến trúc,
   feature/normalization contract và component identities bằng loader hiện có.
3. Build/reload Full7 v3 cho từng split; đối chiếu row count, thứ tự, SHA,
   nhãn train/validation và test không nhãn.
4. Chạy audit score **train-only**, kiểm E3 checkpoint/output thật và ý nghĩa
   E1/E2 similarity proxy; lập `data.score_review_file` với reviewer, bằng chứng,
   hash store/contract/audit và trạng thái đã rà cho E1–E7 trước khi tối ưu gate.
5. Có hướng dẫn bằng văn bản của ban tổ chức về pretrained resources và
   validation-driven scheduler/model selection; đối chiếu header CSV chính thức
   (`wave_file_name` theo yêu cầu Core, writer cũ dùng `wav_file_name`).
6. Sau đó mới chạy official train gate, validation evaluation, resume smoke
   test và test inference, ghi metric thực cùng run manifest.
