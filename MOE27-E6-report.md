# Báo cáo MOE27-E6 — Set B Residual Expert

## 1. Git và trạng thái ban đầu

- Worktree: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE-E6`.
- Nhánh: `dev/moe27-e6`; HEAD ban đầu `1d03682`.
- Trước khi sửa: `git status --short` rỗng. Đã fetch `origin/feature/xacle2027-7expert-moe` và `origin/dev/moe27-e6` sau khi được cấp quyền ghi metadata Git ở worktree cha. `git merge-base --is-ancestor origin/feature/xacle2027-7expert-moe HEAD` trả mã 0, nên không cần merge.
- `e6-overview-v2.md` không có trong worktree; `e6-overview.md` hiện có và ghi quyết định v2. `Task-rules.txt` không có trong worktree. Cảnh báo quy chế dưới đây dựa trên phần trích trong `e6-overview.md`.

## 2. Tệp tạo mới

- Mô hình: `model/composite_experts/set_b_residual.py`, `model/expert_adapters/set_b_residual_adapter.py`.
- Dữ liệu: `data/set_b_cached_dataset.py`.
- Huấn luyện/cấu hình: `training/set_b_config.py`, `training/set_b_trainer.py`, `configs/set_b_residual.example.yaml`, `scripts/train_set_b_residual.py`.
- Suy luận: `scripts/infer_set_b_residual.py`.
- Kiểm thử: `tests/test_set_b_residual.py`, `tests/test_set_b_cached_dataset.py`, `tests/test_set_b_training.py`.
- Tài liệu: `docs/MOE27-E6-implementation.md`, báo cáo này.

Không sửa E5, Whisper extractor, LAION adapter, Full9 legacy, registry hay Global Gating Network. Không commit/push/merge nhánh.

## 3. Kiến trúc và dữ liệu

E6 ghép `top2_laion [2049] → Whisper audio [1280] → Whisper text [1280]`, đầu ra `features6 float32 [B,4609]`. Head dùng `scale_B * laion_cosine + bias_B + residual_mlp(features6)`, khởi tạo scale/bias là 1/0 và lớp cuối MLP bằng zero. Backbone output được detach; optimizer chỉ có tham số head. Adapter xác nhận toàn bộ 5 block LAION bằng helper `build_paired_cross_features`, đối chiếu cosine với score, kiểm tra shape, dtype, device và finite.

Dataset chỉ đọc `top2_laion` từ Full9 đã kiểm tra hash/provenance và hai vector từ Whisper cache đã kiểm tra manifest/shard SHA-256. Join yêu cầu cùng split, ID và thứ tự đúng tuyệt đối. Fingerprint Whisper được dựng lại từ source manifest Full9 bằng chính canonical JSON, SHA-256 caption gốc, SHA-256 byte WAV và dấu xuống dòng từng record. Có thể cung cấp `source_audio_dirs` để xác nhận file WAV thực tế. Dataset kiểm tra checkpoint LAION, model Whisper large-v2, extraction batch-padding identity, hash mã unified preprocessing, kích thước và nhãn. Train/validation dùng nhãn raw `[0,10]`, chuyển sang `(raw-5)/5`; test không nạp nhãn. Không có logic model-ID hay cache E6 riêng.

## 4. Huấn luyện và checkpoint

Protocol cố định: 100% official train để tối ưu, 100% official validation để chọn checkpoint, đúng 30 epoch, AdamW/MSE, LR `1e-3`, weight decay `1e-4`, batch 8, seed 42, không holdout và không early stopping. Validation chạy sau mỗi epoch với `eval()`/`inference_mode()`, tính tổng lỗi bình phương chia đúng tổng mẫu. `expert6_best.pt` giữ epoch có validation MSE nhỏ nhất với quy tắc cải thiện nghiêm ngặt; hòa giữ epoch trước. `expert6_latest.pt` giữ trạng thái optimizer và resume; `expert6_final.pt` giữ chính model epoch 30, không chép best. Các tệp checkpoint được ghi nguyên tử, chứa contract, identity/provenance train và validation, cấu hình, metric, epoch/step, lịch sử và SHA-256 của best để kiểm tra resume. Không chứa tensor backbone.

Resume từ `expert6_latest.pt` trong cùng output directory kiểm tra cấu hình, cả hai provenance, contract, selection policy và hash best; chạy tiếp từ epoch kế tới tổng epoch 30. Suy luận CSV giữ thứ tự và ghi `score_normalized`, `score_raw=5*score_normalized+5`, không clamp.

## 5. Bằng chứng kiểm thử

- E6 tập trung: **20 passed** với `tests/test_set_b_residual.py`, `tests/test_set_b_cached_dataset.py`, `tests/test_set_b_training.py` (CPU, cache tổng hợp). Bài end-to-end dựng Full9/Whisper cache hợp lệ, train 30 epoch, kiểm tra best/latest/final, ngắt và resume từ epoch 2, chạy inference CLI ra CSV có thứ tự. Bài tie xác nhận best ở epoch 1 còn final ở epoch 30.
- Hồi quy Whisper, Full9, paired features, Top2 và E5: **77 passed**.
- Toàn repo tại thời điểm chạy: **440 passed, 27 failed, 1 skipped, 4 warnings** trong 91,11 giây. 27 lỗi thuộc test E2/E3 import `MOESCORE` theo tên thư mục gốc không có ở worktree `MOESCORE-E6`, và test E4 thiếu notebook `expert4_finetune/colab/Finetune_MOESCORE_Expert4.ipynb` hoặc package `expert4_repro`. Không sửa test ngoài E6 để che các lỗi này. Lần full suite diễn ra trước khi thêm vài ca E6 cuối; bộ E6 đã được chạy lại riêng.

Lệnh kiểm thử dùng `conda run -n nlp env PYTHONPATH=/home/vitquay1708/.cache/uv/archive-v0/UfEXJ4HVLkLakVBe/lib/python3.12/site-packages python -m pytest ... -q`. Môi trường shell mặc định không có `python`; `/usr/bin/python3` không có pytest.

## 6. Xác minh dữ liệu thật và rủi ro

**Real-cache alignment: NOT VERIFIED. Real 30-epoch training: NOT VERIFIED.** Tìm trong worktree và các checkout tại `/home/vitquay1708/Working_Space/Research/2027/code` không thấy Full9/Whisper manifest tương thích, `expert6_*.pt` hay Whisper-large-v2 weights. Trong worktree này còn không có thư mục `weights`; `result/whisper_feature_cache` và `result/moe_feature_store` không tồn tại. Không tải model lớn, không chạy extraction hay huấn luyện dữ liệu thật. Test tổng hợp không chứng minh parity số học với checkpoint thật.

Chọn `expert6_best.pt` theo nhãn official validation là **model selection**. Phần trích `Task-rules.txt` trong `e6-overview.md` chưa làm rõ hành vi này có được phép để nộp thi hay không. Cần xác nhận bằng văn bản với ban tổ chức trước khi dùng best cho submission. Cũng cần xác nhận việc khai báo/chấp thuận pretrained LAION và Whisper cùng dữ liệu ngoài theo quy chế.

## 7. Hướng dẫn Full7 Core và trạng thái

Core cần tạo `SetBResidualAdapter(SetBResidualHead(...))`, nạp checkpoint E6 đã kiểm tra provenance, truyền `component_outputs.top2_laion.{score [B,1], features [B,2049]}`, `whisper_audio [B,1280]`, `whisper_text [B,1280]`, rồi ghép `(score6 [B,1], features6 [B,4609])` vào vị trí E6 của assembly/gating Full7. Nhánh E6 chưa sửa registry hay kích thước global gating.

**Trạng thái cuối:** triển khai E6 và kiểm thử tổng hợp hoàn tất; chờ cache/checkpoint thật để xác minh runtime và huấn luyện official. Các tệp E6 mới còn untracked để code review; không commit.
