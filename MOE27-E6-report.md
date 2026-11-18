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

Không sửa E5, Whisper extractor, LAION adapter, Full9 legacy, registry hay Global Gating Network. Đoạn này mô tả lần triển khai ban đầu; E6 đã được commit ở `ffb6c77` trước đợt sửa P1 bên dưới.

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

**Trạng thái sau lần triển khai ban đầu:** triển khai E6 và kiểm thử tổng hợp hoàn tất; chờ cache/checkpoint thật để xác minh runtime và huấn luyện official. Trạng thái Git hiện tại được cập nhật trong mục P1 bên dưới.

## 8. Sửa hai lỗi P1 trước Integration (2026-11-19)

### Trạng thái đầu đợt sửa

Worktree `MOESCORE-E6` ở nhánh `dev/moe27-e6`, HEAD `ffb6c77 feat(e6): add Set B residual expert pipeline`. `git status --short` rỗng; `git merge-base --is-ancestor feature/xacle2027-7expert-moe HEAD` trả mã 0. Không có thay đổi cục bộ cần bảo tồn.

### P1-1 — Nhất quán identity LAION

`training/set_b_trainer.py` nay dùng `validate_expert_cache_identity` để xác thực identity LAION train và validation, sau đó so canonical `cache_id`. Kiểm tra này ràng buộc implementation, preprocessing, input, inference policy, score/feature contract, dependencies và checkpoint provenance. Cùng checkpoint nhưng khác implementation/preprocessing bị từ chối trước optimizer step. Khác biệt store/source manifest riêng theo split vẫn được chấp nhận.

Khi nạp checkpoint cho split chưa nằm trong checkpoint (như `test`), loader yêu cầu provenance train đã lưu, xác thực cả identity LAION train và identity LAION của split suy luận, so `cache_id` trước khi nạp model parameters. Nếu thiếu, sai schema/hash hoặc khác cache, loader từ chối. Kiểm tra Whisper checkpoint/preprocessing và exact provenance đối với train/validation được giữ nguyên.

### P1-2 — WAV trùng byte giữa train/validation

Trainer đọc `wav_sha256` từ hai Full9 source manifest đã xác thực, chuẩn hóa hex rồi tìm giao hai tập hash bằng map/set, không so từng cặp record. Nếu cùng byte WAV xuất hiện ở hai split, trainer báo hash và cả sample ID train/validation trước khi tạo output directory hoặc chạy optimizer. Quy tắc này vẫn áp dụng khi tên file, ID và caption khác nhau. Không xóa/lọc/sắp lại dữ liệu official. Trùng byte cần được review theo thiết kế bộ dữ liệu. Audio tương đương về nội dung nhưng đã mã hóa lại có thể có SHA-256 khác và không bị phát hiện bởi phép kiểm này.

### Tệp thay đổi và kiểm thử

Chỉ sửa `training/set_b_trainer.py`, `tests/test_set_b_training.py`, `tests/test_set_b_cached_dataset.py`, `docs/MOE27-E6-implementation.md` và báo cáo này. Fixture tổng hợp tạo lại Full9 và Whisper manifest/cache hợp lệ với các hash identity chuẩn; không vô hiệu hóa kiểm tra hash để tạo test.

Test mới gồm: cùng identity qua hai split có source/store fingerprint khác nhau; implementation hoặc preprocessing LAION khác dù cùng checkpoint; test split LAION khác; provenance LAION đã lưu bị thiếu/sai; identity inference trong RAM sai; WAV trùng với ID/tên/caption khác; không có optimizer step, không tạo output và không thay dữ liệu khi phát hiện overlap; SHA-256 không phải hex bị từ chối. Trước sửa, 7 biến thể P1 fail đúng vì thiếu kiểm tra; thêm một ca riêng cho SHA-256 không phải hex đã fail trước khi siết validation.

- E6 tập trung sau sửa: **31 passed** (`tests/test_set_b_residual.py`, `tests/test_set_b_cached_dataset.py`, `tests/test_set_b_training.py`).
- Hồi quy Full9, paired LAION, Top2, Whisper extractor/cache và E5: **77 passed**.
- Full suite trên mã đã sửa: **458 passed, 27 failed, 1 skipped, 2 warnings** trong 76,16 giây. 27 lỗi giữ nguyên nhóm chữ ký so với báo cáo trước: test E2/E3 import tên package `MOESCORE` không có ở worktree `MOESCORE-E6`; test E4 thiếu notebook `expert4_finetune/colab/Finetune_MOESCORE_Expert4.ipynb` hoặc package `expert4_repro`. Không có lỗi E6. Full suite được chạy trước khi bổ sung một test cuối về hex viết hoa; bộ E6 đã chạy lại toàn bộ sau bổ sung này.

Lệnh E6: `conda run -n nlp env PYTHONPATH=/home/vitquay1708/.cache/uv/archive-v0/UfEXJ4HVLkLakVBe/lib/python3.12/site-packages python -m pytest tests/test_set_b_residual.py tests/test_set_b_cached_dataset.py tests/test_set_b_training.py -q --tb=short`. Lệnh hồi quy dùng cùng môi trường với `tests/test_whisper_top2.py tests/test_whisper_feature_cache.py tests/test_moe_feature_store.py tests/test_paired_cross_features.py tests/test_set_a_residual.py tests/test_set_a_training.py tests/test_top2_adapters.py`. Full suite dùng `python -m pytest -q --tb=short` trong cùng môi trường.

Không chạy dữ liệu thật, không tải backbone, không commit/push/merge trong đợt sửa P1. Việc xác minh với cache thực vẫn **NOT VERIFIED**; caveat về official validation checkpoint selection và pretrained disclosure ở mục 6 vẫn còn hiệu lực.

Trạng thái Git cuối đợt sửa: `dev/moe27-e6` đồng bộ với `origin/dev/moe27-e6`; đúng 5 tệp E6 nêu trên đã sửa nhưng chưa stage, không có tệp mới và không có commit mới. `git diff --check` trả mã 0.
