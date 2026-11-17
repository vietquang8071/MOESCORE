# Báo cáo MOE-09.3 — Mở rộng đặc trưng audio–text

## Kết quả

Đã thêm helper không có tham số học `build_paired_cross_features()` tại `model/expert_adapters/paired_cross_features.py`. Với hai embedding `[B,D]`, helper trả về `[a, t, |â−t̂|, â⊙t̂, cos(â,t̂)·1_D]` theo đúng thứ tự yêu cầu. `a` và `t` được giữ nguyên sau khi đổi sang float32; chỉ ba khối tương tác dùng L2-normalize. Helper kiểm tra shape, device, dtype và giá trị hữu hạn, đồng thời xử lý embedding zero mà không sinh NaN.

Đã cập nhật đúng sáu adapter: Expert 1, Expert 2, Top2 M2D, Top2 MS-CLAP, Top2 LAION và Top2 MGA. Các encoder chỉ chạy một lần mỗi lần gọi adapter. Expert 1 tiếp tục dùng raw dot product; Expert 2 trả nguyên score từ model gốc; bốn Top2 giữ cosine score từ `pack_embeddings()`. Không thêm score vào vector đặc trưng.

Expert 3, Expert 4 và M2D-LoRA không bị sửa adapter hoặc runtime; feature formula và score của chúng giữ nguyên.

## Kích thước đặc trưng

| Expert | Trước | Sau | Trạng thái score |
|---|---:|---:|---|
| `expert1` | 1024 | 2560 | Raw dot product giữ nguyên |
| `expert2` | 2048 | 5120 | Score gốc giữ nguyên |
| `expert3` | 3840 | 3840 | Không đổi |
| `expert4` | 1024 | 1024 | Không đổi |
| `top2_m2d` | 1536 | 3840 | Cosine giữ nguyên |
| `top2_msclap` | 2048 | 5120 | Cosine giữ nguyên |
| `top2_laion` | 1024 | 2560 | Cosine giữ nguyên |
| `top2_mga` | 2048 | 5120 | Cosine giữ nguyên |
| `m2d_lora` | 3073 | 3073 | CORAL/regression giữ nguyên |

Registry bốn expert hiện tại có tổng chiều rộng **12544**; hợp đồng score vẫn `[B,4]`, feature là `[B,12544]`. Registry giữ bốn expert và thứ tự cũ. Metadata chuẩn bị cho `full_9` có tổng chiều rộng **32257**, score `[B,9]`, feature `[B,32257]`; gate và pipeline full-9 chưa được tích hợp trong milestone này.

Input của gate bốn expert đổi từ 7936 sang 12544. Kiến trúc gate giữ nguyên: Linear → LayerNorm(512) → ReLU → Dropout(0.1) → Linear(4) → Softmax; fusion vẫn là tổng có trọng số của bốn score. Checkpoint gate cũ có input 7936 không tương thích và bị từ chối bởi kiểm tra kích thước. Cần huấn luyện lại gate cho feature store mới.

## Cache và feature store

Cache identity cho sáu expert thay đổi có chứa policy `paired_cross_v1`, tên đủ năm khối, hệ số chiều rộng 5 và SHA-256 của helper. Kiểm thử gọi `compute_expert_cache_identity()` thật với checkpoint và Hugging Face snapshot tạm, chứng minh identity đổi khi đổi representation, helper source hoặc nội dung BERT snapshot.

Expert 3, Expert 4 và M2D-LoRA không nhận policy/helper hash mới; test identity xác nhận cache ID của chúng không đổi khi chỉ bật/tắt policy cho sáu expert. Cache từng expert không bị ghi lại hoặc đổi nhãn. Shard cũ của sáu expert có dạng `[B,2D]` bị validator từ chối theo chiều rộng mới; không thể tái sử dụng như vector `[B,5D]`. Assembly Top1 mới có identity khác nhờ cache ID và feature dimension mới; artifact cũ được giữ nguyên.

Writer/reader, collate, validator, training/evaluation shape checks và gate input đều dùng chiều rộng Top1 mới. Test cũng xác nhận vector legacy 7936 bị từ chối. Cấu hình local mẫu được chỉnh đường dẫn checkpoint sang `weights/pretrained/`, khớp vị trí các file hiện có trong repository.

## Real inference

Đã chạy tuần tự trên CPU, với checkpoint thật, không chạy full-dataset. CUDA không khả dụng. Bốn Top2 dùng cùng cặp train `00000.wav` — “A water vehicle travels through the water with wind noise and a man talking”. Expert 1 và Expert 2 dùng cặp validation `07500.wav` — “Humming and accelerating of a car engine with squealing tires and people speaking faintly”. Nhãn không được đưa vào adapter hoặc phép tính score.

| Expert | Checkpoint đã nạp | Score quan sát | Score shape | Feature shape | Kết quả |
|---|---|---:|---|---|---|
| `expert1` | `weights/pretrained/laion-clap.pt` + `weights/finetuned_weights/expert1_best.pt` | 0.6266124844551086 | `[1,1]` | `[1,2560]` | PASS, score parity chính xác với Expert 1 gốc |
| `expert2` | `weights/pretrained/mga-clap.pt` + `weights/finetuned_weights/expert2_best.pt` | 0.4505678415298462 | `[1,1]` | `[1,5120]` | PASS, strict restore và score parity chính xác |
| `top2_m2d` | `weights/pretrained/checkpoint-30.pth` | 0.3516223430633545 | `[1,1]` | `[1,3840]` | PASS |
| `top2_msclap` | `weights/pretrained/CLAP_weights_2023.pth` | 0.6499422788619995 | `[1,1]` | `[1,5120]` | PASS |
| `top2_laion` | `weights/pretrained/laion-clap.pt` | 0.5418541431427002 | `[1,1]` | `[1,2560]` | PASS |
| `top2_mga` | `weights/pretrained/mga-clap.pt` | 0.4798489511013031 | `[1,1]` | `[1,5120]` | PASS |

Cả sáu lần chạy trả float32 hữu hạn trên CPU và đúng feature width đã khai báo. Verifier Expert 1 và Expert 2 so sánh score adapter với score gốc; sai số quan sát là 0. Expert 2 cần `NUMBA_CACHE_DIR=/tmp/moescore-moe093-numba` do package Numba cài đặt không có cache locator ghi được. Không cần tải hoặc thay model. Lệnh Expert 1 dùng `validation_average.csv` vì `validation.csv` chứa schema per-listener và không có cột `average_score` mà verifier yêu cầu.

## Kiểm thử

- Full suite: `python -m unittest discover -s tests -p 'test*.py' -q` — **331 test OK, 1 skip** (case cần CUDA; CUDA không có trong môi trường).
- Helper: 5 test về block order/value, cosine, embedding zero, dtype/device và input không hợp lệ.
- Cache representation: 4 test chạy qua identity thực, bao gồm sáu expert đổi ID, đổi helper/BERT snapshot và giữ ID ba expert không đổi.
- Adapter: Expert 1 22/22, Expert 2 24/24, Top2 7/7; Expert 3 20 pass/1 skip, Expert 4 25/25 và M2D-LoRA 8/8.
- Feature store 6/6; registry/coexistence 12/12; gate 9/9; unified integration 2/2; cache/resume MOE-09.2 18/18.
- `py_compile` cho module thay đổi và `git diff --check` đều thành công.

Real inference đã chạy cho cả sáu expert. Không chạy gate full-9, huấn luyện, hay trích xuất toàn bộ dataset.

## File thay đổi

- Helper mới: `model/expert_adapters/paired_cross_features.py`.
- Adapter: `model/expert_adapters/expert1_adapter.py`, `expert2_adapter.py`, `top2_m2d_adapter.py`, `top2_msclap_adapter.py`, `top2_laion_adapter.py`, `top2_mga_adapter.py`.
- Contract/gate/store: `model/expert_adapters/registry.py`, `model/moe_gating_core.py`, `data/moe_training_dataset.py`, `scripts/extract_moe_feature_store.py`, `scripts/build_moe_feature_store.py`, `scripts/validate_moe_feature_store.py`, `training/checkpoint_manager.py`, `training/moe_trainer.py`, `training/moe_evaluator.py`.
- Runtime verifier: `scripts/verify_expert1_runtime.py`, `scripts/verify_expert2_runtime.py`, `scripts/verify_real_expert_to_gate.py`.
- Config: `configs/moe_local.example.yaml`.
- Tests: `tests/test_paired_cross_features.py`, `tests/test_moe09_3_cache_identity.py`, cùng regression tests hiện có được cập nhật cho chiều rộng mới; fixture cache MOE-09.2.1 cũng bổ sung helper làm dependency.
- Báo cáo: `docs/Ex1/moe09_3_cross_feature_expansion_report.md`.

Không sửa các adapter/runtime Expert 3, Expert 4, M2D-LoRA, trọng số checkpoint, registry expert count, hoặc gating network full-9.
