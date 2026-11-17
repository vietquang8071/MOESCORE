# Báo cáo triển khai MOE27-E5

## 1. Worktree và nhánh

- Worktree: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE-E5`
- Nhánh: `dev/moe27-e5`
- Base hiện tại khi bắt đầu: `4a1008effa220875c1d83a755606aadea3939704` (`Merge branch 'fix/moe09-4d-plus-one'`)
- Thay đổi chỉ được thực hiện trong worktree E5; không đổi nhánh, reset, push hay merge.

## 2. Mã nguồn và checkpoint đã kiểm tra

Đã đọc mã Top 2 trong `/home/vitquay1708/Working_Space/Research/2027/code/EnsembleSVR-XACLE`: `train.py`, `features/extract_features.py`, `features/all_feature_dict.py`, `features/proximity_features.py`, `load_pretrained_models/load_model.py`, `datasets/fetch_data.py`, và `config_submission1.json`.

Ba checkpoint tìm được trong repo MOESCORE chính, không sao chép vào worktree E5:

- M2D-CLAP: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/checkpoint-30.pth`
- MS-CLAP 2023: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/CLAP_weights_2023.pth`
- MGA-CLAP: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/mga-clap.pt`

Các file triển khai và YAML MGA hiện có được dùng từ `expert2_finetune/MGA-CLAP` trong worktree MOESCORE-E5. Đường dẫn checkpoint tuyệt đối được ghi trong `configs/set_a_residual.local.yaml`; cấu hình portable dùng biến môi trường để Kaggle có thể chỉ định resource root khác.

## 3. File đã tạo

- E5 head và adapter: `model/composite_experts/set_a_residual.py`, `model/expert_adapters/set_a_residual_adapter.py`.
- Adapter MGA chỉ dành cho E5, giữ preprocessing Top 2: `model/expert_adapters/set_a_mga_source_adapter.py`.
- Cache/dataset: `data/set_a_cached_dataset.py`.
- Cấu hình và trainer: `training/set_a_config.py`, `training/set_a_trainer.py`.
- Entry point: `scripts/train_set_a_residual.py`, `scripts/infer_set_a_residual.py`.
- Smoke test có checkpoint thật: `scripts/smoke_test_set_a_mga_source.py`.
- Cấu hình: `configs/set_a_residual.example.yaml`, `configs/set_a_residual.local.yaml`.
- Kiểm thử: `tests/test_set_a_residual.py`, `tests/test_set_a_source_compat.py`, `tests/test_set_a_training.py`.
- Tài liệu kỹ thuật: `docs/MOE27-E5-implementation.md`.
- Kế hoạch làm việc: `docs/superpowers/plans/2026-11-18-moe27-e5.md`.

Không sửa adapter E1–E4, adapter Top2 dùng chung, registry, bộ ghép feature toàn cục, gating, E7 hay notebook/Kaggle pipeline.

## 4. Kiến trúc và tensor

- M2D-CLAP: `D=768`, feature `[B,3073]`.
- MS-CLAP: `D=1024`, feature `[B,4097]`.
- MGA-CLAP: `D=1024`, feature `[B,4097]`.
- Ghép đúng thứ tự M2D, MS-CLAP, MGA: `features5` có dạng `[B,11267]`.
- Mỗi vector component dùng năm khối `[audio, text, abs(normalize(audio)-normalize(text)), normalize(audio)*normalize(text), cosine_scalar]`; helper hiện hữu cho đúng 4D+1, cosine là một scalar.
- Điểm E5 là `scale * ((s_m2d+s_msclap+s_mga)/3) + bias + residual_mlp(features5)`. Calibrator khởi tạo `scale=1`, `bias=0`; lớp cuối residual khởi tạo bằng 0.
- `SetAResidualAdapter` giữ giao diện `ExpertAdapter` `(score, features)` với `[B,1]` và `[B,11267]`. Backbone bị đóng băng và không nằm trong optimizer/checkpoint E5.

## 5. Tái sử dụng implementation 4D+1

Worktree đã có commit `c15cd10` cập nhật helper và các adapter Top2 sang 4D+1, nên E5 tái sử dụng `build_paired_cross_features` hiện hữu, không tạo helper trùng lặp.

## 6. Kết quả đối chiếu mã nguồn Set A

- M2D dùng `PortableM2D`, `flat_features=True`, `encode_clap_audio/text`, đầu ra rộng 768. Kiểm tra độc lập trực tiếp với vendor M2D của repo Top 2 không thực hiện được vì thư mục vendor `m2d` không có trong source repo cục bộ.
- MS-CLAP dùng API path-based và checkpoint 2023, đầu ra rộng 1024. Source Top 2 chọn audio segment ngẫu nhiên 7 giây, còn adapter MOESCORE hiện tại dùng crop giữa 7 giây; đây là khác biệt preprocessing, không khẳng định parity số học.
- MGA dùng ASE, `msc()` theo codebook và chuẩn hóa của source. Adapter chung MOESCORE nhận waveform 32 kHz (`expert2_wavs`), trong khi Top 2 `fetch_data.py` truyền waveform giả định 16 kHz đã pad/cắt 160.000 mẫu. Vì vậy E5 có `SetAMGASourceAdapter` riêng dùng `expert3_wavs`, không sửa adapter dùng chung.
- Checkpoint MGA khôi phục 415/415 tensor khớp, có 1 key thừa. Source repo Top 2 thiếu vendor `MGA-CLAP` và `examples`, do đó không thể instantiate riêng toàn bộ source stack để so sánh embedding từng phần.
- Smoke test real checkpoint một mẫu train: M2D score `0.35162234`, `[1,3073]`; MS-CLAP score `0.64994228`, `[1,4097]`; MGA adapter chung 32 kHz score `0.47984895`, `[1,4097]`; MGA E5 16 kHz score `0.30485559`, `[1,4097]`. Tất cả output kiểm tra đều hữu hạn, float32, CPU, model ở eval và frozen. Hai score MGA khác nhau do input contract khác nhau; đây không phải phép so sánh parity với source độc lập.

## 7. Lệnh train E5

Chưa có cache feature trong worktree lúc triển khai. Trước khi train, cần cung cấp full9 store đã xác minh và output MGA source-compatible 16 kHz, ghép thành E5 cache theo hướng dẫn trong tài liệu kỹ thuật, sau đó cập nhật `feature_store_root` của local config. Lệnh train:

```bash
python scripts/train_set_a_residual.py --config configs/set_a_residual.local.yaml
```

Resume:

```bash
python scripts/train_set_a_residual.py \
  --config configs/set_a_residual.local.yaml \
  --resume /path/to/expert5_latest.pt
```

Không khởi chạy trích xuất toàn bộ dữ liệu hay train thật trong tác vụ này. CLI được kiểm thử trên cache synthetic CPU; chưa thể chạy trên dữ liệu thật vì chưa có feature cache tương thích.

## 8. Nạp checkpoint và chạy inference

Lệnh inference trên cache test:

```bash
python scripts/infer_set_a_residual.py \
  --config configs/set_a_residual.local.yaml \
  --checkpoint /path/to/expert5_final.pt \
  --split test \
  --output /path/to/expert5_test_scores.csv
```

Checkpoint chứa trạng thái E5 và optimizer, cấu hình kiến trúc, hợp đồng feature, epoch/step, seed, provenance dữ liệu/cache/component, cùng metadata chuẩn hóa score. Checkpoint không chứa trọng số backbone.

## 9. Lệnh kiểm thử và kết quả

Lệnh E5: `/home/vitquay1708/miniconda3/envs/nlp/bin/python -m unittest discover -s tests -p 'test_set_a_*.py' -v` — **27/27 đạt**.

Các suite hiện hữu đạt:

- `test_paired_cross_features.py`: 5/5.
- `test_top2_adapters.py`: 7/7.
- `test_moe_feature_store.py`: 6/6.
- `test_moe09_4_full9.py`: 4/4.

Toàn bộ suite: `/home/vitquay1708/miniconda3/envs/nlp/bin/python -m unittest discover -s tests -v` — **376 test, 374 đạt, 1 skip, 1 fail**. Test fail là `test_lightweight_diagnostic_emits_json_and_marks_real_checks_unverified` trong `test_expert_factory_coexistence.py`: script preflight nhìn thấy các checkpoint bị thiếu trong worktree tách biệt, trả trạng thái `BLOCKED_MISSING_RESOURCE` và exit 1; test hiện tại lại đòi `PASS`. Không có file preflight, registry hay adapter E1–E4 nào trong diff E5. Đây là giới hạn test/resource hiện hữu được giữ nguyên, không bị che giấu bằng sửa test.

CUDA không khả dụng nên không chạy train CUDA.

## 10. Commit

Commit triển khai trên nhánh `dev/moe27-e5`: `ac837adc7e530a0d23c208bbbfef4c135d74fe4b` (`feat(e5): implement Set A cosine residual expert`). Báo cáo ghi lại SHA commit triển khai này; bản cập nhật SHA của báo cáo nằm trong commit tài liệu tiếp theo.

## 11. Integration blockers còn lại

Worktree chưa có feature cache E5 tương thích; cần full9 cache cùng MGA source-compatible 16 kHz rows trước khi chạy training thật. Bộ vendor M2D/MGA trong repo Top 2 không hiện diện để đo parity embedding độc lập. CUDA không khả dụng trong môi trường này.

## 12. Hợp đồng chính xác cho Full7 Core

Full7 gọi `score5, features5 = expert5(batch)`, trong đó `batch["component_outputs"]` có đúng ba khóa `top2_m2d`, `top2_msclap`, `top2_mga`; mỗi giá trị có `score: float32[B,1]` và `features: float32[B,4D+1]`. Các tensor phải cùng batch/device và feature từng component phải theo đúng năm block. E5 trả score `[B,1]` cùng feature nối nguyên vẹn `[B,11267]`. Full7 cần thêm E5 vào cấu hình/khởi tạo expert và global feature-store assembly theo thứ tự và dimension đã thống nhất; E5 chưa được đăng ký vào registry trong commit này.
