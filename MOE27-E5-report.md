# Báo cáo sửa E5: tái sử dụng Full9 và holdout train-only

## Kết quả

E5 hiện lấy M2D-CLAP, MS-CLAP 2023 và MGA-CLAP trực tiếp từ **cùng một verified Full9 feature store** qua `SetACachedDataset(full9_root, split=...)`. MGA dùng đúng hợp đồng Full9: `expert2_wavs`, `[B,1,320000]`, 32 kHz, tối đa 320.000 mẫu. Đã xóa adapter MGA riêng cho E5, smoke script của adapter đó và bộ test ép MGA 16 kHz. Không cần tái trích xuất MGA hay tạo cache MGA riêng.

Kiến trúc và công thức điểm được giữ nguyên: từng component rộng 3.073 / 4.097 / 4.097; ghép M2D → MS-CLAP → MGA thành `[B,11267]`; điểm là affine trên trung bình cosine số học cộng residual MLP. Giao diện vẫn là `(score5, features5)` với `[B,1]`, `[B,11267]`. Không thêm gating, SVR hay compression.

## Nhánh, mã nguồn và tài nguyên

- Worktree: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE-E5`
- Nhánh: `dev/moe27-e5`
- Worktree được xác nhận thuộc MOESCORE-E5; không sửa nhánh integration hay worktree của agent khác.
- Source Top 2 đã kiểm tra: `/home/vitquay1708/Working_Space/Research/2027/code/EnsembleSVR-XACLE`, gồm `train.py`, `features/extract_features.py`, `features/all_feature_dict.py`, `features/proximity_features.py`, `load_pretrained_models/load_model.py`, `datasets/fetch_data.py`, `config_submission1.json`.
- Tài nguyên checkpoint đã định vị trong repo MOESCORE chính, không sao chép vào worktree: M2D `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/checkpoint-30.pth`; MS-CLAP 2023 `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/CLAP_weights_2023.pth`; MGA `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/mga-clap.pt`.
- Không tìm thấy verified Full9 feature store tại các vị trí local đã kiểm tra. Vì vậy kiểm chứng runtime ở tác vụ này dùng Full9 store synthetic trên CPU; không có real-cache smoke test hay train thật.

## Thay đổi

- `data/set_a_cached_dataset.py`: dùng Full9 trực tiếp; xác nhận MGA 32 kHz/320.000 mẫu và `expert2_wavs`; giữ kiểm tra provenance, thứ tự sample, WAV và feature 4D+1; dùng một helper so nhãn sau khi đổi sang float32, kiểm tra hữu hạn và miền `[0,10]`.
- `training/set_a_trainer.py`, `training/set_a_config.py`, `scripts/train_set_a_residual.py`: thêm split train-only xác định, MSE fit/holdout riêng, best checkpoint, early stopping và resume có kiểm tra split, seed, cấu hình, provenance và hash best checkpoint.
- `scripts/infer_set_a_residual.py`: dùng resolver thiết bị chung; `auto` được phân giải trước khi gọi `torch.device`, hỗ trợ `cpu`/`cuda`, báo lỗi rõ khi CUDA không khả dụng hoặc device sai.
- `configs/set_a_residual.example.yaml`, `configs/set_a_residual.local.yaml`: dùng `${MOESCORE_FULL9_FEATURE_STORE}` và `${MOESCORE_E5_OUTPUT}`, bỏ đường dẫn checkpoint và cấu hình MGA riêng.
- `tests/test_set_a_residual.py`, `tests/test_set_a_training.py`: thêm kiểm tra Full9 MGA, decimal labels, provenance, split, resume, early stopping, best selection, device auto và inference CLI.
- `docs/MOE27-E5-implementation.md`: cập nhật hợp đồng, khác biệt preprocessing, lệnh vận hành và giao diện Full7.
- Đã xóa `model/expert_adapters/set_a_mga_source_adapter.py`, `scripts/smoke_test_set_a_mga_source.py`, `tests/test_set_a_source_compat.py` vì chỉ phục vụ luồng MGA riêng cũ.
- Giữ thay đổi inference `device: auto` chưa commit đã có trong worktree lúc bắt đầu; refactor gọi resolver trainer dùng chung thay vì lặp logic.

Không sửa registry, helper 4D+1 dùng chung, các adapter Top2 dùng chung, Full9 extractor, Global MoE, E1–E4, E6/E7 hoặc Kaggle notebook.

## Label, holdout, checkpoint

So sánh label dùng cùng biểu diễn float32 của cache, tránh false mismatch với `6.7`, `4.43`, `5.9`, `7.1`, `0.0`, `10.0`; label thay đổi thật vẫn bị từ chối.

Holdout mặc định là 15% dữ liệu chính thức `train`, seed 42. Manifest Full9 hiện tại không có `group_id`, nên fallback xác định theo SHA256 của sample ID và seed. Nếu toàn bộ manifest có `group_id`, splitter chia nguyên nhóm; thiếu/malformed group metadata bị từ chối. Hai phần fit/holdout phải không rỗng và không giao nhau. Không dùng validation hoặc test labels để tối ưu, dừng sớm hay chọn checkpoint.

Sau mỗi epoch, trainer ghi `train_mse`, `holdout_mse`, `best_holdout_mse`, learning rate; cập nhật `expert5_latest.pt` để resume và `expert5_best.pt` khi holdout MSE giảm hơn `min_delta`. Mặc định patience là 5 epoch; `null` tắt early stopping. `expert5_final.pt` được định nghĩa rõ là bản sao của checkpoint tốt nhất theo holdout, còn `latest` là epoch cuối đã chạy. Resume phục hồi optimizer, epoch/global step, best metric/epoch, tiến độ patience, seed và split; xác thực cache/provenance, cấu hình train và hash của `expert5_best.pt`. Tăng tổng `epochs` được phép; các thiết lập khác phải khớp.

Checkpoint chỉ chứa E5 head/optimizer và metadata; không chứa tensor của ba CLAP backbones.

## Khác biệt với source Top 2

E5 cố ý sử dụng preprocessing MOESCORE Full9. MGA của source Top 2 đọc/pad/cắt 16 kHz và 160.000 mẫu, còn Full9 MGA nhận 32 kHz và 320.000 mẫu. MS-CLAP cũng theo crop giữa xác định của adapter MOESCORE thay vì lựa chọn ngẫu nhiên đoạn bảy giây trong source. Vendor M2D/MGA cần để chạy source song song không có đầy đủ trong source repo đã kiểm tra, nên không tuyên bố parity số học giữa embeddings source và Full9. Việc chuyển sang Full9 là quyết định kiến trúc hiện hành, không phải khẳng định output tương đương tuyệt đối.

## Kiểm thử

Python môi trường kiểm thử: `/home/vitquay1708/miniconda3/envs/nlp/bin/python`.

- `python -m unittest discover -s tests -p 'test_set_a_*.py' -v`: **33 đạt**.
- `test_paired_cross_features.py`: **5 đạt**.
- `test_top2_adapters.py`: **7 đạt**.
- `test_moe_feature_store.py`: **6 đạt**.
- `test_moe09_4_full9.py`: **4 đạt**.
- Toàn bộ `python -m unittest discover -s tests -v`: **382 test, 380 đạt, 1 skip, 1 fail**. Lỗi duy nhất là `test_lightweight_diagnostic_emits_json_and_marks_real_checks_unverified` trong `tests/test_expert_factory_coexistence.py`: preflight yêu cầu các checkpoint E1–E4 không có trong worktree E5 tách biệt, trả `BLOCKED_MISSING_RESOURCE`, còn test mong exit code 0. Đây là lỗi môi trường/tài nguyên ngoài phạm vi E5; không sửa test hoặc checkpoint để che lỗi.
- CUDA không khả dụng trong phiên chạy; không kiểm thử CUDA.
- Synthetic CPU end-to-end đã chạy: tạo Full9 store → đọc ba component → train với holdout → lưu latest/best/final → resume/restore → chạy inference CLI với `device: auto`.

## Lệnh train, resume và inference

```bash
export MOESCORE_FULL9_FEATURE_STORE=/path/to/verified/full9_feature_store
export MOESCORE_E5_OUTPUT=/path/to/e5_runs
python scripts/train_set_a_residual.py --config configs/set_a_residual.example.yaml
```

```bash
python scripts/train_set_a_residual.py \
  --config configs/set_a_residual.example.yaml \
  --resume "$MOESCORE_E5_OUTPUT/expert5_latest.pt"
```

```bash
python scripts/infer_set_a_residual.py \
  --config configs/set_a_residual.example.yaml \
  --checkpoint "$MOESCORE_E5_OUTPUT/expert5_best.pt" \
  --split test \
  --output "$MOESCORE_E5_OUTPUT/test_scores.csv"
```

## Hợp đồng Full7 và trạng thái tích hợp

Full7 truyền đúng mapping `batch["component_outputs"]` với ba khóa `top2_m2d`, `top2_msclap`, `top2_mga`; mỗi component cung cấp `score: float32[B,1]` và `features: float32[B,4D+1]`. Tất cả cùng batch size/device và feature theo năm block hiện hành. E5 trả `(score5, features5)` với `[B,1]` và `[B,11267]`; vector nối nguyên vẹn này là input cho Global Gating Network. Full7 Core cần thêm adapter E5 vào khởi tạo nội bộ và assembly tại đúng thứ tự/dimension đã thống nhất; tác vụ này không đăng ký E5 toàn cục.

- Commit SHA: sẽ được điền sau khi commit trên `dev/moe27-e5`.
- Trạng thái: synthetic/regression verification đã đạt; sẵn sàng tích hợp sau khi cung cấp verified Full9 store và lưu ý lỗi test toàn repo nêu trên. Chưa xác minh trên cache thật.
