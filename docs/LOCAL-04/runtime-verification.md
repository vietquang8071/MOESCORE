# LOCAL-04.1 — Báo cáo xác minh runtime Expert 1

## 1. Tóm tắt

**Trạng thái: `PASS — REAL CHECKPOINT & NUMERICAL PARITY VERIFIED`.**

Theo kết quả runtime mới, LAION-CLAP pretrained checkpoint và fine-tuned Expert 1 checkpoint đều được nạp thành công; fine-tuned state dict được khôi phục strict với 0 missing key và 0 unexpected key. So sánh standalone Expert 1 và `Expert1Adapter` đạt parity cho waveform tổng hợp B=1/B=2 và hai mẫu validation thật. Score, audio embedding và text embedding có max absolute difference bằng 0. Feature giữ thứ tự audio trước, text sau; mỗi adapter forward gọi encoder đúng một lần.

Preprocessing gốc trên hai audio validation đã chạy thành công. Đây là kiểm tra subset nhỏ, không phải kiểm tra toàn validation set.

Không chạy training, optimizer, backward, hoặc Expert 2–4. Checkpoint và source model không bị sửa.

## 2. Môi trường

| Mục | Quan sát |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.1+cu130 |
| LAION-CLAP | 1.1.7 |
| Thiết bị | CUDA `cuda:0`, NVIDIA GeForce GTX 1650 |
| Precision | `torch.float32`; không autocast/half precision |
| Chế độ | Offline |
| CUDA memory trước khi nạp | 3,734,241,280 / 3,895,132,160 byte trống/tổng |
| CUDA peak allocated | 726,977,536 byte |
| Seed | 2027 |

### RoBERTa cache

Lần chạy mới ghi nhận cache mặc định `/home/vitquay1708/.cache/huggingface/hub` có `config.json`, model weights và tokenizer; LAION-CLAP dựng model được trong offline mode. Khác biệt repo ID/cache từng gây lỗi ở lần trước đã được khắc phục. Một lần chẩn đoán trước đó đã xác minh `RobertaModel.from_pretrained("roberta-base", local_files_only=True)` qua alias cache tạm: model type `roberta`, hidden size 768, 124,645,632 parameters. Lần chạy mới không cần alias tạm.

Transformers có cảnh báo masked-LM head weights không dùng và pooler weights thiếu khi tải vào `RobertaModel`; pretrained LAION-CLAP và full fine-tuned Expert 1 sau đó vẫn được nạp thành công.

## 3. Tài nguyên đã kiểm tra

| Resource | Đường dẫn | Kích thước | Kết quả |
|---|---|---:|---|
| LAION-CLAP pretrained | `weights/laion-clap.pt` | 1,863,587,645 byte | Nạp thành công |
| Expert 1 fine-tuned | `weights/finetuned_weights/expert1_best.pt` | 635,384,245 byte | Strict restore thành công |

Verifier ghi nhận kích thước và mtime trước/sau không đổi (`resources_unchanged=true`). Không tải hoặc sửa checkpoint.

Artifact fine-tuned là raw full `Expert1.state_dict()`. Strict-load xác nhận tương thích cấu trúc/tensor với model được dựng, nhưng artifact không có fingerprint pretrained-base nên không tự xác minh provenance của base checkpoint dùng trong lần fine-tuning.

## 4. Khôi phục Expert 1 gốc

Factory hiện có `Expert1Adapter.from_finetuned(...)` dựng model một lần, nạp pretrained CLAP, rồi strict-load fine-tuned state dict.

| Kiểm tra | Kết quả |
|---|---|
| Dựng original Expert1 | PASS |
| Nạp pretrained CLAP | PASS |
| Strict-load fine-tuned state | PASS |
| Missing / unexpected keys | 0 / 0 |
| Shape incompatibility | Không có |
| Pretrained-only fallback | Không có |

## 5. Kích thước embedding thực tế

| Giá trị | Quan sát |
|---|---|
| CLAP `joint_embed_shape` | 512 |
| Audio embedding | `[B,512]` |
| Text embedding | `[B,512]` |
| Adapter `embedding_dim` | 512 |
| Adapter `feature_dim` | 1024 |
| Adapter feature | `[B,1024]` |
| `feature_dim == 2 * D_CLAP` | PASS |

Metadata khớp kích thước embedding thực tế trong cả ba case: synthetic B=1, synthetic B=2 và validation B=2.

## 6. Input inference

Synthetic inputs được tạo với seed 2027, float32, biên độ nhiễu 0.05, và được chuyển lên `cuda:0`:

- B=1: `[1,1,480000]`
- B=2: `[2,1,480000]`

Captions nguyên văn, thứ tự và audio tensor được giữ giống nhau giữa standalone và adapter. Adapter không resample/crop/pad. Inference dùng eval mode và `torch.no_grad()`.

Validation batch được tạo qua `Expert1Dataset(training=False)` và `collate_batch()` gốc, không qua preprocessing thay thế.

## 7. Standalone và adapter score parity

Tolerance: `atol=1e-6`, `rtol=1e-5`.

| Batch | Standalone shape | Adapter score shape | Max abs diff | Mean abs diff | Max relative diff | Kết quả |
|---|---|---|---:|---:|---:|---|
| Synthetic B=1 | `[1]` | `[1,1]` | 0.0 | 0.0 | 0.0 | PASS |
| Synthetic B=2 | `[2]` | `[2,1]` | 0.0 | 0.0 | 0.0 | PASS |
| Validation B=2 | `[2]` | `[2,1]` | 0.0 | 0.0 | 0.0 | PASS |

`adapter_score.squeeze(-1)` khớp chính xác standalone score trong float32 trên cùng model instance. Cả ba case đạt `torch.testing.assert_close`.

Score synthetic quan sát được: -0.0430295803 (B=1), và [-0.0430295952, 0.2228537798] (B=2). Validation batch score range: [-0.0993716940, 0.6266126037]. Đây là số đo của các mẫu thử, không phải phân phối toàn dataset.

## 8. Feature parity và thứ tự

Adapter trả `torch.cat([audio_embedding, text_embedding], dim=-1)`.

| Batch | Feature shape | Audio / Text shape | Audio feature max diff | Text feature max diff |
|---|---|---|---:|---:|
| Synthetic B=1 | `[1,1024]` | `[1,512]` mỗi tensor | 0.0 | 0.0 |
| Synthetic B=2 | `[2,1024]` | `[2,512]` mỗi tensor | 0.0 | 0.0 |
| Validation B=2 | `[2,1024]` | `[2,512]` mỗi tensor | 0.0 | 0.0 |

Ở mọi case, nửa đầu đúng bằng audio embedding và nửa sau đúng bằng text embedding. Kiểm tra thứ tự và feature parity đều PASS.

## 9. Số lần gọi encoder

Forward hook trên encoder thật ghi nhận cho mỗi case:

| Đường gọi | Số lần |
|---|---:|
| Standalone Expert1 forward | 1 |
| Một adapter forward | 1 |

Các lượt standalone, adapter và lần lặp deterministic là những forward riêng; adapter không gọi encoder lần thứ hai trong cùng forward.

## 10. Freeze và eval

| Kiểm tra | Kết quả |
|---|---|
| Tổng Expert1 parameters | 158,334,169 |
| Expert parameters trainable | 0 |
| Adapter-only trainable parameters | 0 |
| Sau `adapter.train()`, mọi expert module vẫn eval | PASS |
| Sau `adapter.train()`, parameters vẫn frozen | PASS |
| Lặp lại cùng input B=1 | Score/features max abs diff 0.0 |

Không tạo optimizer, không backward, không training step.

## 11. Thuộc tính số học

Tất cả output score, features và embeddings đều hữu hạn, float32 trên `cuda:0`.

Norm thống kê trên cả ba case:

| Đại lượng | Min | Max | Mean |
|---|---:|---:|---:|
| Audio embedding L2 norm | 1.0 | 1.0 | 1.0 |
| Text embedding L2 norm | 0.9999999404 | 1.0 | 0.9999999762 |

Score min/max quan sát được trong tất cả batch thử: -0.0993716940 / 0.6266126037. Đây không phải kết quả calibration hay đánh giá toàn validation set.

## 12. Kiểm tra dữ liệu validation thật

Lần chạy thành công đã xử lý hai sample validation:

| Sample ID | Batch waveform | Sample rate | Score/feature parity |
|---|---|---:|---|
| `07500.wav#row0` | `[2,1,480000]` trong batch | 48 kHz | PASS |
| `07501.wav#row1` | `[2,1,480000]` trong batch | 48 kHz | PASS |

Dataset gốc và collate thực hiện TorchAudio decode, mono conversion bằng trung bình các kênh, resample lên 48 kHz, rồi center crop hoặc repeat/pad theo logic validation. Batch validation có shape `[2,1,480000]`; captions và thứ tự được giữ nguyên. Standalone và adapter score cùng audio/text features đều có max absolute difference 0.0; outputs hữu hạn.

**`REAL_DATA_PREPROCESSING_PARITY = PASS (2 samples)`.** Phạm vi xác minh chỉ gồm hai mẫu trên.

## 13. Lệnh kiểm tra và kết quả

Theo log đầu ra do người dùng cung cấp, lệnh chạy offline là:

```bash
python3 scripts/verify_expert1_runtime.py \
  --pretrained-checkpoint weights/laion-clap.pt \
  --finetuned-checkpoint weights/finetuned_weights/expert1_best.pt \
  --device auto \
  --offline \
  --validation-csv /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/meta_data/validation_average.csv \
  --audio-root /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/wav/validation \
  --num-real-samples 2
```

Verifier kết thúc với exit code 0, `status=PASS_REAL_CHECKPOINT_AND_NUMERICAL_PARITY`, `real_validation.status=PASS`, không có errors.

| Kiểm tra | Kết quả |
|---|---|
| Runtime verifier, pretrained + fine-tuned checkpoints, synthetic B=1/B=2 và real B=2 | PASS |
| Expert 1 unit tests | PASS — 22 tests |
| Toàn bộ lightweight test suite đã chạy ở lần xác minh trước | PASS — 63 tests |
| `py_compile scripts/verify_expert1_runtime.py` | PASS |

Regression suite không cần chạy lại cho cập nhật báo cáo này vì không có mã nguồn thay đổi.

## 14. Lỗi và nguyên nhân

- Lần thử trước khi TorchCodec được cài đã dừng ở bước decode validation. Lần chạy mới trong log cung cấp hoàn tất thành công; lỗi trước đây đã được khắc phục.
- Cache preflight hiện báo config, model weights và tokenizer đều sẵn có tại cache mặc định; model dựng được offline.
- Không có lỗi restore hoặc parity trong lần chạy mới.

## 15. Rủi ro còn lại

- Preprocessing và parity trên real audio được xác minh cho hai sample, chưa bao phủ toàn bộ dataset hoặc mọi sample rate/codec.
- Raw fine-tuned checkpoint không có SHA/fingerprint của pretrained base; strict-load không chứng minh provenance của base checkpoint.
- Observed score range chỉ thuộc ba batch nhỏ đã thử, không đại diện cho toàn tập.
- Transformers phát cảnh báo về LM-head/pooler khi khởi tạo RoBERTa từ checkpoint khác task; quá trình nạp CLAP và fine-tuned Expert1 sau đó thành công.

## 16. Files

Báo cáo `docs/LOCAL-04/runtime-verification.md` được cập nhật với log mới về validation subset thành công. Không sửa script, adapter, Expert1 source, dependency hay checkpoint. Không commit/push.

## 17. Trạng thái xác minh cuối

| Hạng mục | Trạng thái |
|---|---|
| LAION pretrained restore | PASS |
| Fine-tuned Expert1 strict restore | PASS; 0 missing / 0 unexpected |
| `D_CLAP` / feature width | 512 / 1024 |
| Score parity B=1, B=2 và validation B=2 | PASS; max absolute diff 0.0 |
| Audio/text feature parity và ordering | PASS; max absolute diff 0.0 |
| Một encoder call cho mỗi adapter forward | PASS |
| Frozen/eval và parent `.train()` behavior | PASS |
| Finite outputs và repeated evaluation | PASS |
| Preprocessing validation gốc | PASS trên 2 sample |

**Overall: `PASS — REAL CHECKPOINT & NUMERICAL PARITY VERIFIED`.** Runtime parity được xác nhận với checkpoint thật trên synthetic và hai sample validation đã được preprocessing bằng pipeline gốc; chưa đánh giá toàn bộ validation set.

## 18. Sẵn sàng cho LOCAL-05

Expert 1 đã runtime-verified về strict checkpoint restore, score/feature parity và preprocessing trên hai mẫu validation thật. Có thể tiếp tục phát triển Expert 2 Adapter. Không tự động bắt đầu LOCAL-05.

## Tham chiếu source

- `expert1_finetune/Train_Expert1.py`: `Expert1Dataset`, `collate_batch`, `CLAPAudioTextEncoder`, `Expert1.forward`, lưu raw best state dict.
- `model/expert_adapters/expert1_adapter.py`: `Expert1Adapter.from_finetuned`, strict state restoration, freeze/eval handling và single encoder call.
- `model/expert_adapters/base.py`: output contract validator.
- `scripts/verify_expert1_runtime.py`: offline cache check, restoration, hook-based call counting, parity tests, validation subset loader và summary JSON.
- Installed LAION-CLAP 1.1.7: `clap_module/model.py` tạo RoBERTa, audio/text embeddings; `clap_module/factory.py` nạp pretrained checkpoint.
