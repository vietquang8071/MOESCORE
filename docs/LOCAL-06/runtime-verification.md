# LOCAL-06.1 — Báo cáo xác minh runtime Expert 3 (M2D-CLAP)

## 1. Tóm tắt

**Trạng thái: `PASS — INCLUDING REAL VALIDATION INFERENCE`**.

Checkpoint M2D gốc có SHA-256 đúng fingerprint trong cấu hình. Factory nguyên bản dựng được `MoescoreExpert3` trong môi trường offline, nạp base checkpoint, xác thực và phục hồi partial-best checkpoint bằng `load_best_checkpoint()`. 77/77 tensor fine-tuned sau khi restore bằng chính xác tensor trong artifact. B=1, B=2 và hai WAV validation đều cho score, feature và năm khối `released` khớp standalone tuyệt đối trong FP32 (sai khác lớn nhất bằng 0).

Đã dùng duy nhất một model Expert 3 cho cả standalone và adapter; không train, không đổi precision, không sửa model hoặc checkpoint.

## 2. Môi trường

| Thuộc tính | Kết quả |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.17.0 |
| Thiết bị | `cuda:0`, NVIDIA GeForce GTX 1650 |
| Precision | FP32 (`torch.float32`) |
| Offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` |
| Seed inference | 2027 |
| Tolerance parity | `atol=1e-6`, `rtol=1e-5` |

## 3. Tài nguyên

| Tài nguyên | Đường dẫn tuyệt đối | Kích thước | Mtime |
|---|---|---:|---|
| M2D pretrained | `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/checkpoint-30.pth` | 1,694,677,655 bytes | 2024-11-22 05:17:00 +07 |
| Expert 3 partial-best | `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/finetuned_weights/expert3_best.pt` | 135,760,942 bytes | 2026-11-01 01:38:29.716582 +07 |
| Cấu hình gốc | `expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json` | 1,435 bytes | 2026-10-29 23:41:05.563909 +07 |
| Source root | `expert3_finetune/XACLE_2027_baselinemodel/` | — | — |

Các file checkpoint được xác nhận đọc được trước chạy. Script kiểm tra lại kích thước và mtime sau chạy: **không thay đổi**.

## 4. SHA-256 base checkpoint

- SHA cấu hình yêu cầu: `238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1`
- SHA tính streaming trên file local: `238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1`
- Kết quả: **PASS — trùng hoàn toàn**.
- SHA và kích thước cũng trùng `base_checkpoint_identity` trong artifact fine-tuned.

JSON gốc đặt đường dẫn tương đối `../weights/checkpoint-30.pth`; đường dẫn này không được dùng để tìm checkpoint. Factory nhận đường dẫn tuyệt đối đã xác minh trong repository root.

## 5. Tài nguyên text encoder offline

Theo PortableM2D, tên model là `google-bert/bert-base-uncased` (được chọn từ model family `B` trong tên M2D checkpoint; xem `portable_m2d.py:L246-L257,L545-L566,L569-L601`). Preflight `local_files_only=True` thành công:

- Config `model_type=bert`, `hidden_size=768`.
- Tokenizer `BertTokenizer`, vocabulary 30,522 tokens.
- Factory dựng tiếp `BertModel` từ cùng cache và nạp model thành công, không truy cập mạng.

Transformers báo các tensor `cls.predictions.*` và `cls.seq_relationship.*` là `UNEXPECTED` khi khởi tạo `BertModel`. Đây là các head của tác vụ pretraining không thuộc encoder BERT mà source Expert 3 sử dụng; không có lỗi khởi tạo hoặc lỗi inference.

## 6. Dựng M2D gốc

`Expert3Adapter.from_finetuned()` gọi constructor gốc `MoescoreExpert3` (`moescore_expert3.py:L98-L130`), giữ `activation_checkpointing=true` và `concat_assignment="released"`. Model import gốc được xác minh từ đúng source tree.

Constructor gọi PortableM2D với checkpoint đã chuẩn bị layout tên model tương thích, khởi tạo text encoder, rồi dựng projectors, metadata và head theo source. Log pretrained ghi nhận 166 parameter entries dùng cho model, `dropped: []` và `All keys matched successfully`. **Model gốc dựng thành công; base pretrained weights đã thực sự được nạp.**

PortableM2D gốc dùng `torch.load(..., weights_only=False)` cho base artifact (`portable_m2d.py:L266-L298`). Trong lần chạy này chỉ dùng checkpoint dự án đã được chỉ định và fingerprint xác minh.

## 7. Kiểm tra partial checkpoint

Artifact thật có đúng schema best-checkpoint:

- `format_version=1`, `checkpoint_kind="best"`.
- Đủ chín khóa cấp cao: `format_version`, `checkpoint_kind`, `model_trainable`, `model_config`, `base_checkpoint_identity`, `epoch`, `best_srcc`, `metrics`, `config`; không có khóa thiếu/thừa.
- `epoch=2`, `best_srcc=0.5960687422427399`.
- Config metadata: 16 kHz, 10 giây, activation checkpointing bật, assignment `released`.
- `model_config`: D=768, cross width=3840, `has_text_projector=false`, hai audio block và hai BERT layer cuối được mở train.
- Base identity: SHA, byte size và model directory khớp; trường đường dẫn artifact ghi `/content/drive/MyDrive/XACLE/weights/checkpoint-30.pth` là đường dẫn máy huấn luyện, không dùng để định danh thay cho fingerprint.

Preflight an toàn tìm thấy 77 trained tensors, tổng 33,932,801 phần tử; không có giá trị không phải tensor hoặc tensor phi hữu hạn. Phân nhóm: regression head 6 khóa, audio projector 15, hai M2D block cuối 24, hai BERT layer cuối 32.

## 8. Phục hồi các tensor fine-tuned

Factory adapter xác thực chính xác tập key so với tập `requires_grad` của model gốc trước freeze, sau đó gọi `expert3_checkpoint.load_best_checkpoint()` nguyên bản (`expert3_checkpoint.py:L60-L93`). Kết quả:

| Kiểm tra | Số lỗi |
|---|---:|
| Key fine-tuned bắt buộc bị thiếu | 0 |
| Key unexpected | 0 |
| Sai shape | 0 |
| Giá trị non-tensor / non-finite | 0 / 0 |
| Tensor sau restore khác artifact | 0/77 |

Đã so sánh từng tensor đã nạp với giá trị trong partial artifact; cả **77/77 tensor khớp chính xác**. Đây là xác minh thực tế việc loader copy weights, không chỉ xác minh schema. `PARTIAL_FINETUNED_RESTORATION=PASS`.

## 9. Kích thước runtime

| Giá trị | Đo được |
|---|---:|
| `embedding_dim` / D_M2D | 768 |
| Audio embedding | `[B,768]` |
| Text embedding | `[B,768]` |
| `cross_feature_dim` | 3,840 |
| `adapter.feature_dim` | 3,840 |
| Score | `[B,1]` |
| Features | `[B,3840]` |

Các đẳng thức `cross_feature_dim = 5 * embedding_dim`, `adapter.feature_dim = cross_feature_dim` và độ rộng tensor thực tế đều đạt.

## 10. Xác minh năm khối `released`

Source `encode_modalities()` trả `(text_emb, audio_emb)`; `forward()` trong released mode cố ý gán cặp này thành `(audio_feats, text_feats)`, nên vector cuối bắt đầu bằng **text rồi audio** (`moescore_expert3.py:L210-L266`). Các khối là:

1. Text embedding.
2. Audio embedding.
3. `abs(text - audio)`.
4. `text * audio`.
5. Cosine giữa hai embedding được lặp 768 lần.

Verifier tạm instrument đúng một lần gọi `encode_modalities()` trong adapter forward để thu embeddings đã normalize từ cùng pass; sau đó so từng lát `[B,768]` với phép tính gốc. B=1, B=2 và validation B=2: cả năm khối có max absolute difference **0.0**, đều pass `atol=1e-6`, `rtol=1e-5`. Không chạy encoder lần nữa để lấy feature tham chiếu.

## 11. Parity synthetic: standalone và adapter

Hai đường chạy dùng cùng một model đã restore, cùng waveform FP32 `[B,1,160000]`, captions, device và eval mode. Model gốc nhận `{ "wavs", "captions" }`; adapter nhận `{ "expert3_wavs", "captions_raw" }`.

| Batch | Score max abs / mean abs | Score max relative | Feature shape | Feature max abs / mean abs | Allclose |
|---|---:|---:|---|---:|---|
| B=1 | 0 / 0 | 0 | `[1,3840]` | 0 / 0 | PASS |
| B=2 | 0 / 0 | 0 | `[2,3840]` | 0 / 0 | PASS |

Score quan sát: B=1 `0.2450184673`; B=2 `[0.2403699160, 0.3550146818]`. Cả score và features finite, float32, trên `cuda:0`. Các giá trị trên chỉ là batch thử, không đặt giới hạn cho regression score.

## 12. Forward một lần

Hook/counter được reset riêng cho từng lời gọi:

| Loại đường chạy | Expert3.forward | `encode_modalities` | Audio encoder | Text encoder | Regression head |
|---|---:|---:|---:|---:|---:|
| Standalone, mỗi case | 1 | — | — | — | — |
| Adapter, mỗi case | 1 | 1 | 1 | 1 | 1 |

Adapter không gọi model forward lần thứ hai và không encode audio/text thêm để lấy feature.

## 13. Freeze, eval và tính lặp lại

- Tổng số tham số Expert 3: **200,556,547**.
- Tham số expert trainable sau factory: **0**.
- Tham số trainable riêng của adapter: **0**.
- Ban đầu mọi module của expert ở eval.
- Sau `adapter.train()`, cờ adapter ngoài là training như PyTorch thông thường, nhưng tất cả module Expert 3 vẫn eval và tất cả tham số vẫn frozen.
- `adapter.train(False)` và `adapter.eval()` cũng được gọi; expert tiếp tục eval.
- Hai inference lặp lại cùng input có score và features sai khác 0.

## 14. Validation WAV thật

Đã chạy 2 mẫu đầu trong validation CSV, dùng đúng `MoescoreExpert3Dataset.__getitem__` và `collate_fn` gốc (`moescore_expert3_dataset.py:L78-L116`). Để không quét toàn bộ manifest/audio chỉ để chọn hai mẫu, verifier đọc hai CSV row đầu rồi tạo subset view dùng nguyên hai method preprocessing/collation của class. Cả WAV và caption đưa vào standalone/adapter là cùng tensor và cùng caption đã `.strip()`.

- WAV IDs: `07500.wav`, `07501.wav`.
- Waveform cuối: `[2,1,160000]`, float32, 16 kHz.
- Preprocessing gốc: SoundFile float32/always-2D, mono bằng trung bình channel, `scipy.signal.resample_poly` tới 16 kHz khi cần, lấy 160,000 mẫu đầu, zero-pad bên phải, caption `.strip()`.
- Score standalone/adapter: `[0.5673313737, 0.1500748545]` / cùng giá trị.
- Score max/mean absolute difference: `0 / 0`.
- Feature `[2,3840]`, max/mean absolute difference: `0 / 0`; cả năm khối assignment đều khớp.
- Kết quả real-data smoke test: **PASS**. Không tính SRCC hoặc đánh giá toàn bộ validation.

## 15. Import isolation

Các module runtime đều ở trong Expert 3 source root:

- `...models.moescore_expert3` → `expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py`.
- `...models.m2d.vendor.portable_m2d` → `expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py`.
- `...utils.expert3_checkpoint` → `expert3_finetune/XACLE_2027_baselinemodel/utils/expert3_checkpoint.py`.

Không có origin ngoài dự kiến. Đây không phải kiểm tra factory coexistence của cả bốn expert trong cùng tiến trình; integration đó vẫn chưa được xác minh.

## 16. GPU memory

| Số đo | Giá trị |
|---|---:|
| VRAM tổng | 3,895,132,160 bytes |
| VRAM trống trước model | 3,734,175,744 bytes |
| VRAM trống sau kiểm tra | 2,788,360,192 bytes |
| Peak PyTorch allocated | 839,717,376 bytes |
| Batch lớn nhất | B=2 |

Không có OOM; giữ nguyên FP32, không offload/quantize/cache model.

## 17. Cảnh báo và lỗi trong quá trình xác minh

Hai lần chạy verifier đầu dừng trước model do chính harness: PyTorch build này không nhận `torch.device` trong một số CUDA memory-stat API; và chạy script từ `scripts/` không tự đưa repo root vào `sys.path`. Script được sửa riêng để dùng fallback API không tham số và thêm repo root trong tiến trình verifier. Sau đó chạy thành công, exit code 0. Không sửa Expert 3, checkpoint hoặc thư viện.

Transformers load report có các BERT pretraining-head keys `UNEXPECTED` như mô tả ở mục 5; encoder `BertModel` vẫn khởi tạo và inference thành công. Log M2D báo toàn bộ keys khớp.

## 18. Regression tests

Baseline trước thay đổi:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Kết quả: **107/107 PASS**.

Sau runtime verification:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_expert3_adapter.py' -v
```

Kết quả: **20/20 PASS**.

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Kết quả cuối: **107/107 PASS**.

```bash
conda run -n nlp --no-capture-output python3 -m py_compile scripts/verify_expert3_runtime.py
```

Kết quả: **PASS**.

## 19. Lệnh tái chạy

```bash
conda run -n nlp --no-capture-output python3 scripts/verify_expert3_runtime.py \
  --base-checkpoint weights/checkpoint-30.pth \
  --finetuned-checkpoint weights/finetuned_weights/expert3_best.pt \
  --model-config expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json \
  --source-dir expert3_finetune/XACLE_2027_baselinemodel \
  --cache-dir /home/vitquay1708/.cache/huggingface/hub \
  --checkpoint-layout-root /tmp/moescore-expert3-m2d-layout \
  --device auto --offline \
  --validation-csv /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/meta_data/validation_average.csv \
  --audio-root /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/wav/validation \
  --num-real-samples 2
```

Lần chạy hoàn tất: status `PASS — INCLUDING REAL VALIDATION INFERENCE`, exit code 0, không có lỗi trong summary.

## 20. File thay đổi

- Tạo `scripts/verify_expert3_runtime.py`: verifier offline, kiểm SHA/schema, gọi factory gốc, so từng trained tensor, đo score/feature/khối assignment, kiểm single-pass/freeze/eval, chạy validation subset và in JSON.
- Tạo `docs/LOCAL-06/runtime-verification.md`: báo cáo này.

Không sửa adapter, expert source, configs, checkpoints hay các file ngoài phạm vi.

## 21. Rủi ro còn lại

- Fine-tuned artifact lưu base SHA, size, model family; đường dẫn lưu trong artifact là đường dẫn môi trường huấn luyện. SHA xác minh base được chọn, nhưng không tạo chữ ký số/provenance độc lập cho chính artifact partial.
- Đã kiểm đầy đủ partial trained parameter set của lần checkpoint này; chưa đánh giá các checkpoint khác.
- Chưa kiểm factory coexistence của cả bốn expert trong một tiến trình.
- Hai mẫu không đại diện cho toàn bộ validation; không tính SRCC hoặc chất lượng dự đoán toàn tập.
- Chưa kiểm end-to-end MOE gating/training.

## 22. Sẵn sàng cho LOCAL-07

Expert 3 đã **runtime verified**: base identity, partial restore, feature semantics, parity, real-data preprocessing smoke test và regression suite đều đạt. Có thể bắt đầu phát triển Expert 4 Adapter độc lập. Việc tích hợp chung bốn factory/gating vẫn cần một bước kiểm thử riêng.
