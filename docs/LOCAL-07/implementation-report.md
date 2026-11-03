# LOCAL-07 — Báo cáo triển khai Expert 4 Adapter

## 1. Tóm tắt

Đã triển khai `Expert4Adapter` quanh đúng `expert4_finetune.model.Expert4`, dùng chung `ExpertAdapter`, giữ nguyên score và fused feature của model gốc, giữ `wav_lens` theo số mẫu hợp lệ và đóng băng toàn bộ model. Factory dùng loader fine-tuned gốc, kiểm tra bundle partial trước khi khởi tạo, kiểm tra tập tham số và giá trị đã phục hồi, rồi mới freeze.

Trạng thái: **PASS — IMPLEMENTED AND UNIT TESTED; RUNTIME VERIFICATION PENDING**. Bộ kiểm thử nhẹ chạy 129/129 PASS. Không dựng BEATs/CLAP model, không deserialize checkpoint thật và không chạy inference thật trong LOCAL-07.

## 2. Baseline repository

- Nhánh: `main`.
- Môi trường dùng để kiểm thử: Conda `nlp`, Python 3.10.21, PyTorch 2.14.1+cu130, CUDA khả dụng; GPU NVIDIA GeForce GTX 1650. Test adapter dùng CPU/stub, không dựng model nặng trên GPU.
- Trước thay đổi, working tree có nhiều thư mục/tệp untracked từ LOCAL-02 đến LOCAL-06, `weights/`, `scripts/`, configs và các test trước đó. Không xóa, sửa hoặc stage các thay đổi này.
- Baseline quan sát trực tiếp: **107 tests, PASS**.
- Sau triển khai: **129 tests, PASS**.
- Không commit, push, đổi branch hoặc remote.

## 3. Kiến trúc Expert 4 gốc

Nguồn gốc là `Expert4` trong `expert4_finetune/model.py:L8-L24`.

- Constructor nhận `audio_encoder` và `text_encoder`, rồi tạo `SeqCoAttn()` và `RegressionHead()`.
- `Expert4.from_pretrained(beats_checkpoint, clap_checkpoint, unilm_source_dir)` tạo `AudioEncoder` và `TextEncoder.load_pretrained()` theo đúng implementation gốc.
- `forward(batch)` đọc `wavs`, `wav_lens`, `captions`, gọi từng encoder một lần, đưa chuỗi và mask vào SeqCoAttn, rồi trả `(head(fused), fused)`.
- SeqCoAttn (`expert4_finetune/seq_coattn.py:L6-L22`) có hai attention hướng audio-query/text-key và text-query/audio-key. Hai nhánh cùng dùng sequence đầu vào gốc; từng nhánh được adaptive max-pool; vector cuối ghép nhánh audio-query trước nhánh text-query sau.
- Head (`seq_coattn.py:L25-L31`) là `Linear(1024,2048) → ReLU → Dropout(0.3) → Linear(2048,1) → Tanh`, sau đó `squeeze(-1)`.

## 4. Preprocessing và input gốc

`Expert4Dataset` và `collate_expert4` nằm trong `expert4_finetune/dataset.py:L23-L26,L73-L115`.

- Đọc WAV bằng `scipy.io.wavfile.read`.
- PCM integer được scale theo kiểu số nguyên nguồn; audio nhiều kênh được lấy trung bình theo channel; audio float chuyển sang `float32`; waveform không hữu hạn bị từ chối.
- Resampling dùng `scipy.signal.resample_poly` với tỷ lệ chính xác qua `Fraction`, sample rate mặc định 16 kHz.
- Cắt 10 giây đầu, tối đa 160000 mẫu. Dataset chưa pad; collate right-pad về 160000 và tạo tensor `[B,1,160000]`.
- `wav_lens` là `torch.long`/`int64`, chứa số mẫu hợp lệ sau resample và crop, trước padding theo batch. Giá trị này được giữ theo từng sample.
- Caption được `.strip()` trong dataset; caption rỗng bị loại. Tokenizer nằm trong `TextEncoder`, pad/truncate nội bộ ở max length 77; adapter không tự tokenize hay truncate.

## 5. Độ dài chuỗi và mask padding

`AudioEncoder.forward` (`expert4_finetune/audio_encoder.py:L50-L76`) yêu cầu waveform `[B,1,N]`, lengths `[B]`, length dương và không vượt chiều waveform. Nó squeeze channel, chuẩn hóa mean/std trên toàn waveform đã pad, tạo mask bằng `arange >= lengths`, rồi chuyển mask qua các bước giảm chiều của BEATs. `True` biểu thị padding.

`TextEncoder` cũng trả mask `True` cho token padding (`text_encoder.py:L111-L124`). SeqCoAttn truyền mask vào `MultiheadAttention.key_padding_mask` (`seq_coattn.py:L17-L21`), nên mask chặn key/value padding. Model gốc không loại query padding trước adaptive max-pool; đó là hành vi nguồn cần giữ nguyên.

Adapter kiểm tra `wav_lens` là `int64 [B]`, cùng device với waveform, dương, không vượt tensor pad hoặc 160000 mẫu. Sau đó adapter chuyển nguyên tensor lengths và thứ tự sample vào key `wav_lens`; không đếm zero, không thay bằng chiều pad và không tự tạo mask.

## 6. Output gốc

- Score gốc có shape `[B]`; Tanh được áp dụng bên trong RegressionHead. Adapter chỉ thêm singleton axis để trả `[B,1]`, không áp dụng Tanh/Sigmoid/clamp hay scale lần nữa.
- Feature là nguyên vector pooled hai chiều cross-attention, audio-query trước text-query, không normalize.
- Chiều được suy từ `2 * model.seq_coattn.dim`, đồng thời đối chiếu với `model.head.net[0].in_features`. Với source hiện tại, metadata source khai báo `dim=512`, head input 1024; chiều 1024 là source-derived, chưa runtime-confirmed với checkpoint thật.

## 7. Khôi phục các thành phần pretrained

Factory gọi `expert4_finetune.inference.load_finetuned()` (`inference.py:L50-L59`), loader này dùng `Expert4.from_pretrained()` gốc (`model.py:L16-L18`).

- **BEATs:** `AudioEncoder` nạp checkpoint `checkpoint['model']` với `strict=True`, đọc cấu hình BEATs từ chính checkpoint và yêu cầu 12 encoder layers (`audio_encoder.py:L21-L41`).
- **CLAP:** `TextEncoder.load_pretrained()` dùng LAION-CLAP `CLAP_Module(enable_fusion=False)` và checkpoint được chỉ định, kiểm tra RoBERTa/projection state rồi nạp text/projection keys theo source (`text_encoder.py:L62-L109`). Loader gốc dùng deserialization unrestricted cho checkpoint CLAP tin cậy.
- **UniLM:** là source tree chứa BEATs (`unilm/beats/`), không phải một pretrained checkpoint riêng. Source fingerprint được tính từ các file BEATs liên quan và git commit nếu có.
- `roberta-base` tokenizer/config/cache cục bộ vẫn là runtime prerequisite của CLAP text path; LOCAL-07 không kiểm tra cache bằng khởi tạo encoder.

## 8. Fine-tuned checkpoint

Training source tạo `best_expert4.pt` với đúng các field `model_state_dict`, `config`, `fingerprints`, `epoch`, `best_srcc` (`trainer.py:L320-L324`). Artifact lưu **các tham số trainable**, không chứa full model state. Không có `format_version` hoặc `checkpoint_kind` trong source hiện tại; vì vậy implementation không tự bịa metadata đó. Resume artifact có thêm optimizer/scheduler/RNG fields và bị từ chối bởi exact top-level schema.

Loader gốc xác minh SHA-256 của BEATs và CLAP cùng identity của UniLM (`inference.py:L50-L56`), dựng model pretrained, rồi gọi `load_trainable_state_dict`. Hàm gốc yêu cầu tập key đúng bằng các tham số có `requires_grad=True`; sau đó dùng `load_state_dict(strict=False)` có chủ đích vì phần pretrained frozen không nằm trong partial artifact (`trainer.py:L120-L129`).

Adapter preflight đọc artifact bằng `weights_only=True`, kiểm tra schema/fingerprint/finite tensors, rồi gọi loader gốc. Sau load, adapter đối chiếu tập key, shape và giá trị các tensor trainable với checkpoint; sai lệch sẽ raise trước khi trả adapter. Factory không có nhánh pretrained-only fallback. Loader gốc vẫn dùng `weights_only=False`, nên chỉ được dùng với artifact tin cậy của dự án.

Source loader xác minh fingerprint của hai pretrained files và UniLM; nó không đối chiếu fingerprint CSV/config để xác định toàn bộ danh tính run. Best payload cũng không có discriminator riêng giữa best/resume; adapter phân biệt theo schema hiện hành (resume có field thừa).

## 9. `Expert4Adapter`

Đã tạo `model/expert_adapters/expert4_adapter.py`.

- API: `Expert4Adapter(model, feature_dim=None)`, `Expert4Adapter.from_finetuned(...)`, `Expert4Adapter.from_config(...)`.
- `_forward()` kiểm tra và map `expert4_wavs → wavs`, `expert4_wav_lens → wav_lens`, `captions_raw → captions`; caption chỉ được `.strip()` như dataset gốc.
- Gọi `self.model(original_batch)` đúng một lần. Trả nguyên fused feature tensor; score `[B]` chỉ được `unsqueeze(-1)`.
- Reuse validator chung `ExpertAdapter.forward()` cho shape, dtype floating, device và finite output.
- Không thêm tầng học được. Constructor và mọi lần gọi `train(mode)` đều freeze model và giữ wrapped Expert 4 trong eval.
- Audio decoding, resampling, crop/pad, chuẩn hóa hoặc mask creation không được thêm vào adapter.

## 10. Factory và cấu hình

`from_finetuned` nhận bốn resource paths tuyệt đối: BEATs checkpoint, CLAP checkpoint, UniLM source directory, fine-tuned best artifact. Nó xác thực file/directory, offline mode, preflight bundle, gọi đúng loader gốc, xác minh nguồn module và kiểm tra tensor đã load. `from_config` nhận full LOCAL-03 resolved config hoặc riêng `experts.expert4`, chỉ kiểm tra resource Expert 4. LOCAL-03 schema đã có đủ bốn khóa (`configs/moe_config.py:L185-L198,L333-L378`), không cần sửa config chung.

`configs/moe_local.example.yaml` hiện còn đường dẫn placeholder cho fine-tuned Expert 4. Cấu hình runtime phải truyền absolute path thực đã resolve; adapter không tự tìm checkpoint.

## 11. Import isolation

Nguồn hiện tại có mismatch import đã tái hiện: `AudioEncoder` gọi `importlib.import_module('BEATs')` (`audio_encoder.py:L28-L32`), nhưng file BEATs dùng `from .backbone import ...` (`unilm/beats/BEATs.py:L16`). Import top-level trực tiếp trong process sạch thất bại với `ImportError: attempted relative import with no known parent package`.

Để giữ source gốc nguyên vẹn, factory adapter nạp cùng file trong private package có package context, kiểm tra origin, và chỉ tạm thời đặt đúng module đó dưới tên `BEATs` mà source gốc yêu cầu. Khi factory kết thúc, alias top-level và đường dẫn `sys.path` do constructor gốc thêm được gỡ đúng phần adapter/source vừa thêm. Không thay import hook, không thay/xóa module bên ngoài.

Đã xác minh helper bằng synthetic relative import unit test và import-only với BEATs source thật: file origin là `expert4_finetune/unilm/beats/BEATs.py`, `.backbone`/`.modules` được import trong private package, alias được dọn và `sys.path` phục hồi. Import-order subprocess của Experts 2/3/4 đều PASS. Các test này không dựng MGA/Expert4 factories đồng thời; factory-level coexistence vẫn **NOT_VERIFIED**. Nếu namespace `BEATs` đã thuộc source khác, factory fail-fast thay vì nhận nhầm module.

## 12. Kiểm kê resource local

Kiểm tra metadata filesystem; không deserialize hoặc sửa các checkpoint.

| Resource | Path local | Trạng thái / kích thước |
|---|---|---:|
| BEATs pretrained | `weights/BEATs_iter3_plus_AS2M.pt` | Có, readable, 361,499,833 bytes |
| LAION-CLAP pretrained | `weights/laion-clap.pt` | Có, readable, 1,863,587,645 bytes |
| Expert 4 fine-tuned best | `weights/finetuned_weights/expert4_best.pt` | Có, readable, 155,455,153 bytes |
| UniLM/BEATs source | `expert4_finetune/unilm/beats/` | Có; `BEATs.py`, `backbone.py`, `modules.py` tồn tại |
| Source training config | `expert4_finetune/expert4_colab.yaml` | Có, 732 bytes |
| Architecture config riêng | Không có trong source hiện hành | Architecture được định nghĩa trong model/encoder classes; BEATs config nằm trong checkpoint |

Sự tồn tại không chứng minh schema, fingerprint hoặc tensor compatibility. Những điều này để LOCAL-07.1 xác minh bằng loader thật.

## 13. Unit tests Expert 4

Lệnh:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_expert4_adapter.py" -v
```

Kết quả: **22/22 PASS**. Bao gồm batch 1/2/4, score/feature preservation, caption mapping, variable lengths, invalid input/output, metadata width, partial schema, tensor value/key validation, namespace origin, scoped BEATs import và frozen/eval.

## 14. Regression tests

Các suite chạy riêng:

| Suite | Kết quả |
|---|---:|
| `test_adapter_base.py` | 13/13 PASS |
| `test_expert1_adapter.py` | 22/22 PASS |
| `test_expert2_adapter.py` | 24/24 PASS |
| `test_expert3_adapter.py` | 20/20 PASS |
| `test_expert_import_isolation.py` | 6/6 PASS |
| `test_moe_config.py` | 21/21 PASS |

Full suite:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_*.py" -v
```

Kết quả baseline: **107 tests PASS**; kết quả cuối sau test Expert 4 mới: **129 tests, 129 PASS, 0 FAIL, 0 SKIP** (134.180 giây).

Syntax:

```bash
conda run -n nlp --no-capture-output python3 -m compileall -q model/expert_adapters tests/test_expert4_adapter.py
```

Kết quả: **PASS**. Suite phát ra các cảnh báo môi trường cũ về `timm.models.layers` và HF token chưa đăng nhập; không làm test thất bại. Không có model/checkpoint Expert 4 nào được khởi tạo trong các test.

## 15. Files thay đổi

- Tạo `model/expert_adapters/expert4_adapter.py`.
- Tạo `tests/test_expert4_adapter.py`.
- Tạo `docs/LOCAL-07/implementation-report.md`.
- Không sửa các Expert 1–4 gốc, adapter cũ, MoE/gating, loss, config schema, checkpoint hoặc remote.

## 16. Rủi ro và giới hạn

- Factory chưa chạy với checkpoint thật; SHA/fingerprint và partial key set của artifact đang có chưa được xác minh.
- Factory giữ nguyên source loaders BEATs/CLAP nhưng chưa gọi chúng với checkpoint thật; cache tokenizer/config `roberta-base` chưa được kiểm tra cho Expert 4.
- Chiều 1024 suy ra từ metadata source; runtime tensor width còn chờ LOCAL-07.1.
- Chưa so standalone score/feature với adapter trên model thật; chưa đo actual masks, score hoặc memory.
- Import-order tests chỉ import adapter/source modules; chưa thực thi coexistence giữa MGA factory và Expert 4 factory trong một process.
- Chưa chạy real validation WAV và chưa kiểm preprocessing parity.
- Chưa tích hợp full MoE hoặc train gating.

## 17. Sẵn sàng cho LOCAL-07.1

Ba checkpoint/resource chính hiện có tại local paths nêu trên, cùng với UniLM source tree và training config. LOCAL-07.1 có thể kiểm tra strict nguồn gốc/partial load, BEATs/CLAP initialization, real dimensions và parity. Trước khi công bố runtime PASS, cần xác minh artifact thật tương thích với exact five-key best schema và fingerprint từ training run; cần có cache `roberta-base` offline.

## 18. Sẵn sàng tích hợp nhiều expert

Adapter Expert 4 và lightweight import-order checks sẵn sàng cho bước tích hợp kế tiếp. Factory coexistence với MGA trong process chung vẫn chưa được chứng minh; conflict `BEATs` từ source khác sẽ bị từ chối rõ ràng. Không tuyên bố full four-expert factory coexistence hoặc MoE training readiness từ LOCAL-07.
