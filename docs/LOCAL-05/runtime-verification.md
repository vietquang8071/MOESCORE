# LOCAL-05.1 — Xác minh runtime Expert 2 (MGA-CLAP)

## 1. Trạng thái tổng thể

**`PASS — INCLUDING REAL VALIDATION INFERENCE`**

Factory MGA/ASE gốc đã dựng thành công trong chế độ offline; checkpoint MGA pretrained và checkpoint fine-tune Expert 2 đều được nạp theo luồng nguồn. Fine-tuned state dict được khôi phục `strict=True`. Inference FP32 trên cùng một model instance cho standalone Expert 2 và `Expert2Adapter` có score và feature giống tuyệt đối ở B=1, B=2 và hai mẫu validation thật. Kiểm tra nguồn module, kích thước, thứ tự feature, single-pass, freeze/eval và regression suite đều đạt.

Trạng thái tương thích cần giữ rõ: namespace `models` của MGA được xác minh ở runtime; `tools` không được import trong đường dựng/inference đã chạy nên origin của namespace này chưa được thực thi. Không dựng factory Expert 4 cùng process, do đó chưa tuyên bố khả năng coexistence factory của cả bốn expert.

## 2. Môi trường

| Mục | Quan sát |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.17.0 |
| Thiết bị | `cuda:0`, NVIDIA GeForce GTX 1650 |
| Precision | FP32 (`torch.float32`) |
| Offline | Bật; đặt `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` |
| Seed | 2027 |
| Tolerance | `atol=1e-6`, `rtol=1e-5` |

### BERT offline cache

Trước khi gọi MGA factory, verifier tải tokenizer, config và `BertModel` với `local_files_only=True` cho đúng identifier `bert-base-uncased`.

- Trạng thái: **PASS**.
- Vocabulary: 30,522; hidden size: 768; BERT parameters: 109,482,240.
- Không có download.
- Transformers in cảnh báo các trọng số NSP/MLM là `UNEXPECTED` khi dựng `BertModel` từ checkpoint pretraining có head; các head không thuộc `BertModel`. Đây là cảnh báo chuyển task/architecture dự kiến. Sau đó MGA factory và hai strict loads đều chạy thành công.

## 3. Tài nguyên

Các file đã được xác nhận là file thường, đọc được; snapshot kích thước và mtime trước/sau runtime không đổi.

| Tài nguyên | Đường dẫn | Kích thước | mtime UTC trước chạy |
|---|---|---:|---|
| MGA pretrained | `weights/mga-clap.pt` | 1,711,356,348 byte | `2026-10-20T06:18:43.279Z` |
| Fine-tuned Expert 2 | `weights/finetuned_weights/expert2_best.pt` | 594,417,402 byte | `2026-10-31T18:46:42.585Z` |
| Architecture YAML | `expert2_finetune/MGA-CLAP/settings/inference_example.yaml` | 504 byte | `2026-10-29T13:19:32.553Z` |
| Fine-tune config | `expert2_finetune/expert2_finetune.yaml` | 611 byte | `2026-10-29T14:32:48.545Z` |
| MGA source | `expert2_finetune/MGA-CLAP/` | Có, `models/ase_model.py` tồn tại | — |

Fine-tuned checkpoint là raw full `Expert2.state_dict()`. Thành công khi strict-load xác nhận tính tương thích cấu trúc/shape, nhưng artifact không gắn fingerprint base checkpoint; vì vậy nguồn gốc chính xác của pretrained run không được chứng minh chỉ bằng strict loading.

## 4. BERT offline validation

`BertTokenizer.from_pretrained`, `BertConfig.from_pretrained` và `BertModel.from_pretrained` đều chạy local-only trên `bert-base-uncased`. Đây là một bước kiểm tra riêng trước khi dựng MGA. Bên trong MGA, `TextEncoder` dùng tokenizer BERT gốc, `init_from_config=True`, và trọng số model được nạp từ checkpoint MGA như implementation quy định.

## 5. MGA factory construction

**PASS.** `Expert2Adapter.from_finetuned()` gọi `expert2_finetune.expert2_model.create_expert2`; factory tạo ASE bằng YAML active, tắt pretrained init nội bộ của audio encoder, tạo BERT từ config, rồi trả wrapper `Expert2` với `freeze_policy=partial`. Không tạo Expert2 thứ hai để đối chiếu.

Luồng được thực thi:

```text
inference_example.yaml
  → ASE(config)
  → weights/mga-clap.pt: checkpoint["model"] → load_mga_state(strict=True)
  → Expert2(ASE, freeze_policy="partial")
  → expert2_best.pt: raw full state_dict → load_state_dict(strict=True)
  → Expert2Adapter: freeze toàn bộ và eval
```

Fine-tuned strict restoration: **PASS**; missing keys **0**, unexpected keys **0**, shape mismatch **0**. Vì loader strict trả thành công, không có pretrained-only fallback. Base loader giữ nguyên nested `checkpoint["model"]` và strict behavior. Source `load_mga_state` chỉ bỏ legacy BERT `position_ids` khi runtime thiếu key đó và tensor là dãy vị trí chuẩn; factory đã đi qua loader này. Verifier không deserialize thêm checkpoint để xác định riêng liệu key compatibility đó có được kích hoạt trong artifact này hay không.

Pretrained MGA loader gốc dùng `torch.load(..., weights_only=False)` cho checkpoint tin cậy của dự án. Đây là hành vi nguồn hiện hữu, không được thay đổi. Fine-tuned raw tensor state dùng `weights_only=True` trong adapter.

## 6. Namespace isolation

Trong lần factory construction thật, mọi origin đã nạp thuộc namespace MGA dưới đây đều resolve về `expert2_finetune/MGA-CLAP/`:

| Module | Origin runtime | Kết quả |
|---|---|---|
| `models` | `.../expert2_finetune/MGA-CLAP/models/__init__.py` | PASS |
| `models.ase_model` | `.../expert2_finetune/MGA-CLAP/models/ase_model.py` | PASS |
| `models.audio_encoder` | `.../expert2_finetune/MGA-CLAP/models/audio_encoder.py` | PASS |
| `models.text_encoder` | `.../expert2_finetune/MGA-CLAP/models/text_encoder.py` | PASS |
| `models.htsat` | `.../expert2_finetune/MGA-CLAP/models/htsat.py` | PASS |
| `models.cnns` | `.../expert2_finetune/MGA-CLAP/models/cnns.py` | PASS |
| `models.feature_extractor` | `.../expert2_finetune/MGA-CLAP/models/feature_extractor.py` | PASS |
| `tools` | Không có trong `sys.modules` | `NOT_IMPORTED` |

`tools` không được gọi: Expert2 inference đi qua `MGAEncoder` và các phương thức ASE encode/MSC, không gọi `ASE.forward()` hay loss path dùng `tools.losses`. Vì thế origin `tools` chưa được kiểm chứng tại runtime. Namespace guard của adapter chạy trước/sau factory; không xóa `sys.modules` hoặc monkey-patch import. Factory Expert 4 không được dựng trong process này; import-only isolation tests không thay thế được kiểm chứng factory coexistence.

## 7. Kích thước thực tế

| Đại lượng | Thực tế |
|---|---:|
| Projection audio MSC | 1024 |
| Projection text MSC | 1024 |
| Shared codebook width | 1024 |
| Audio embedding trong forward | `[B,1024]` |
| Text embedding trong forward | `[B,1024]` |
| Adapter feature dimension | 2048 |
| Runtime feature output | `[B,2048]` |
| Score output | `[B,1]` |

Kích thước được đối chiếu cả với `frame_proj[-1]`, `word_proj[-1]`, `codebook` và output tensors thực. Không thêm projection hoặc ép width trong adapter.

## 8. Synthetic parity

Waveform tổng hợp deterministic: FP32 `[B,1,320000]`, 32 kHz boundary, seed 2027; captions khác nhau theo sample. Standalone và adapter nhận cùng audio/caption trên cùng restored model, CUDA device, evaluation mode và precision.

| Batch | Standalone score | Adapter score | Max abs / mean abs / max relative | Score | Feature max abs | Kết quả |
|---|---|---|---|---|---:|---|
| B=1 | `[1,1]` | `[1,1]` | `0 / 0 / 0` | `0.1494668871164322` | 0 | PASS |
| B=2 | `[2,1]` | `[2,1]` | `0 / 0 / 0` | `[0.14946691691875458, 0.4067768454551697]` | 0 | PASS |

Tolerance kiểm tra đã đặt trước: `atol=1e-6`, `rtol=1e-5`. Score parity và full feature parity đều đạt chính xác; audio/text embedding parity riêng cũng có max abs diff **0**.

## 9. Feature ordering

Hook không đổi trên `MGAEncoder.forward()` thu đúng output thật `(audio_embed, text_embed)` cho từng invocation. Script xác minh:

- Nửa đầu features bằng audio MSC embedding.
- Nửa sau bằng text MSC embedding.
- Score nguồn bằng `(audio * text).sum(dim=-1, keepdim=True)`.
- Các tensor lấy từ standalone và adapter bằng nhau chính xác.

Thứ tự `audio_MSC_then_text_MSC` được xác minh trên synthetic B=1/B=2 và real B=2.

## 10. Single-pass

Forward hook/counter được reset cho từng đường gọi. Cả standalone và từng adapter forward đều có số lần gọi:

```text
Expert2.forward:       1
MGAEncoder.forward:    1
ASE audio_encoder:     1
ASE text_encoder:      1
```

Không có lần encode ASE bổ sung để tách features. Các invocation riêng trong parity comparison được đo độc lập.

## 11. Freeze và evaluation

| Kiểm tra | Kết quả |
|---|---:|
| Tổng tham số Expert 2 | 148,092,376 |
| Trainable Expert 2 params sau restore | 0 |
| Trainable adapter-only params | 0 |
| Sau `adapter.train()`: mọi Expert2 module ở eval | PASS |
| Sau `adapter.train()`: mọi parameter vẫn frozen | PASS |
| Lặp lại B=1 cùng input: max score/feature diff | 0 / 0 |

Không tạo optimizer và không chạy backward/training.

## 12. Numerical properties

Mọi score/features hữu hạn, FP32, trên `cuda:0`. Norm L2 embedding MSC:

| Batch | Audio min/max/mean | Text min/max/mean |
|---|---|---|
| Synthetic B=1 | `1 / 1 / 1` | `1 / 1 / 1` |
| Synthetic B=2 | `1 / 1 / 1` | `0.9999999404 / 1 / 1` |
| Validation B=2 | `1 / 1 / 1` | `1 / 1 / 1` |

Đây là normalized MSC representations theo forward gốc; adapter không chuẩn hóa lần hai. Score synthetic quan sát được: B=1 `0.1494668871`; B=2 min/max `0.1494669169/0.4067768455`. Validation min/max `0.0975997597/0.4505676329`. Đây chỉ là batch smoke-test, không phải phân phối toàn tập.

## 13. Real validation inference

Hai WAV được nạp bằng `Expert2Dataset` và `collate_expert2` gốc, sau đó cùng tensor/caption được truyền vào standalone và adapter:

| File | Standalone score | Adapter score | Diff |
|---|---:|---:|---:|
| `07500.wav` | 0.4505676329135895 | 0.4505676329135895 | 0 |
| `07501.wav` | 0.09759975969791412 | 0.09759975969791412 | 0 |

Batch waveform `[2,1,320000]`, FP32, 32 kHz. Preprocessing là đường dataset gốc: SciPy decode, PCM scaling, mono mean, `resample_poly` đến 32 kHz, giữ 10 giây đầu, pad phải bằng zero; caption `.strip()`. Tất cả output finite, score/features parity và ordering đạt. Không tính SRCC hoặc đánh giá toàn validation set.

## 14. GPU memory

| Quan sát | Byte |
|---|---:|
| Free trước restore | 3,734,175,744 |
| Tổng VRAM | 3,895,132,160 |
| Peak PyTorch allocated | 691,505,152 (xấp xỉ 659 MiB) |
| Free sau kiểm chứng | 2,935,160,832 |

Không xảy ra OOM. Một model được tạo và dùng lại, không hạ precision/đổi kiến trúc/cache/offload.

## 15. Warnings và blockers

- Transformers `BertModel LOAD REPORT` nêu MLM/NSP keys unexpected do checkpoint BERT có pretraining heads; local-only load hoàn thành, MGA strict restoration và parity sau đó đều PASS.
- PyTorch phát `torch.meshgrid` warning yêu cầu truyền `indexing` trong phiên bản tương lai; warning không ảnh hưởng forward hiện tại.
- Không có checkpoint/cache/dependency/runtime blocker.
- Lần chạy verifier đầu dừng trước model construction do API `reset_peak_memory_stats(device)` không tương thích bản PyTorch này. Verifier đã sửa để chọn CUDA device hiện tại và reset peak không truyền tham số; lần chạy cuối hoàn tất. Không sửa adapter/model/vendor.
- Trong regression suite có warning HF Hub authentication từ test import smoke; test isolation ghi nhận `NETWORK_BLOCKED`, suite vẫn PASS. Verifier runtime đặt offline flags và không tải tài nguyên.

## 16. Regression tests

| Lệnh | Kết quả |
|---|---|
| `conda run -n nlp --no-capture-output python3 -m py_compile scripts/verify_expert2_runtime.py` | PASS |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_expert2_adapter.py" -v` | PASS, 24 tests |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_*.py" -v` | PASS, 87 tests, 68.274 s |

Không có test skip hoặc failure.

## 17. Files changed

- Tạo `scripts/verify_expert2_runtime.py`: verifier offline, strict restore qua factory gốc, hook/count assertions, synthetic parity, tùy chọn validation subset và JSON summary.
- Tạo `docs/LOCAL-05/runtime-verification.md`: báo cáo này.

Không sửa Expert2 source, adapter, vendor code, checkpoint, training/config/model khác. Branch hiện tại là `main`; working tree trước task đã có nhiều file untracked từ các milestone trước. Không commit/push.

## 18. Rủi ro còn lại

1. Checkpoint không lưu fingerprint base MGA; strict load kiểm tra key/shape, không chứng minh đầy đủ provenance của lần fine-tune.
2. Namespace `tools` không được import ở đường runtime được kiểm tra; origin namespace này vẫn `NOT_IMPORTED`.
3. Factory-level coexistence MGA với Expert 4 chưa được thực thi. LOCAL-03 import-order tests chỉ chứng minh module imports, không chứng minh hai factory dựng model trong cùng process.
4. Hai mẫu validation chỉ xác minh preprocessing/inference parity; không kết luận metric toàn dataset.
5. Chưa chạy đầy đủ gating/MoE hoặc training (ngoài scope).

## 19. Trạng thái cuối

| Yêu cầu | Trạng thái |
|---|---|
| BERT offline tokenizer/config/model | PASS |
| MGA/ASE factory gốc | PASS |
| Pretrained MGA checkpoint, loader gốc/nested `model`, strict | PASS |
| Fine-tuned raw full checkpoint, strict | PASS; missing/unexpected/shape mismatch = 0 |
| Runtime dimension và adapter width | PASS; 1024 + 1024 = 2048 |
| Synthetic B=1/B=2 score + feature parity | PASS; max absolute difference = 0 |
| Feature order và single-pass | PASS |
| Frozen/eval/repeat determinism | PASS |
| Hai WAV validation qua preprocessing gốc | PASS |
| `models` vendor origin | PASS; MGA source tree |
| `tools` vendor origin | NOT_IMPORTED |
| Coexistence của MGA factory và Expert4 factory | NOT_VERIFIED |
| Checkpoint files changed | Không; size/mtime không đổi |

## 20. Sẵn sàng cho LOCAL-06

Có thể bắt đầu Expert 3 Adapter độc lập. Expert 2 hiện đã runtime-verified cho restoration, output contract và parity trên inputs synthetic/validation đã kiểm tra. Trước khi tích hợp toàn bộ MoE, cần một task riêng kiểm chứng MGA/Expert4 factory coexistence và nguồn gốc namespace `tools` nếu factory Expert4 hoặc execution path thực tế dùng namespace đó.

## Tài liệu nguồn đã đối chiếu

- `expert2_finetune/expert2_model.py:12-23,26-71,74-99` — embedding MSC chuẩn hóa, score dot product, features `[audio,text]`, base loader và factory.
- `expert2_finetune/MGA-CLAP/models/ase_model.py:95-128` — projection widths và codebook.
- `expert2_finetune/MGA-CLAP/models/text_encoder.py:38-75` — BERT config/tokenizer và max length 30.
- `expert2_finetune/expert2_data.py:14-16,18-85` — preprocessing validation và collation gốc.
- `expert2_finetune/expert2_finetune.yaml:10` — freeze policy `partial`.
- `expert2_finetune/MGA-CLAP/settings/inference_example.yaml:1-26` — `embed_size=1024`, 32 kHz, BERT base.
- `model/expert_adapters/expert2_adapter.py:232-285,287-392,500-560` — feature metadata, strict fine-tuned restoration, factory, freeze/eval và forward mapping.
- `scripts/verify_expert2_runtime.py` — runtime verifier và các assertion parity/memory/resource immutability.
