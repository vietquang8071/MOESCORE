# LOCAL-06 — Báo cáo triển khai Expert 3 Adapter (M2D-CLAP)

## 1. Tóm tắt

Đã triển khai `Expert3Adapter` quanh class `MoescoreExpert3` hiện có. Adapter giữ nguyên forward gốc, bao gồm regression score không giới hạn và toàn bộ cross-feature; chỉ ánh xạ key batch, kiểm tra input và dùng validator chung. Feature width được suy từ metadata `embedding_dim`/`cross_feature_dim`; assignment bị giới hạn ở `released` vì đó là cấu hình của checkpoint fine-tune.

Factory gọi constructor gốc để dựng M2D và xác thực SHA-256 pretrained, kiểm tra cấu trúc partial-best artifact, rồi gọi loader gốc `load_best_checkpoint`. Adapter freeze toàn bộ model và giữ model ở eval khi parent gọi `.train()`.

**Implementation:** `IMPLEMENTED_AND_UNIT_TESTED`  
**Runtime:** `REAL_CHECKPOINT_RESTORED` chưa xác minh; numerical parity chưa xác minh. Không có checkpoint thật nào được deserialize hoặc model M2D nào được dựng trong LOCAL-06.

**Trạng thái:** **PASS — IMPLEMENTED AND UNIT TESTED; RUNTIME VERIFICATION PENDING**

## 2. Repository baseline

- Branch: `main`; không đổi branch hoặc Git remote.
- Trước thay đổi, `git status --short` chỉ liệt kê các đường dẫn untracked đã có: `configs/moe_config.py`, `configs/moe_local.example.yaml`, báo cáo LOCAL-02–05, `docs/dataset-location.md`, package adapters, `scripts/`, các test LOCAL-03–05 và `weights/`. Không có tracked file nào bị báo modified.
- Baseline full suite: `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_*.py" -v` — **87/87 PASS**.
- Baseline có cảnh báo dependency `timm.models.layers` deprecated và Hugging Face Hub không xác thực; import tests chặn network và vẫn pass.
- LOCAL-02/03/04/05 implementation/runtime reports và các audit kiến trúc, checkpoint, input/output, adapter design đã được xem xét. Source hiện tại được ưu tiên khi báo cáo trước đó khác.

## 3. Kiến trúc Expert 3 gốc

Nguồn chính là [moescore_expert3.py](../../expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py):

- `MoescoreExpert3.__init__` nhận `checkpoint_path`, `cache_dir`, `activation_checkpointing`, `concat_assignment`, `checkpoint_layout_root`, `expected_checkpoint_sha256` (L98–105). Nó tính SHA-256 base checkpoint, so với hash mong đợi nếu được đặt, dựng layout tương thích PortableM2D, khởi tạo text encoder và cấu hình partial fine-tuning (L108–130).
- Chiều embedding được suy từ projector audio/text hoặc metadata text encoder; constructor yêu cầu hai nhánh có cùng D, đặt `embedding_dim=D`, `cross_feature_dim=5D` và tạo regression head `5D→512→128→1` (L132–172).
- Partial fine-tuning gốc freeze backbone rồi mở train các block audio cuối, các layer BERT cuối, projector tương ứng và alignment head (L181–208). Factory của adapter dùng tập tham số `requires_grad` sau chính bước dựng này để xác định tập key bắt buộc từ artifact.
- `encode_modalities()` nhận waveform `[B,T]` hoặc `[B,1,T]` và list caption thô; nó chuẩn hóa embedding hai modality rồi trả `(text_emb, audio_emb)` (L210–242).
- `forward()` trả `(predicted_score, cross_features)`. Score là output trực tiếp của regression head và không bị sigmoid/tanh/clamp (L244–266).

### Released assignment

Đây là chi tiết tương thích checkpoint cần giữ nguyên. Encoder trả `(text_emb, audio_emb)`. Với `concat_assignment="released"`, source gán cặp đó vào `(audio_feats, text_feats)`, tức là:

1. block 1 = embedding text đã chuẩn hóa;
2. block 2 = embedding audio đã chuẩn hóa;
3. block 3 = `abs(block1 - block2)`;
4. block 4 = tích từng phần tử;
5. block 5 = cosine scalar được lặp D lần.

Các block được nối theo đúng thứ tự trên và chính vector `[B,5D]` đó vừa đi vào alignment head vừa được trả về (`moescore_expert3.py:L242–266`). Đây là hành vi source “released”, không phải sắp xếp lại thành thứ tự semantic. `Expert3Adapter` từ chối `semantic` để tránh tải/ghép một checkpoint đã học theo thứ tự khác.

## 4. Hợp đồng input gốc

Dataset gốc ở [moescore_expert3_dataset.py](../../expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py):

- `soundfile.read(..., dtype="float32", always_2d=True)`; trộn kênh bằng trung bình float32 (L78–85).
- Nếu khác 16 kHz, dùng `scipy.signal.resample_poly` với tỉ lệ rút gọn theo gcd; cắt từ đầu tối đa 160.000 mẫu rồi pad zero bên phải nếu ngắn (L85–90).
- Trả waveform `[1,160000]` float32. Caption được `.strip()` trong dataset; collate trả `wavs=[B,1,160000]`, `captions=list[str]` (L91–116).

Adapter nhận `expert3_wavs` và `captions_raw`, rồi ánh xạ thành `{"wavs": audio, "captions": stripped_captions}`. Nó kiểm tra batch không rỗng, rank/channel/length, dtype float32, giá trị hữu hạn, cùng device với model, số lượng caption và kiểu string. Caption chỉ được strip để khớp dataset gốc; caption rỗng sau strip bị từ chối giống dataset. Adapter không decode, resample, crop, pad, tokenize hay move tensor.

## 5. Hợp đồng output gốc

- Score: `[B,1]`, output nguyên trạng của regression head; không có score transformation.
- Features: `[B,5D_M2D]`, cross-feature FP32 nguyên trạng trước head.
- Adapter gọi `self.model(original_batch)` đúng một lần và trả tuple không sửa `(score, features)`. `ExpertAdapter.forward()` hiện có kiểm tra score `[B,1]`, feature `[B,feature_dim]`, batch/device/dtype và finite values.
- D không hard-code. Constructor adapter đọc `model.embedding_dim`, `model.cross_feature_dim` (hoặc metadata `model_config` đáng tin cậy), yêu cầu `cross_feature_dim == 5 * embedding_dim`, assignment `released`, rồi đặt `feature_dim` bằng cross width. Width tường minh nếu có phải khớp.

Chưa có D số học thực tế được xác nhận trong LOCAL-06 vì không chạy model thật. Con số 768 trong source `utils/config.py` được ghi là assertion cho model family này, nhưng adapter vẫn lấy metadata từ model đã dựng làm chuẩn và không nhúng 768/3840 thành production constant.

## 6. Cấu trúc partial checkpoint và kiểm tra

Nguồn [expert3_checkpoint.py](../../expert3_finetune/XACLE_2027_baselinemodel/utils/expert3_checkpoint.py) định nghĩa `FORMAT_VERSION=1` (L13). `save_best_checkpoint()` ghi payload gồm `format_version`, `checkpoint_kind="best"`, `model_trainable`, `model_config`, `base_checkpoint_identity`, `epoch`, `best_srcc`, `metrics`, `config` (L16–58). `train_expert3.py` lưu best tại `best_model.pt` qua hàm này (L483–487). Resume artifact là loại khác và có thêm optimizer/scheduler/scaler/RNG (L130–156).

Adapter không xem partial artifact như full `state_dict`. Trước khi gọi loader nguồn, nó kiểm tra:

- đúng top-level key set, version 1 và `checkpoint_kind="best"`;
- `model_config` khớp chính xác với model gốc được dựng, gồm width và `released`;
- base identity gồm path, size, SHA-256, model directory. Path được yêu cầu là chuỗi nhưng không so sánh tuyệt đối vì đường dẫn nơi fine-tuning chạy có thể khác. SHA, byte size và model family phải khớp model hiện tại;
- run config giữ 16 kHz, 10 giây, assignment released, activation-checkpointing phù hợp và SHA base đã chọn;
- tập `model_trainable` phải chính xác bằng tập parameter `requires_grad` của model partial trước freeze; thiếu key, key thừa, giá trị không tensor, shape khác hoặc giá trị không hữu hạn đều bị từ chối.

Sau đó gọi đúng `load_best_checkpoint(model, path)` gốc. Source loader kiểm version, base SHA, `model_config`, chính xác tập tên trained key rồi copy các tensor (L60–93). Source không preflight shape trước `copy_`; adapter kiểm shape trước để chặn broadcast/sai shape mà không sửa source. Vì không sửa loader gốc, factory đọc artifact an toàn lần đầu để preflight rồi source loader đọc lại khi restore; điều này tốn thêm một lần đọc file nhưng không tạo model thứ hai.

M2D pretrained loader trong source dùng `torch.load(..., weights_only=False)` (PortableM2D L266–273). Chỉ dùng checkpoint dự án đáng tin cậy. Adapter không tự load base: constructor gốc xử lý PortableM2D, text encoder cache và SHA check. Fine-tuned artifact được preflight với `weights_only=True`; fallback chỉ theo hành vi tương thích của loader nguồn trên PyTorch cũ. Không có pretrained-only fallback.

## 7. Base identity và cấu hình

Source run config hiện có là JSON [expert3_moescore.json](../../expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json), không phải YAML. Nó ghi:

- `m2d_checkpoint_sha256 = 238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1` (L15–16),
- `max_len=10`, `sample_rate=16000`, `activation_checkpointing=true`, `concat_assignment="released"` (L34–40).

LOCAL-03 example YAML vẫn để `expected_base_sha256: null`; factory có thể nhận null ở field override nhưng dùng fingerprint SHA-256 tường minh của source run JSON. Nếu caller đưa hash tường minh khác, factory fail trước model construction. Constructor gốc sau đó hash file base thật và đối chiếu fingerprint. LOCAL-06 không tính hash trên file base 1.69 GB; bước hash thật chỉ xảy ra khi factory được gọi trong LOCAL-06.1.

Factory chấp nhận đường dẫn đã resolve tuyệt đối, không tự diễn giải lại path tương đối trong JSON. Đây là khác biệt quan trọng: `m2d_checkpoint` trong JSON là `../weights/checkpoint-30.pth`, và `train_expert3.py:45–47` resolve tương đối với source root, trong khi artifact hiện có ở repo root `weights/checkpoint-30.pth`. LOCAL-03 resolved `experts.expert3.base_checkpoint` là đường dẫn rõ ràng được truyền cho constructor; không có auto-search/substitution.

`from_config()` nhận full LOCAL-03 resolved mapping hoặc section Expert 3, chỉ kiểm tài nguyên E3 và không gọi global `validate_moe_resources()` vốn đòi tất cả expert resource. Full mapping cần `runtime.cache_dir`, `offline`, `device`. Factory chỉ hoạt động offline.

## 8. Factory và import isolation

Public factory:

```python
Expert3Adapter.from_finetuned(
    source_dir=..., base_checkpoint=..., finetuned_checkpoint=...,
    model_config=..., cache_dir=..., expected_base_sha256=None,
    concat_assignment="released", device="cpu", offline=True,
)
```

`from_config(resolved_config)` là shortcut theo schema LOCAL-03. Factory import trễ `expert3_finetune.XACLE_2027_baselinemodel.models.moescore_expert3` và `...utils.expert3_checkpoint`, kiểm `__file__`/origin nằm trong source root đã cấu hình, đặt biến Hugging Face offline tạm thời, gọi constructor/loader gốc và khôi phục biến môi trường sau construction. Không sửa `sys.path`, không xóa `sys.modules`, không monkey-patch import.

M2D sử dụng relative package import cho `models.moescore_expert3` và `models.m2d.vendor.portable_m2d`; không phát hiện `models`/`tools`/`BEATs` top-level dynamic import kiểu MGA. Helper origin test chấp nhận đúng source tree và từ chối origin ngoài nó. Factory-level coexistence thực tế chưa được chạy do giới hạn LOCAL-06.

## 9. Freeze/eval

- Khi adapter nhận model đã dựng hoặc sau factory restore: toàn bộ Expert 3 parameter `requires_grad=False`, model `.eval()`.
- Override `train(mode)` gọi hành vi `nn.Module` cho chính adapter, sau đó refreeze và ép wrapped model về eval. Điều này cũng xử lý descendant custom `train()` của M2D nếu có.
- Adapter không gọi `torch.no_grad()` hoặc `inference_mode()` nội bộ; caller/MoE sau này quyết định autograd context.
- Unit test xác nhận `.train()` và `.train(False)` giữ model/dropout eval, không có parameter trainable, forward trong no-grad hợp lệ và lặp deterministic với stub.

## 10. Tài nguyên local

Chỉ kiểm tra file metadata/existence/readability; không deserialize hoặc hash:

| Resource | Đường dẫn | Byte | Hiện trạng |
|---|---|---:|---|
| Pretrained M2D | `weights/checkpoint-30.pth` | 1,694,677,655 | Tồn tại, readable |
| Fine-tuned partial E3 | `weights/finetuned_weights/expert3_best.pt` | 135,760,942 | Tồn tại, readable |
| Model/training config | `expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json` | 1,435 | Tồn tại, readable |
| Canonical source tree | `expert3_finetune/XACLE_2027_baselinemodel/` | — | Tồn tại |

Kích thước và modified time lấy bằng `stat` trong lần kiểm tra này. Chúng không xác nhận nội dung checkpoint, key set, hash khớp hoặc khả năng dựng model. Không file weight/config gốc nào được sửa.

## 11. Test results

Tất cả lệnh dùng Conda environment `nlp`:

| Lệnh | Kết quả |
|---|---:|
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_expert3_adapter.py" -v` | **PASS**, 20 test |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_adapter_base.py" -v` | **PASS**, 13 test |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_expert1_adapter.py" -v` | **PASS**, 22 test |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_expert2_adapter.py" -v` | **PASS**, 24 test |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_expert_import_isolation.py" -v` | **PASS**, 6 test |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_moe_config.py" -v` | **PASS**, 21 test |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p "test_*.py" -v` | **PASS**, **107/107**, 0 failures, 0 skips |
| `conda run -n nlp --no-capture-output python3 -m compileall -q model/expert_adapters tests/test_expert3_adapter.py` | **PASS** |

Tổng count theo từng file kiểm tra riêng không cộng được với full suite do import/base/config test cũng được tính trong suite tổng. Full suite xác nhận 107 test. Import isolation cố ý chặn network; log `NETWORK_BLOCKED` là guard quan sát, không phải lỗi test. Test import legacy in cảnh báo `timm` deprecation và HF Hub unauthenticated warning như baseline.

### Phạm vi test mới

20 test method bao phủ B=1/2/4, score ngoài `[-1,1]` nguyên trạng, đủ năm block released bằng dữ liệu phân biệt, một lần gọi model, key/caption mapping, input/output errors, metadata dimension động, freeze/eval, import nhẹ, factory resource failures, SHA/assignment, top-level schema, exact trainable key set, shape validation và cache path lồng mới. Factory test ghi tiny synthetic payload nhưng gọi `load_best_checkpoint` gốc; nó không xác minh checkpoint thật.

## 12. Khác biệt và rủi ro còn lại

- Tên config thực tế là `expert3_moescore.json`, khác một số tên YAML chung trong kế hoạch; implementation đọc JSON đúng với `train_expert3.py`.
- Đường base checkpoint trong JSON tương đối không trỏ đến artifact repo-root hiện có khi resolve theo training source root. Factory nhận `base_checkpoint` tuyệt đối từ LOCAL-03; không thay config nguồn.
- Checkpoint có thật tại disk, nhưng LOCAL-06 không xác nhận nó là payload đúng, đúng base SHA hay compatible với architecture. Không nên coi việc `ls/stat` là restore.
- Factory tiền kiểm tra shape rồi gọi loader gốc, do đó fine-tuned partial file được deserialize hai lần trong restoration. Source không nhận preloaded payload API; source không bị sửa để tránh lặp đọc.
- Base M2D constructor dùng `weights_only=False` khi mở pretrained artifact. Chỉ dùng file đã tin cậy; cần cache text model local và offline để constructor chạy.
- D số học, output shapes thật, regression parity, feature parity bằng model thật và pretrained hash thật vẫn chưa xác nhận.
- Import test module-level không chứng minh real factory construction hoặc simultaneous coexistence với các factory E1/E2/E4. Các namespace factory compatibility này tiếp tục chưa xác minh.

## 13. Files changed

Đã tạo mới:

- `model/expert_adapters/expert3_adapter.py`
- `tests/test_expert3_adapter.py`
- `docs/LOCAL-06/implementation-report.md`

Không sửa shared `base.py`, `__init__.py`, `configs/moe_config.py`, YAML mẫu, expert source E1–E4, model/training/loss files, E1/E2 adapters, scripts hoặc checkpoint. Không cài dependency, tải resource, train model, đổi Git remote, commit hay push. Các untracked file có từ trước được giữ nguyên.

## 14. Sẵn sàng cho LOCAL-06.1

Các file cần thiết đã có ở disk: base M2D, partial fine-tuned Expert 3, source tree và JSON run config. LOCAL-06.1 có thể thực hiện restoration thật bằng factory này. Nó cần:

1. xác nhận local HF text encoder cache/offline construction;
2. hash base artifact và so với SHA trong `expert3_moescore.json`;
3. dựng đúng `MoescoreExpert3` với assignment released;
4. xác nhận partial artifact thực sự có schema/version/key set/shape/metadata phù hợp và loader gốc restore;
5. ghi lại D thực tế, score/features, released block parity với standalone, freeze/eval và real validation preprocessing.

Không thực hiện các bước này trong LOCAL-06.

## 15. Sẵn sàng cho LOCAL-07

Có thể bắt đầu phát triển Expert 4 độc lập; LOCAL-06 không đổi code Expert 1/2 hay hệ thống gating. Expert 3 runtime restoration/parity vẫn pending và phải được hoàn tất trước khi tuyên bố tích hợp bốn expert đã runtime-verified.
