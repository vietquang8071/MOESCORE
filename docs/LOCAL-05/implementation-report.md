# LOCAL-05 — Báo cáo triển khai Expert 2 Adapter (MGA-CLAP)

## 1. Tóm tắt

Đã triển khai `Expert2Adapter` bọc trực tiếp lớp `Expert2` fine-tuning hiện có, dùng chung `ExpertAdapter` của LOCAL-03. Adapter ánh xạ batch thống nhất sang khóa MGA gốc, trả nguyên score và feature từ một lần gọi model, suy ra chiều feature từ các projection/codebook của ASE, strict-load checkpoint fine-tuned, đóng băng toàn bộ model và giữ model ở evaluation mode.

**Trạng thái:** `IMPLEMENTED_AND_UNIT_TESTED` — 23 test mới của Expert 2 và toàn bộ 86 test đều PASS. Hai checkpoint Expert 2 hiện có trên đĩa nhưng LOCAL-05 chỉ kiểm tra đường dẫn/kích thước; không deserialize, khởi tạo ASE thật hoặc chạy inference. Vì vậy checkpoint restoration và numerical parity vẫn thuộc LOCAL-05.1.

## 2. Trạng thái repository ban đầu

- Nhánh hiện tại: `main`.
- Remote `origin` và `upstream` được đọc, không thay đổi.
- Trước LOCAL-05, working tree đã có nhiều đường dẫn untracked thuộc LOCAL-02/03/04, `weights/`, `scripts/` và test trước đó. Những nội dung này được giữ nguyên.
- Baseline trước thay đổi: `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_*.py' -q` — **63 test, PASS**.
- Không cài package, tải tài nguyên, chạy training, mở checkpoint thật, commit hoặc push.

## 3. Hợp đồng nguồn Expert 2

Nguồn authoritative là `expert2_finetune/expert2_model.py`:

- `MGAEncoder.forward()` gọi `ase.encode_text()` rồi MSC và L2-normalize text; tiếp đó gọi `ase.encode_audio()` rồi MSC và L2-normalize audio. Xem `MGAEncoder` dòng 12–23.
- `Expert2(ase, freeze_policy="partial")` hỗ trợ `partial`/`full`; `_set_trainable()` áp dụng các nhóm trainable của fine-tuning, và `Expert2.train()` có override bật lại những block đó khi `mode=True`. Xem dòng 26–66.
- `Expert2.forward(batch)` nhận `mga_wavs`, `caption_tokens`, tính `(audio * text).sum(dim=-1, keepdim=True)` và trả `(score, torch.cat((audio, text), dim=-1))`. Xem dòng 68–71. Adapter vì thế không tự tính lại score, không chuẩn hóa thêm và giữ nguyên thứ tự `[audio MSC, text MSC]`.
- `load_mga_state()` xử lý khóa BERT `position_ids` cũ nếu cần rồi strict-load state vào ASE. Xem dòng 74–84.
- `create_expert2()` deep-copy cấu hình, tắt pretrained/spec augmentation theo đường inference, tạo ASE, nạp `checkpoint["model"]` qua `load_mga_state()` rồi tạo `Expert2`. Xem dòng 86–99. Hàm này đưa MGA source vào `sys.path` và import `models.ase_model` bằng tên top-level.
- `Train_Expert2.py` gọi factory này trong đường huấn luyện/đánh giá (dòng 143–151 và 103–115). Best checkpoint lưu raw full `model.state_dict()` (dòng 202–207); checkpoint resume riêng là mapping có optimizer/scheduler và các metadata (dòng 210–218).
- Dataset gốc `expert2_data.py` xử lý audio 32 kHz, tối đa 320000 mẫu, mono/resample/crop/zero-pad và gọi `.strip()` trên caption; collate tạo `mga_wavs` và danh sách `caption_tokens`. Xem dòng 14–15, 49–84.

## 4. Adapter đã triển khai

Tệp mới: `model/expert_adapters/expert2_adapter.py`.

API chính:

```python
adapter = Expert2Adapter(model=restored_expert2, feature_dim=None)
score, features = adapter(batch)

adapter = Expert2Adapter.from_finetuned(
    source_dir=..., architecture_yaml=..., training_config=...,
    pretrained_checkpoint=..., finetuned_checkpoint=...,
    freeze_policy="partial", device="cpu", offline=True,
)

adapter = Expert2Adapter.from_config(resolved_moe_config)
```

- Constructor nhận model gốc đã tạo, đăng ký nó làm child module và không tạo ASE thứ hai. `feature_dim` tùy chọn; nếu ASE có metadata, dimension được lấy từ metadata và giá trị truyền vào phải trùng.
- `_forward()` ánh xạ batch rồi gọi `self.model(original_batch)` đúng một lần, trả nguyên tuple. `ExpertAdapter.forward()` trong LOCAL-03 thực hiện output validation chung; adapter không override/bỏ qua validator.
- `from_config()` nhận config đã qua schema/path resolution của LOCAL-03 hoặc section `experts.expert2`. Nó chỉ kiểm tra năm resource của Expert 2; không yêu cầu checkpoint của E1/E3/E4.
- Import adapter không import MGA vendor modules. MGA factory chỉ được import khi gọi restoration factory.
- Tệp `model/expert_adapters/__init__.py` không sửa; người dùng import trực tiếp `model.expert_adapters.expert2_adapter.Expert2Adapter`.

## 5. Input, score và feature dimension

Batch contract:

```text
expert2_wavs : float32 Tensor [B, 1, 320000], B >= 1, finite
captions_raw : sequence[str] có đúng B phần tử
```

Adapter map:

```text
expert2_wavs  -> mga_wavs
captions_raw  -> caption_tokens
```

Caption được `.strip()` đúng như standalone dataset; adapter từ chối caption rỗng sau bước này. Không lower-case, tokenize, crop, resample hoặc biến đổi audio. Bên tạo batch chịu trách nhiệm preprocessing tương đương dataset gốc.

Output không đổi từ model nguồn:

```text
score    [B,1]
features [B,D2] = [audio_MSC, text_MSC]
```

`D2` được xác định bằng chiều output của `ASE.frame_proj`, `ASE.word_proj` và chiều cột của `ASE.codebook`; ba giá trị phải thống nhất, sau đó tính tổng hai modality. Cơ sở nguồn/config hiện hành là `embed_size: 1024` trong `MGA-CLAP/settings/inference_example.yaml` dòng 1; `ase_model.py` dùng giá trị này làm chiều cuối của cả hai projection và codebook (dòng 103–128). Do đó **chiều dự kiến theo source/config là 2048**. Đây chưa phải dimension đo được từ model runtime thật. Không có số 2048 hardcode trong adapter; output validator sẽ bắt mismatch ở forward đầu tiên.

## 6. Khôi phục checkpoint

`from_finetuned()` yêu cầu các đường dẫn tuyệt đối đã resolve; kiểm tra source directory, YAML, pretrained và fine-tuned files trước khi tạo model. YAML đọc bằng `yaml.safe_load`; cấu hình kiến trúc phải có `embed_size`, `audio_encoder_args`, `text_encoder_args`, `training`. Freeze policy phải là `partial` hoặc `full` và khớp `training_config` gốc.

Chuỗi restoration được thiết kế:

1. Gọi đúng factory `create_expert2()` của fine-tuning. Factory này khôi phục MGA pretrained state qua `load_mga_state()`, bao gồm `checkpoint["model"]`, kiểm tra nghiêm ngặt và xử lý `position_ids` tương thích.
2. Nạp best fine-tuned raw tensor state với `torch.load(map_location="cpu", weights_only=True)`.
3. Từ chối state rỗng, structured resume bundle hoặc giá trị không phải tensor.
4. Gọi `model.load_state_dict(state, strict=True)`; thiếu key, key thừa hoặc shape sai phát sinh lỗi tường minh.
5. Đóng băng toàn bộ model, gọi `eval()` và bọc adapter. Không có fallback sang pretrained-only.

Factory MGA hiện dùng `torch.load(..., weights_only=False)` cho pretrained checkpoint trong code nguồn. Adapter giữ nguyên loader để bảo toàn tương thích; chỉ dùng checkpoint đáng tin cậy. Fine-tuned raw state dùng restricted `weights_only=True`. Offline mode bắt buộc; factory tạm đặt các biến/cờ Transformers/HF offline và khôi phục trạng thái sau khi kết thúc. Chỉ unit test factory giả lập được chạy trong LOCAL-05; restoration thật chưa chạy.

## 7. MGA namespace isolation

Hàm `validate_mga_import_namespace()` xem origin của module top-level `models` và `tools` đã nạp, gồm `__file__`, `__spec__.origin` và `__path__`; nếu origin nằm ngoài MGA source, nó ném `MGAImportNamespaceError` với đường dẫn xung đột. Hàm chỉ quan sát, không xóa `sys.modules`, thay module hoặc monkey-patch import. Factory gọi kiểm tra trước và sau original factory.

Test kiểm soát xác nhận origin trong MGA tree được chấp nhận, origin `models`/`tools` ngoài cây bị từ chối, và module registry không bị sửa. Import-order smoke tests hiện có cũng PASS.

**`MGA_FACTORY_NAMESPACE_RUNTIME = NOT_VERIFIED`**: chưa gọi factory thật. Source `create_expert2()` chèn MGA source vào `sys.path` nếu chưa có và import `models.ase_model`; các import-only tests không khởi tạo ASE nên không chứng minh việc đồng tồn tại lúc model construction. Factory gốc có thể để lại path entry theo chủ ý; adapter không thay đổi hành vi vendor này.

## 8. Cấu hình và tài nguyên hiện có

Các tệp sau tồn tại và readable theo metadata filesystem; chỉ kiểm tra tên/kích thước, không mở nội dung checkpoint:

| Resource | Repo-relative path | Kích thước |
|---|---|---:|
| MGA source | `expert2_finetune/MGA-CLAP/` | Thư mục có `models/ase_model.py`, `models/`, `tools/` |
| Architecture YAML | `expert2_finetune/MGA-CLAP/settings/inference_example.yaml` | 504 bytes |
| Training config | `expert2_finetune/expert2_finetune.yaml` | 611 bytes |
| MGA pretrained | `weights/mga-clap.pt` | 1,711,356,348 bytes |
| Expert 2 fine-tuned | `weights/finetuned_weights/expert2_best.pt` | 594,417,402 bytes |

Config LOCAL-03 dùng `freeze_policy: partial`; kiến trúc đang dùng HTSAT và `bert-base-uncased`. Trong restoration, `from_config()` cần cấu hình đã resolve; adapter không diễn giải lại đường dẫn tương đối. Tồn tại trên đĩa không đồng nghĩa checkpoint đã được deserialize hay tương thích.

## 9. Kết quả test

Mọi lệnh sau dùng Conda `nlp`; không có test nào mở checkpoint thật.

| Lệnh / suite | Kết quả |
|---|---:|
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_expert2_adapter.py' -v` | 24 test PASS |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_adapter_base.py' -v` | 13 test PASS |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_expert_import_isolation.py' -v` | 6 test PASS; vendor factory roots `NOT_EXERCISED` |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_expert1_adapter.py' -v` | 22 test PASS |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_moe_config.py' -v` | 21 test PASS |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_imports.py' -v` | 1 test PASS |
| `conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_*.py' -v` | **87 test PASS**, 61.039 giây |
| `conda run -n nlp --no-capture-output python3 -m compileall -q model/expert_adapters tests/test_expert2_adapter.py` | PASS |

Không có test skip hay failure ở kết quả cuối. Test import isolation xác nhận module symbols và thứ tự import; test báo rõ `models`, `tools`, `BEATs` factory path chưa được kích hoạt. Bài import smoke test cũ in cảnh báo deprecation từ `timm` và thông báo HF Hub không xác thực; suite vẫn qua, và không có model/checkpoint nào được khởi tạo hay nạp trong test đó. Cảnh báo này không phải bằng chứng về tương thích factory hoặc parity.

Nhóm test mới kiểm tra B=1/2/4; ánh xạ audio/caption và strip; một lần gọi model; nguyên score/features/order; dimension metadata giả lập; lỗi input/output; đóng băng/eval kể cả khi gọi `adapter.train()`; scoped config; missing resource; strict load với state nhỏ tổng hợp (missing/unexpected/shape mismatch/structured bundle); và phát hiện namespace sai nguồn. Test checkpoint giả lập **không** chứng minh tương thích checkpoint Expert 2 thật.

## 10. Khác biệt so với LOCAL-02.5

- Giữ interface và luồng đã đề xuất; dùng hook `_forward()` của base class.
- Không sửa package `__init__.py`, config schema hoặc model legacy: import module trực tiếp giữ package import nhẹ.
- Thêm yêu cầu offline trong factory và kiểm tra namespace bằng origin trước/sau factory, để tránh network access ngoài ý muốn và module cùng tên từ cây khác.
- Không bổ sung preprocessing, lớp trainable hoặc chuyển đổi score.

## 11. Rủi ro và giới hạn còn lại

- **Checkpoint runtime:** hai file tồn tại nhưng chưa được đọc; base restoration và strict fine-tuned load chưa được chứng minh.
- **Dimension/runtime outputs:** 2048 là suy luận từ YAML và code projection/codebook; chưa đo MSC embeddings hoặc đầu ra thật.
- **Score/features parity:** chưa so sánh với standalone Expert 2 trên audio thật.
- **Tokenizer/config cache:** MGA `TextEncoder` gọi `from_pretrained("bert-base-uncased")` khi tạo ASE (MGA `models/text_encoder.py` dòng 38–53). Offline factory sẽ fail rõ nếu tokenizer/config cache thiếu; cache chưa được kiểm tra ở LOCAL-05.
- **Namespace:** kiểm tra sai origin được unit-test, nhưng factory-level `models`/`tools` coexistence chưa thực thi.
- **Pretrained deserialization:** loader nguyên bản dùng unrestricted pickle. Chỉ truyền artifact nội bộ đáng tin cậy.
- **Warning ở legacy import smoke:** Hub warning không được dùng làm bằng chứng tải thành công/thất bại; kiểm tra import-only mới của E2 đặt offline env, nhưng runtime factory là phạm vi LOCAL-05.1.

## 12. Tệp thay đổi

Tạo mới:

- `model/expert_adapters/expert2_adapter.py`
- `tests/test_expert2_adapter.py`
- `docs/LOCAL-05/implementation-report.md`

Không sửa Expert 1, vendor MGA, Expert 3/4, `Basic_comp.py`, `MOE_model.py`, wrapper, dataset, `Train_MOE.py`, loss hoặc `configs/config.yaml`. Không sửa `model/expert_adapters/__init__.py`, checkpoint hoặc remote Git. Những untracked files có trước LOCAL-05 được giữ nguyên.

## 13. Sẵn sàng cho LOCAL-05.1

Có thể bắt đầu LOCAL-05.1: pretrained và fine-tuned Expert 2 artifact đã hiện diện theo đường dẫn đã ghi nhận, còn adapter/factory interface và kiểm tra contract đã được chuẩn bị. LOCAL-05.1 cần xác minh cache `bert-base-uncased`, chạy factory offline thật, kiểm tra namespace sau model construction, strict restore, xác nhận kích thước feature và thực hiện standalone-versus-adapter score/feature parity. Cho đến khi các bước đó hoàn tất, Expert 2 chưa được xem là runtime-verified.

## 14. Sẵn sàng cho LOCAL-06

LOCAL-06 có thể phát triển độc lập dựa trên adapter base và quy ước `Expert2Adapter`; không cần chờ Expert 2 real parity để thiết kế Expert 3. Tuy nhiên, tích hợp chung bốn expert/gating nên chờ các runtime blocker và namespace factory của từng expert được xác minh.

## Nguồn mã chính

- `expert2_finetune/expert2_model.py`: `MGAEncoder` 12–23; `Expert2` 26–71; `load_mga_state` 74–84; `create_expert2` 86–99.
- `expert2_finetune/Train_Expert2.py`: model creation 143–151; evaluation restoration 103–115; best raw state save 202–207; resume bundle 210–218.
- `expert2_finetune/expert2_data.py`: waveform/caption preprocessing 49–75; collation 78–85.
- `expert2_finetune/MGA-CLAP/settings/inference_example.yaml`: active embedding width 1; text/audio architecture 17–26.
- `expert2_finetune/MGA-CLAP/models/ase_model.py`: projection, codebook và encode output 103–144.
- `expert2_finetune/MGA-CLAP/models/audio_encoder.py`: HTSAT path và audio width 35–56.
- `expert2_finetune/MGA-CLAP/models/text_encoder.py`: tokenizer/model creation, max length và mask 38–75.
- `model/expert_adapters/base.py`: shared output validation và `_forward` contract 18–117.
- `configs/moe_config.py`: LOCAL-03 config schema/path resolution; `configs/moe_local.example.yaml`: proposed E2 resource keys.
- `model/expert_adapters/expert2_adapter.py`: namespace check 115–157; dynamic width 160–196; restore factory 259–385; config integration 387–438; freeze/eval and input/output path 470–540+.
