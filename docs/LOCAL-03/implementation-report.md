# LOCAL-03 — Báo cáo triển khai nền tảng Package và Configuration

## 1. Tóm tắt

Đã thêm package `model.expert_adapters` với interface chung không chứa model hoặc parameter mới, validator output, API đọc YAML/schema/path/resource riêng, file YAML mẫu và test kiểm tra import cô lập. Pipeline legacy và code của bốn expert được giữ nguyên. Toàn bộ 41 test trong `tests/` pass bằng Python 3.10 của Conda `nlp`.

Kết quả import chỉ xác nhận các module và class có thể được import trong các thứ tự đã kiểm tra. Các factory Expert2 và BEATs không được gọi, vì việc gọi chúng sẽ khởi tạo model/đọc checkpoint; do đó namespace `models`, `tools`, `BEATs` ở thời điểm dựng model vẫn **NOT_EXERCISED**.

## 2. Baseline repository state

- Branch khi bắt đầu: `main`. Các remote `origin` và `upstream` đã tồn tại; không thay branch hoặc remote.
- `git status --short` ban đầu: `?? docs/LOCAL-02/` và `?? docs/dataset-location.md`. Đây là thay đổi có trước, được giữ nguyên.
- Baseline `test_imports.py`: **PASS**, 1 test. Có cảnh báo `timm.models.layers` deprecated và Hugging Face Hub chưa xác thực; không có checkpoint/model nào được khởi tạo bởi task này.
- `utils.load_config()` hiện đọc JSON, trong khi `configs/config.yaml` là YAML; `Train_MOE.py` gọi loader legacy đó. LOCAL-03 không đổi giao diện legacy, cũng không làm `Train_MOE.py` phụ thuộc schema mới.

## 3. Package Adapter được triển khai

```text
model/expert_adapters/
├── __init__.py
└── base.py
```

`__init__.py` chỉ export `ExpertAdapter` và `validate_expert_output`. Import package không import các source Expert1–4. Không cần `model/__init__.py` mới để import package trong Python 3.10.

`ExpertAdapter(nn.Module, ABC)` lưu `feature_dim` nguyên dương, không thêm parameter, và cung cấp `forward(batch)` cuối cùng để tự kiểm tra output mỗi lần gọi. Adapter cụ thể cài `_forward(batch)`; public `forward()` yêu cầu tuple `(score, features)` rồi tự gọi validator. `validate_expert_output(...)` yêu cầu score đúng `[B,1]`, features `[B,D]`, cùng batch/device, kiểu floating point, batch không rỗng và mọi giá trị hữu hạn. Hàm trả lại đúng tensor đầu vào, không reshape, hiệu chỉnh hay chuẩn hóa. Vì vậy adapter tương lai không cần nhớ gọi validator bằng tay.

## 4. Hệ thống cấu hình

API mới ở `configs/moe_config.py`:

```python
config = load_moe_config(config_path)  # YAML safe-load + schema; không kiểm tra file model
resolved = resolve_moe_paths(config, config_path, ...)
validate_moe_resources(resolved)      # kiểm tra resource cục bộ một cách tường minh
```

`validate_moe_config()` xác nhận schema version 1, các section/expert bắt buộc, không nhận field lạ, đúng thứ tự expert, gate dimension/dropout, runtime/device, Expert1 fusion tắt, freeze policy Expert2, `concat_assignment` và định dạng SHA-256 Expert3. SHA `null` hợp lệ ở bước schema để template đọc được; resource validation yêu cầu SHA tin cậy trước khi restoration.

Thứ tự chọn root cho từng loại: đối số override → biến môi trường (`MOESCORE_ROOT`, `MOESCORE_DATA_ROOT`, `MOESCORE_RESOURCE_ROOT`) → `paths` trong YAML → mặc định. Root override tương đối theo working directory; root trong YAML tương đối theo thư mục chứa config. Path dữ liệu/resource tương đối theo root tương ứng; path tuyệt đối giữ nguyên. Mặc định `project_root` là thư mục config, còn `data_root` và `resource_root` fallback về project root. `resolve_moe_paths()` trả bản sao với các path đã thành `Path` tuyệt đối, không kiểm tra hay mở checkpoint.

`validate_moe_resources()` kiểm tra file/thư mục bắt buộc tồn tại và đọc được, báo field, loại resource và path đã resolve. Hàm chỉ yêu cầu SHA Expert3 có giá trị; nó không so sánh digest với nội dung file. LOCAL-03 cấm hash checkpoint nhiều GB, nên đối chiếu identity được để cho `MoescoreExpert3` khi restoration (source hiện hash và so sánh tại `.../models/moescore_expert3.py:L54-L58,L105-L116`). Hàm không deserialize hay thay checkpoint. Không có fallback pretrained-only.

`configs/moe_local.example.yaml` dùng path tương đối, thứ tự E1→E4, `enable_fusion: false`, Expert2 `partial`, Expert3 `released`, gate hidden size 512/dropout 0.1 và không có gate input dimension hard-code. Checkpoint Expert3 dùng placeholder, SHA để `null`; file mẫu parse và hợp lệ với schema nhưng strict resource validation sẽ từ chối cho đến khi path và SHA được thay bằng giá trị thật. Không thêm `configs/__init__.py`: namespace package import đã được test.

Không thêm cấu hình entropy hoặc thay đổi objective; hệ số entropy `0.2`, clipped MSE, contrastive loss và mọi training logic đều không bị chạm tới.

## 5. Kết quả kiểm tra import cô lập

Test chạy mỗi thứ tự bằng subprocess riêng, dùng interpreter hiện tại, thư mục gốc repository, timeout, biến môi trường offline cho Hugging Face và socket guard chặn kết nối mạng. Nó kiểm tra symbol/module origin, các vendor namespace nếu đã được nạp, cùng thay đổi `sys.path`/working directory, kể cả trường hợp chỉ reorder.

| Kịch bản | Kết quả import | Vendor namespace |
|---|---|---|
| Expert2 → Expert4 | **PASS** | `models`, `tools`, `BEATs`: **NOT_EXERCISED** |
| Expert4 → Expert2 | **PASS** | `models`, `tools`, `BEATs`: **NOT_EXERCISED** |
| Expert1 → Expert2 → Expert3 → Expert4 | **PASS** | `models`, `tools`, `BEATs`: **NOT_EXERCISED** |
| Expert4 → Expert3 → Expert2 → Expert1 | **PASS** | `models`, `tools`, `BEATs`: **NOT_EXERCISED** |
| Checker phát hiện reorder-only của `sys.path` | **PASS** | Test cố ý đảo list và xác nhận subprocess lỗi rõ ràng |

Các class import được từ đúng file thuộc source tree của expert. Khi import Expert1, dependency `laion_clap` tự thêm chính thư mục package cài đặt của nó vào `sys.path`; test ghi nhận thay đổi này và chỉ chấp nhận đúng thư mục đó. Expert2/Expert4 không tạo thay đổi `sys.path` ở module-import path được thử.

Source cho thấy `expert2_model.create_expert2()` mới chèn MGA source vào `sys.path` và import `models.ase_model` (`expert2_finetune/expert2_model.py:L86-L99`). BEATs được import trong `AudioEncoder.__init__()` (`expert4_finetune/audio_encoder.py:L20-L38`). Không gọi các hàm này để tránh dựng model/đọc checkpoint. Vì vậy test chưa xác nhận nguồn thực tế của `models`, `tools`, `BEATs` khi các factory chạy; đây là giới hạn đã ghi rõ, không phải PASS cho model-construction compatibility.

Import `model.expert_adapters` trong subprocess riêng **PASS**: không đổi `sys.path`/cwd, không khởi tạo CUDA và không nạp `laion_clap`, `models`, `tools`, `BEATs` hoặc `transformers`. Kiểm tra này quan sát side effect của package mới, không khẳng định mọi side effect của dependency ngoài.

Không có test nào bị skip. Ba namespace vendor được đánh dấu **NOT_EXERCISED**, không tính là tương thích đã xác minh.

## 6. Kết quả test

Các lệnh được chạy từ repository root bằng `/home/vitquay1708/miniconda3/envs/nlp/bin/python3` qua `conda run`.

| Lệnh | Kết quả |
|---|---|
| `python3 -m unittest discover -s tests -p "test_imports.py" -v` | **PASS**, 1 test; cảnh báo `timm` deprecated và HF Hub unauthenticated |
| `python3 -m unittest discover -s tests -p "test_adapter_base.py" -v` | **PASS**, 13 test |
| `python3 -m unittest discover -s tests -p "test_moe_config.py" -v` | **PASS**, 21 test trong lần chạy cuối toàn suite; gồm schema, path, precedence, fixture resource, safe YAML và example config |
| `python3 -m unittest discover -s tests -p "test_expert_import_isolation.py" -v` | **PASS**, 6 test: bốn thứ tự import, side-effect package và path reorder detection |
| `python3 -m unittest discover -s tests -p "test_*.py" -v` | **PASS**, 41 test, 0 failure |
| `python3 -m compileall -q model/expert_adapters configs/moe_config.py tests/test_adapter_base.py tests/test_moe_config.py tests/test_expert_import_isolation.py` | **PASS**, không có lỗi cú pháp |

Lệnh chạy lại chính xác phải dùng `conda run -n nlp --no-capture-output python3` thay cho `python3` trần nếu shell không ở môi trường `nlp`.

## 7. Khác biệt so với LOCAL-02.5

- Giữ config loader riêng thay vì thay `utils.load_config()` để tránh đổi hành vi script legacy.
- Path roots và schema được triển khai; không triển khai adapter/model factory, unified dataset, MoE refactor hoặc train config.
- Không thêm entropy field vào schema vì LOCAL-03 không cấu hình objective.
- Không sửa các import vendor; chỉ test import cấp module và ghi rõ phần factory chưa chạy.

## 8. Rủi ro và blocker

1. Import-level PASS không đảm bảo Expert2 MGA factory cùng tồn tại với BEATs factory. Cần test factory/origin khi có thể khởi tạo model trong task tích hợp.
2. Module `models`, `tools`, `BEATs` không được nạp trong import-only scenarios; các origin tương ứng là **NOT_EXERCISED**.
3. `laion_clap` dependency thay đổi `sys.path` khi import Expert1. Chưa có bằng chứng thay đổi này gây namespace collision, nhưng cần giữ trong tầm kiểm tra khi tích hợp factory.
4. Fine-tuned checkpoint chưa có ở local; strict restore, pretrained identity, prediction parity và dimension runtime chưa được kiểm tra.
5. Config/resource validation chưa đọc file để xác nhận expected SHA Expert3. Đây là giới hạn theo phạm vi LOCAL-03; restoration phải đi qua constructor Expert3 đang hash và so sánh `expected_checkpoint_sha256` (`expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py:L105-L116`).
6. Example YAML cố ý chưa dùng được cho strict restoration: fine-tuned paths là placeholder và SHA Expert3 chưa được cung cấp.
7. Test import legacy tiếp tục in cảnh báo `timm` và Hugging Face Hub; không sửa dependency hoặc pipeline trong LOCAL-03.

## 9. Sẵn sàng cho LOCAL-04

Foundation phù hợp để bắt đầu Expert1 Adapter: import package, base class, output validator và config API đều đã test độc lập. Hoàn tất kiểm chứng restore/parity vẫn cần checkpoint fine-tuned và dữ liệu mẫu. Trước khi dựng Expert2/Expert4 cùng tiến trình, cần kiểm tra origin khi factory thực sự import `models`/`tools`/`BEATs`; LOCAL-03 chưa xác nhận phần đó.

## 10. Files changed

Đã tạo:

- `model/expert_adapters/__init__.py`
- `model/expert_adapters/base.py`
- `configs/moe_config.py`
- `configs/moe_local.example.yaml`
- `tests/test_adapter_base.py`
- `tests/test_moe_config.py`
- `tests/test_expert_import_isolation.py`
- `docs/LOCAL-03/implementation-report.md`

Không sửa file có sẵn. Đặc biệt không sửa `utils.py`, `configs/config.yaml`, `Train_MOE.py`, `model/Basic_comp.py`, `model/MOE_model.py`, `model/wrapper.py`, `data/dataset.py`, loss hoặc source Expert1–4. Không tải/nạp checkpoint, không chạy model/training, không cài dependency, không thay đổi GPU memory strategy, không commit/push. Các thay đổi có trước trong `docs/LOCAL-02/` và `docs/dataset-location.md` được giữ nguyên.

## 11. Trạng thái

**PASS — FOUNDATION IMPLEMENTED AND TESTED.** Đây là trạng thái package/config/tests foundation; không xác nhận checkpoint restoration, simultaneous model construction, numerical parity hay khả năng training của MoESCORE.
