# LOCAL-08 — Nền tảng tích hợp bốn expert

## 1. Tóm tắt

**Trạng thái: `PASS — LIGHTWEIGHT INTEGRATION FOUNDATION VERIFIED`.**

Đã thêm registry lazy cho bốn adapter, hợp đồng input/output có thứ tự cố định, hàm kiểm tra/ghép tensor đầu ra, diagnostic CLI không khởi tạo model, và kiểm thử import order bằng subprocess. Bốn adapter stub cùng tồn tại trong một process ở B=1/2/4; score và feature được ghép lần lượt thành `[B,4]` và `[B,7936]`.

Đây chỉ là bằng chứng cho import adapter nhẹ, factory mapping, và coexistence của stub. **Không gọi factory pretrained thật, không giải mã checkpoint, không đo bộ nhớ đồng thời và không chạy inference đa expert thật.** Các việc đó thuộc LOCAL-08.1.

## 2. Baseline repository

- Nhánh: `main`; HEAD trước khi làm: `6dce165236290142452e48f6d2a2c45c6359ad31`.
- Remotes `origin` và `upstream` được giữ nguyên.
- Trước khi sửa, workspace có nhiều cây/file `??` của các milestone LOCAL-02–07 và weights; không reset, xóa, stage, commit hay push chúng.
- Baseline chạy bằng Conda `nlp`: **136/136 PASS**, không skip/failure.
- Môi trường: Python 3.10.21, PyTorch 2.14.1+cu130, Transformers 5.17.0; CUDA khả dụng. LOCAL-08 không khởi tạo model nặng.

## 3. Kiểm kê factory bốn expert

| Expert | Adapter / model gốc / factory | Pretrained và fine-tuned resources | Input adapter | Score / feature |
|---|---|---|---|---|
| E1 | `Expert1Adapter`; `expert1_finetune.Train_Expert1.Expert1`; adapter `from_config`/`from_finetuned` | `weights/laion-clap.pt`; `weights/finetuned_weights/expert1_best.pt` | `expert1_wavs [B,1,480000]`, 48 kHz; `captions_raw` giữ nguyên | cosine gốc; `[B,1]`, feature audio rồi text `[B,1024]` |
| E2 | `Expert2Adapter`; `expert2_finetune.expert2_model.Expert2`; `create_expert2` + `load_mga_state`, qua adapter factory | `weights/mga-clap.pt`; `weights/finetuned_weights/expert2_best.pt`; MGA source và YAML | `expert2_wavs [B,1,320000]`, 32 kHz; caption theo chính sách standalone đã xác minh | cosine MSC; `[B,1]`, audio/text MSC `[B,2048]` |
| E3 | `Expert3Adapter`; `MoescoreExpert3`; constructor gốc và `load_best_checkpoint` partial | `weights/checkpoint-30.pth`; `weights/finetuned_weights/expert3_best.pt`; source/config Expert3 | `expert3_wavs [B,1,160000]`, 16 kHz; raw caption qua đường source | regression score gốc; `[B,1]`, `released` cross-feature `[B,3840]` |
| E4 | `Expert4Adapter`; `expert4_finetune.model.Expert4`; `inference.load_finetuned` → `Expert4.from_pretrained` | BEATs, LAION-CLAP, fine-tuned partial; UniLM clone độc lập | `expert4_wavs [B,1,160000]`; `expert4_wav_lens [B] int64`; caption `.strip()` | tanh gốc, adapter chỉ chuẩn hóa rank; `[B,1]`, fused `[B,1024]` |

Nguồn chính gồm `expert1_finetune/Train_Expert1.py`, `expert2_finetune/expert2_model.py`, `expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py`, `expert4_finetune/model.py`, `expert4_finetune/inference.py`, cùng các báo cáo runtime LOCAL-04.1–07.1-FIX. Các chi tiết và parity của model thật được lấy từ những báo cáo đó; LOCAL-08 chỉ xác nhận metadata tích hợp và resource path, không tải checkpoint.

## 4. Kiểm toán namespace import

| Namespace | Sử dụng / rủi ro | Kết luận hiện tại |
|---|---|---|
| `models` | MGA-CLAP dùng import top-level `models.*`, factory thêm MGA source vào `sys.path` và cần origin đúng MGA tree | E2 runtime độc lập đã xác minh origin MGA; import wrapper nhẹ của E2 không nạp namespace này. Cùng-process coexistence với factory E4 chưa chạy. |
| `tools` | Có thể xuất hiện ở code/vendor MGA phụ trợ | Không được import trong E2 runtime đã báo cáo; origin chưa được quan sát, không khẳng định có collision. |
| `BEATs`, `backbone`, `modules` | UniLM gốc dùng import tuyệt đối; BEATs có relative subimports | E4 có scoped bridge/private package, kiểm tra origin và phục hồi alias/path exception-safe; unit tests E4 kiểm tra bridge/import order/cleanup. Import adapter không kích hoạt bridge. |
| `laion_clap` | E1 và E4 cùng sử dụng LAION-CLAP dependency/checkpoint ecosystem | Hai factory độc립 runtime đều thành công, nhưng đồng thời load trong cùng process chưa kiểm tra. Không có collision nào được tái hiện trong lightweight adapter import. |
| `m2d`, `transformers`, `bert`/`roberta` | E3 dùng M2D package-relative và Transformers; text encoders cũng là dependencies | Không thấy xung đột ở adapter import; thư viện chung được xem là dependency dùng chung, không phải bằng chứng factory tương thích. |
| `sys.path`/`sys.modules` | MGA và BEATs cần quan tâm namespace/path riêng | Các import-order subprocess ghi nhận `sys_path_unchanged=true`, trạng thái alias `BEATs` giữ nguyên, không có vendor module ngoài dự kiến được nạp. |

`validate_module_origins()` trong `model/expert_adapters/registry.py` kiểm tra `__file__`, `__spec__.origin`, `__path__` và báo rõ expected/observed origin; helper không sửa `sys.modules`. Test dùng module origins có kiểm soát. Không có namespace conflict thật nào được tái hiện trong LOCAL-08.

## 5. E2/E4 và danh tính UniLM

Expert 2 dùng MGA top-level namespace `models`; Expert 4 BEATs dùng alias top-level có scope. Kiểm tra import adapter theo thứ tự không thay thế hai namespace đó. Tuy nhiên, **import-only không kiểm chứng thứ tự khởi tạo factory**.

Expert 4 checkpoint yêu cầu clone UniLM riêng có commit `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` và source SHA-256 `fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487`. Diagnostic chạy `expert4_finetune.trainer._source_identity()` trên `/home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original`; commit và SHA quan sát đều khớp. `configs/moe_local.example.yaml` chỉ chứa placeholder `/path/to/unilm_expert4_original`, không ghim đường dẫn máy cá nhân. CLI nhận clone qua `--unilm-source-dir`; không dùng bundled `expert4_finetune/unilm/` làm fallback.

## 6. Import-order isolation test matrix

Tám thứ tự được chạy trong subprocess sạch: E1→E2→E3→E4; E4→E3→E2→E1; E2→E4; E4→E2; E1→E4; E4→E1; E2→E3; E3→E2. Kết quả **8/8 PASS**.

Hai lần diagnostic CLI riêng cũng thành công:

1. `expert1,expert2,expert3,expert4`
2. `expert4,expert3,expert2,expert1`

Cả hai xác nhận adapter class/factory được đăng ký, import không gọi factory, không deserialize checkpoint, không nạp LAION CLAP hoặc vendor module ngoài ý muốn, không đổi `sys.path`, và trạng thái tạm BEATs alias không thay đổi. Đây là `IMPORT_ORDER_VERIFIED`, không phải `REAL_FACTORY_COEXISTENCE_VERIFIED`.

## 7. Registry và ordering

`model/expert_adapters/registry.py` định nghĩa:

```python
EXPERT_ORDER = ("expert1", "expert2", "expert3", "expert4")
EXPERT_FEATURE_DIMS = {"expert1": 1024, "expert2": 2048,
                       "expert3": 3840, "expert4": 1024}
EXPECTED_FEATURE_WIDTH = 7936
```

Registry immutable, ánh xạ tên đến adapter module/class và resolve lazy. Import registry không dựng expert. `create_expert_adapter()` chỉ gửi section expert tương ứng cùng runtime vào factory hiện hữu; sau đó kiểm tra class và `feature_dim`. Quy trình này dùng contract metadata đã runtime-verified, không ép/chiếu feature của model.

`EXPERT_INPUT_CONTRACTS` ghi riêng sample rate, waveform key/shape, caption policy và E4 valid lengths. Nó không decode/resample/crop/pad audio; mỗi adapter/dataset gốc vẫn sở hữu preprocessing của mình.

## 8. Hợp đồng output

`validate_and_assemble_expert_outputs()` yêu cầu chính xác bốn expert và xác thực score/features bằng validator chung: rank, batch, width, finiteness, device và dtype. Thứ tự mapping đầu vào không ảnh hưởng đến kết quả; luôn dùng `EXPERT_ORDER`.

| Thành phần | Shape |
|---|---|
| Mỗi score | `[B,1]` |
| score assembly | `[B,4]` theo E1,E2,E3,E4 |
| feature assembly | `[B,7936]` theo E1,E2,E3,E4 |

Không tính score cuối, không average/vote, không gating/softmax, không chuẩn hóa hoặc sửa tensor gốc.

## 9. Stub coexistence

Bốn adapter thật nhưng bao quanh original-model stubs deterministic đã cùng sống trong một process. B=1, B=2 và B=4 đều qua. Mỗi forward gọi stub gốc đúng một lần; input từng expert được map đúng, bao gồm E4 lengths không đổi. Kiểm tra parameters frozen, mode eval qua parent `.train()`, output ordering và repeated inference.

Negative tests bao gồm thiếu/thừa expert, score/feature shape sai, batch mismatch, khác device/dtype, NaN/Inf và origin conflict. Kết quả được phân loại `STUB_MULTI_EXPERT_COEXISTENCE_VERIFIED`; không suy ra real pretrained coexistence.

## 10. Cấu hình và kiểm tra resources

Tái sử dụng schema LOCAL-03 và `resolve_moe_paths()`. `configs/moe_local.example.yaml` trỏ tới các local pretrained/fine-tuned artifact và khai báo Expert3 base SHA mong đợi; UniLM dùng placeholder có hướng dẫn để cấu hình clone có identity đúng. CLI nhận `--unilm-source-dir` và chỉ kiểm tra resource path đọc được, không `torch.load`/deserialize.

Hai lần chạy sau đây đều trả `PASS`, không có `failures`, và 15/15 resource khai báo tồn tại/đọc được:

```bash
conda run -n nlp --no-capture-output python3 scripts/check_expert_factory_coexistence.py \\
  --mode lightweight --order expert1,expert2,expert3,expert4 \\
  --config configs/moe_local.example.yaml \\
  --unilm-source-dir /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original \\
  --offline --report-json /tmp/moescore_local08_forward_final.json

conda run -n nlp --no-capture-output python3 scripts/check_expert_factory_coexistence.py \\
  --mode lightweight --order expert4,expert3,expert2,expert1 \\
  --config configs/moe_local.example.yaml \\
  --unilm-source-dir /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original \\
  --offline --report-json /tmp/moescore_local08_reverse_final.json
```

Mỗi JSON ghi `real_factory_coexistence=NOT_VERIFIED_REAL_FACTORY`, `real_inference=NOT_VERIFIED_REAL_INFERENCE`, `memory=NOT_VERIFIED_MEMORY`, `heavy_checkpoints_deserialized=false`, `real_models_constructed=false`. Hai báo cáo JSON được lưu trong `/tmp`.

Resource inventory đã xác nhận 15 resource path trong config có mặt/đọc được. Các file checkpoint hiện diện gồm: LAION-CLAP 1,863,587,645 byte; MGA-CLAP 1,711,356,348; M2D 1,694,677,655; BEATs 361,499,833; fine-tuned E1 635,384,245; E2 594,417,402; E3 135,760,942; E4 155,455,153. Thông tin size/mtime lấy từ filesystem; không deserialize checkpoint trong LOCAL-08.

## 11. Kế hoạch tài nguyên cho LOCAL-08.1

Các runtime riêng trước đó báo peak allocated xấp xỉ E1 726,977,536 byte, E2 691,505,152 byte, E3 839,717,376 byte, E4 1,067,779,584 byte trên GPU GTX 1650 khoảng 3.90 GB VRAM. Đây là peak của từng quy trình riêng, **không phải** tổng resident memory đồng thời và không chứng minh bốn expert có thể cùng nằm trên GPU.

LOCAL-08.1 nên đo trong process mới: RAM RSS, free/total CUDA trước mỗi factory, `max_memory_allocated` và free VRAM sau mỗi model/forward, kiểm tra lại resource size/hash theo loader gốc, ghi stage và traceback. Dùng từng expert model duy nhất mỗi process/order; không tự giảm precision/độ dài hoặc offload. Phân loại riêng import collision, factory init, base/fine checkpoint identity/loading, thiếu resource, CUDA OOM, RAM limit, parity failure và input contract mismatch.

## 12. Kế hoạch kiểm thử factory thật cho LOCAL-08.1

Hai ca bắt buộc dùng process độc lập:

1. **E2→E4:** dựng E2 bằng factory gốc, kiểm tra `models.*` origin; dựng E4 với UniLM clone đã xác minh, kiểm tra `BEATs/backbone/modules`, chạy input tối thiểu từng adapter, đối chiếu standalone baseline, ghi RAM/VRAM.
2. **E4→E2:** chạy đảo thứ tự, lặp lại origin checks sau mỗi factory và parity từng expert.

Trước hết làm pairwise với batch nhỏ trong FP32 nếu tài nguyên cho phép; không giả định tất cả model đồng thời khả thi. Chỉ thử bốn model nếu đo VRAM/RAM sau pairwise cho thấy đủ headroom. So sánh score/feature với standalone runtime tolerance đã dùng cho mỗi expert, kiểm tra input E4 mixed lengths, freeze/eval và forward counts. Bất kỳ factory failure nào phải dừng đúng stage và không được coi import-only PASS là thành công.

## 13. Kiểm thử LOCAL-08

- `test_expert_factory_coexistence.py`: **12/12 PASS**; bao gồm 8 import orders, registry, config scoping, origin checker, output checker và stubs B=1/2/4.
- `test_moe_config.py`: **21/21 PASS** sau cập nhật config mẫu.
- Diagnostic CLI chạy xuôi/ngược, JSON hợp lệ, exit code 0.
- `compileall` trên adapters, harness và test mới: PASS.
- Baseline trước thay đổi: 136 tests PASS.
- Full suite sau thay đổi: **148/148 PASS**, 0 failed, 0 skipped.

Lệnh full suite:

```bash
conda run -n nlp --no-capture-output python3 \\
  -m unittest discover -s tests -p 'test_*.py' -v
```

Lệnh focused và compile:

```bash
conda run -n nlp --no-capture-output python3 \\
  -m unittest discover -s tests -p 'test_expert_factory_coexistence.py' -v
conda run -n nlp --no-capture-output python3 \\
  -m unittest discover -s tests -p 'test_moe_config.py' -v
conda run -n nlp --no-capture-output python3 \\
  -m compileall -q model/expert_adapters \\
  scripts/check_expert_factory_coexistence.py \\
  tests/test_expert_factory_coexistence.py
```

Full suite mất 130.291 giây. Có một `FutureWarning` sẵn từ `timm.models.layers` trong import smoke test; test vẫn PASS. Các dòng `NETWORK_BLOCKED` do import-isolation helper chạy thử giới hạn network và xác nhận import không cần tải mạng; chúng không phải lỗi suite.

## 14. Files changed

- `model/expert_adapters/registry.py` — registry, input/output contracts, origin validation, output assembly.
- `scripts/check_expert_factory_coexistence.py` — lightweight JSON diagnostic CLI.
- `tests/test_expert_factory_coexistence.py` — import order, registry, stubs, negative tests, diagnostic.
- `configs/moe_local.example.yaml` — local fine-tuned paths/Expert3 SHA và UniLM clone placeholder có hướng dẫn.
- `tests/test_moe_config.py` — kiểm tra các giá trị config mẫu cập nhật.
- `docs/LOCAL-08/integration-design.md` — báo cáo này.

Các expert gốc, adapters E1–E4, checkpoints, gate/loss, remotes và nhánh không bị sửa trong task này.

## 15. Rủi ro còn lại

- MGA `models.*` và BEATs aliases chưa được kiểm tra sau khi **cả hai factory thật** cùng chạy trong process.
- LAION-CLAP E1/E4 shared dependency khi load đồng thời chưa được thử.
- UniLM identity trong lightweight CLI khớp clone đã truyền rõ; config mẫu để placeholder cho tính di động.
- Không đo joint RAM/VRAM; peak riêng không cộng được thành resident estimate đáng tin.
- Không kiểm tra full MoE forward, gating/training, Unified Dataset, full validation metrics hoặc score fusion.

## 16. Sẵn sàng cho LOCAL-08.1

Nền tảng nhẹ sẵn sàng. LOCAL-08.1 có đủ registry, config và origin diagnostics để chạy E2→E4/E4→E2 trong process cô lập. Trạng thái thực tế hiện tại vẫn là **factory coexistence, multi-expert inference và memory feasibility chưa xác minh**.
