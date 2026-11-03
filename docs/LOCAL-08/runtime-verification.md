# LOCAL-08.1 — Xác minh đồng tồn tại factory và bộ nhớ thực tế

## 1. Tóm tắt

**PAIRWISE = PASS — REAL BIDIRECTIONAL E2/E4 COEXISTENCE VERIFIED.** Hai thứ tự `Expert 2 → Expert 4` và `Expert 4 → Expert 2` được chạy trong hai tiến trình Python mới, độc lập. Mỗi tiến trình giữ đồng thời cả hai adapter đã khôi phục checkpoint thật. Cả hai expert chạy inference; module origins, trạng thái đóng băng/eval, parity trước/sau khởi tạo factory thứ hai và parity standalone–adapter đều đạt.

**FOUR_EXPERT = NOT_ATTEMPTED_RESOURCE_RISK.** Số đo sau khi giữ E2 và E4 cho thấy không đủ headroom GPU/RAM theo ngưỡng bảo thủ đã định trước. Không dựng E1/E3 cùng hai expert đó. Đây là quyết định không thử vì rủi ro tài nguyên, không phải bằng chứng rằng bốn expert chắc chắn không thể chạy trên phần cứng khác.

Không phát hiện xung đột namespace trong hai thứ tự đã chạy. Đây là xác minh coexistence thực tế cho E2/E4, không phải xác minh tích hợp MoE, Unified Dataset hay khả năng huấn luyện.

## 2. Môi trường

| Thuộc tính | Quan sát |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.1+cu130 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 3,895,132,160 bytes (~3.63 GiB) |
| RAM hệ thống | 16,552,214,528 bytes (~15.41 GiB) |
| CPU | 8 logical CPUs |
| Precision | FP32; không autocast/quantization |
| Offline | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` |
| Seed | 2027 |
| Parity tolerance | `atol=1e-6`, `rtol=1e-5` |

Các lần chạy runtime được ghi trong JSON lúc `2026-11-01T17:17–17:19 UTC` (02-11 theo giờ Jakarta). Hai kịch bản dùng process mới riêng biệt. Không tải tài nguyên qua mạng.

## 3. Tài nguyên và checkpoint

Tất cả checkpoint được đọc từ đường dẫn tuyệt đối trong cấu hình đã resolve. Factory gốc được gọi qua registry/adapters; verifier không tự thay thế checkpoint loader. Kích thước trong bảng là giá trị quan sát ở lần preflight; metadata file size/mtime trước và sau không đổi.

| Expert | Tài nguyên | Kích thước |
|---|---|---:|
| E2 | `weights/mga-clap.pt` | 1,711,356,348 B |
| E2 | `weights/finetuned_weights/expert2_best.pt` | 594,417,402 B |
| E2 | `expert2_finetune/MGA-CLAP/settings/inference_example.yaml` | 504 B |
| E2 | `expert2_finetune/expert2_finetune.yaml` | 611 B |
| E4 | `weights/BEATs_iter3_plus_AS2M.pt` | 361,499,833 B |
| E4 | `weights/laion-clap.pt` | 1,863,587,645 B |
| E4 | `weights/finetuned_weights/expert4_best.pt` | 155,455,153 B |

E2 factory báo khôi phục pretrained MGA thành công và fine-tuned checkpoint được strict-load thành công qua loader gốc. E4 factory báo BEATs strict loader, CLAP text/projection loader và original partial fine-tuned loader/tensor validation thành công. Không phát sinh identity mismatch hay fallback pretrained-only.

E4 dùng UniLM clone:

```text
/home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original
Commit: 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd
Source SHA-256: fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487
```

Commit và fingerprint quan sát khớp giá trị yêu cầu. Không dùng bundled `expert4_finetune/unilm/`.

## 4. Cách xác minh

Tạo [verify_multi_expert_runtime.py](../../scripts/verify_multi_expert_runtime.py) để mỗi lần chạy nhận một thứ tự hai expert, khởi tạo từng factory bằng registry hiện có, và ghi trạng thái theo stage vào JSON ngay trong quá trình chạy. Hai lệnh runtime thực tế:

```bash
conda run -n nlp --no-capture-output python3 scripts/verify_multi_expert_runtime.py \
  --config configs/moe_local.example.yaml \
  --unilm-source-dir /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original \
  --order expert2,expert4 --device cuda:0 --batch-size 1 --seed 2027 --offline \
  --report-json /tmp/moescore_local081_e2_e4.json
```

```bash
conda run -n nlp --no-capture-output python3 scripts/verify_multi_expert_runtime.py \
  --config configs/moe_local.example.yaml \
  --unilm-source-dir /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original \
  --order expert4,expert2 --device cuda:0 --batch-size 1 --seed 2027 --offline \
  --report-json /tmp/moescore_local081_e4_e2.json
```

Inputs FP32 được tạo tất định tại boundary adapter theo từng contract: E2 `[B,1,320000]` ở 32 kHz; E4 `[B,1,160000]` ở 16 kHz, cùng `int64 wav_lens`. Không có preprocessing chung hay resampling trong harness. Mỗi expert được chạy một lần trước khi dựng expert còn lại, rồi chạy lại bằng chính model instance và input cũ. Sau đó so sánh original forward với adapter forward trên cùng model instance. B=2 mixed-length E4 `[80000,160000]` được thêm để kiểm tra mask.

## 5. Kịch bản A — E2 → E4

**Kết quả: PASS.** E2 được dựng trước, inference baseline được lưu, sau đó E4 được dựng trong khi E2 vẫn nằm trong bộ nhớ. Hai expert chạy lại thành công.

| Kiểm tra | Kết quả |
|---|---|
| E2 original class | `expert2_finetune.expert2_model.Expert2` |
| E2 feature width / parameters | 2048 / 148,092,376 |
| E2 pretrained + fine-tuned restore | PASS; MGA strict load thành công |
| E4 original class | `expert4_finetune.model.Expert4` |
| E4 feature width / parameters | 1024 / 220,212,605 |
| E4 BEATs + CLAP + partial fine-tuned restore | PASS qua loader gốc |
| E2 score trước/sau tạo E4 | `[1,1]`, max absolute diff 0 |
| E2 feature trước/sau tạo E4 | `[1,2048]`, max absolute diff 0 |
| E2 standalone/adapter | score/feature max absolute diff 0; adapter gọi original forward 1 lần |
| E4 standalone/adapter | score/feature max absolute diff 0; adapter gọi original forward 1 lần |
| Module-origin/coexistence check | PASS; không có module object MGA bị thay |
| Trạng thái kết thúc | `SCENARIO_COMPLETE`, không lỗi |

Điểm số synthetic B=1 ghi nhận: E2 `0.1190353036`; E4 `-0.2966610193`. Shape standalone của E4 là `[1]`; adapter chuẩn hóa shape thành `[1,1]` mà không đổi giá trị. Feature shapes lần lượt là `[1,2048]` và `[1,1024]`.

## 6. Kịch bản B — E4 → E2

**Kết quả: PASS.** Đây là process độc lập mới. E4 được dựng và baseline inference chạy trước; sau đó E2 được dựng trong khi E4 vẫn còn sống. Cả hai chạy lại thành công.

| Kiểm tra | Kết quả |
|---|---|
| E4 original class / feature width | `expert4_finetune.model.Expert4` / 1024 |
| E4 pretrained + fine-tuned restore | PASS qua loader gốc |
| E2 original class / feature width | `expert2_finetune.expert2_model.Expert2` / 2048 |
| E2 pretrained + fine-tuned restore | PASS; MGA strict load thành công |
| E4 score trước/sau tạo E2 | `[1]`, max absolute diff 0 |
| E4 feature trước/sau tạo E2 | `[1,1024]`, max absolute diff 0 |
| E4 standalone/adapter | score/feature max absolute diff 0; một original forward |
| E2 standalone/adapter | score/feature max absolute diff 0; một original forward |
| Module-origin/coexistence check | PASS; private BEATs và MGA modules giữ đúng origin |
| Trạng thái kết thúc | `SCENARIO_COMPLETE`, không lỗi |

Điểm số synthetic B=1 ghi nhận: E4 `-0.2768948972`; E2 `0.1207399070`. Các output đều hữu hạn, FP32 và đúng shape. Điểm giữa hai scenario không dùng để so sánh chéo vì chúng là hai process/input generation độc lập; mỗi phép so sánh parity chỉ dùng chính input và model instance tương ứng.

## 7. Namespace và UniLM

Trong cả hai scenario, MGA modules quan sát được, gồm `models`, `models.ase_model`, `models.audio_encoder`, `models.text_encoder`, `models.cnns`, `models.htsat` và `models.feature_extractor`, đều xuất phát từ:

```text
expert2_finetune/MGA-CLAP/models/
```

BEATs private package cùng `BEATs.py`, `backbone.py`, `modules.py` đều resolve dưới `unilm_expert4_original/beats/`. Module identity của MGA giữ nguyên trước/sau E4 factory. `laion_clap` resolve về package cài trong Conda `nlp`.

Scoped BEATs bridge đã thực sự chạy trong factory. Báo cáo xác nhận top-level aliases `BEATs`/`backbone`/`modules` được khôi phục sau scope, còn private package modules được giữ lại để model instance tiếp tục hoạt động. Không có alias không liên quan bị ghi đè. Chạy độc lập cả hai thứ tự không tái hiện import collision.

## 8. Mixed-length và mask E4

Mixed-length B=2 được kiểm tra trong cả hai initialization order:

```text
waveform: [2,1,160000], float32
valid lengths: [80000,160000], int64
score standalone: [2]
score adapter: [2,1]
fused features: [2,1024]
scores: [-0.1131708100, -0.0705822632]
```

Lengths được chuyển nguyên vẹn vào AudioEncoder. Mask padding ban đầu có shape `[2,160000]`, true counts `[80000,0]`; temporal reductions quan sát được cho `[2,998]` với counts `[498,0]`, rồi `[2,496]` với counts `[246,0]`. Mask cuối là bool, padding nằm ở suffix, số vị trí hợp lệ `[250,496]`. SeqCoAttn nhận masks không đổi; `MultiheadAttention.key_padding_mask` trùng với source masks. Text mask `[2,76]` có `[67,67]` vị trí pad. Encoder/audio, text encoder, SeqCoAttn, hai hướng attention và regression head được gọi một lần mỗi original forward (pool hai lần theo hai nhánh).

Standalone–adapter score và feature parity đều có max absolute diff `0.0`. Hai nhánh fused `[2,512]` ghép thành `[2,1024]`; mỗi block so với phần tương ứng của original output sai khác `0.0`. Không thay đổi masking hay toán tử mô hình.

## 9. Frozen/eval

Sau khôi phục và sau mô phỏng `adapter.train()`:

| Expert | Tổng parameters | Trainable | Adapter-only trainable | Tất cả module eval |
|---|---:|---:|---:|---|
| E2 | 148,092,376 | 0 | 0 | Có |
| E4 | 220,212,605 | 0 | 0 | Có |

Lần inference lặp lại cho score và feature của mỗi expert khớp tuyệt đối (`max_abs_diff=0`). Không tạo optimizer, không backward.

## 10. Bộ nhớ đo được

Đơn vị trong bảng là MiB (1 MiB = 1,048,576 bytes); số liệu gốc byte nằm trong JSON. RSS là mức dùng hiện tại tại stage; CUDA peak là peak PyTorch allocation tích lũy trong process đến stage đó. Đây là giá trị đo trực tiếp từng process, không phải tổng từ peak độc lập.

| Scenario / stage | Process RSS | RAM available | CUDA allocated | CUDA reserved | CUDA peak allocated | VRAM free |
|---|---:|---:|---:|---:|---:|---:|
| E2→E4 / start | 604.5 | 5543.8 | 0 | 0 | 0 | 3561.2 |
| E2→E4 / after E2 factory | 1801.6 | 4771.8 | 569.9 | 618.0 | 569.9 | 2943.2 |
| E2→E4 / after E2 inference | 1961.5 | 4687.9 | 578.1 | 672.0 | 620.4 | 2869.2 |
| E2→E4 / after E4 factory | 3288.5 | 3822.2 | 1421.5 | 1540.0 | 1421.5 | 2001.2 |
| E2→E4 / scenario complete | 3303.6 | 3785.1 | 1439.5 | 1806.0 | 1587.3 | 1725.2 |
| E4→E2 / start | 604.4 | 6253.2 | 0 | 0 | 0 | 3561.2 |
| E4→E2 / after E4 factory | 2171.6 | 5042.5 | 844.2 | 912.0 | 844.2 | 2649.2 |
| E4→E2 / after E4 inference | 2360.9 | 4925.3 | 870.3 | 1026.0 | 954.8 | 2507.2 |
| E4→E2 / after E2 factory | 3930.4 | 4715.2 | 1445.7 | 1538.0 | 1445.7 | 1995.2 |
| E4→E2 / scenario complete | 3938.7 | 4406.3 | 1445.7 | 1792.0 | 1592.9 | 1739.2 |

Process-wide peak RSS E2→E4 là 4,242,522,112 B (~4046.0 MiB); E4→E2 là 4,496,060,416 B (~4287.8 MiB). Final CUDA allocated peak lần lượt là 1,664,364,544 B (~1587.3 MiB) và 1,670,271,488 B (~1592.9 MiB). Final reserved peak lần lượt là 1,893,728,256 B (~1806 MiB) và 1,879,048,192 B (~1792 MiB). Free VRAM cuối lần lượt 1,809,055,744 B (~1725.2 MiB) và 1,823,735,808 B (~1739.2 MiB).

Trình đo báo cáo đủ current/peak allocation, reservation, free VRAM, RSS, system available RAM và swap cho từng stage. Không thấy OOM trong hai pairwise runs. Các cảnh báo deprecation của `torch.nn.utils.weight_norm`/`timm.models.layers` và cảnh báo `torch.meshgrid` được ghi nhận; chúng không làm factory hoặc inference thất bại và không dẫn đến sửa source.

## 11. Khả năng chạy cả bốn expert

Sau mỗi pairwise run, gate an toàn đánh giá tài nguyên bổ sung dựa trên free VRAM/RAM quan sát được và tham chiếu peak/checkpoint E1+E3. Báo cáo tổng hợp được tạo bằng:

```bash
conda run -n nlp --no-capture-output python3 scripts/verify_multi_expert_runtime.py \
  --assess-four-expert-resources \
  /tmp/moescore_local081_e2_e4.json /tmp/moescore_local081_e4_e2.json \
  --report-json /tmp/moescore_local081_four_experts.json
```

Ngưỡng bảo thủ yêu cầu 2,640,436,736 B GPU headroom; mức thấp nhất quan sát sau pair là 1,809,055,744 B. Host headroom cần 5,403,152,311 B gồm payload checkpoint E1/E3 4,329,410,487 B cộng 1 GiB margin; available thấp nhất sau pair là 3,968,991,232 B. Cả hai điều kiện đều không đạt. Vì vậy `FOUR_EXPERT_COEXISTENCE=NOT_ATTEMPTED_RESOURCE_RISK`; không tạo process hoặc nạp checkpoint E1/E3. Những giá trị peak độc lập chỉ là tham chiếu cho gate, không được cộng thành ước lượng chính xác bộ nhớ joint.

## 12. Kiểm thử và hồi quy

Baseline trước khi thêm verifier/test mới:

```text
python3 -m unittest discover -s tests -p 'test_*.py' -v
Ran 148 tests — OK
```

Kết quả sau thay đổi:

| Lệnh | Kết quả |
|---|---|
| `python3 -m unittest discover -s tests -p 'test_multi_expert_runtime.py' -v` | 11/11 PASS |
| `python3 -m unittest discover -s tests -p 'test_expert_factory_coexistence.py' -v` | 12/12 PASS |
| `python3 -m unittest discover -s tests -p 'test_*.py' -v` | 159/159 PASS, 0 fail, 0 skip |
| `python3 -m py_compile scripts/verify_multi_expert_runtime.py tests/test_multi_expert_runtime.py` | PASS |

Full suite mất 132.696 s. Import-order tests chạy subprocesses và đã qua. Có warning `timm.models.layers` trong test import smoke; test vẫn pass.

## 13. Tệp được tạo/thay đổi

Tạo trong milestone này:

- `scripts/verify_multi_expert_runtime.py` — pairwise runtime verifier và bốn-expert resource gate.
- `tests/test_multi_expert_runtime.py` — 11 test cho parsing, stage/JSON reporting, memory, error classification, parity và resource gate.
- `docs/LOCAL-08/runtime-verification.md` — báo cáo này.

Không sửa original expert, model/checkpoint, config, gating, loss hoặc các adapter đã xác minh. Trạng thái repository trước/sau còn một file modified có sẵn `expert1_finetune/Train_Expert1.py`; file đó không thuộc thay đổi của LOCAL-08.1 và được giữ nguyên. Branch vẫn `main`; không stage/commit/push.

## 14. Giới hạn còn lại

- Chỉ E2/E4 được dựng đồng thời bằng factory thật. E1/E3 không được nạp chung trong process này.
- Không chạy inference validation audio ở milestone này; input là deterministic synthetic boundary tensors. E4 mixed-length masking được thử trực tiếp bằng `[80000,160000]`.
- Chưa kiểm tra coexistence factory của đủ bốn expert, full-validation metrics, Unified Dataset, MoE forward, gating hoặc huấn luyện.
- Pairwise GPU footprint không chứng minh cấu hình GPU khác có/không chạy được bốn expert.
- Mọi nguồn pretrained vẫn phụ thuộc vào checkpoint provenance/cache đã cấu hình; verifier dùng original loader và không tự tải thay thế.

## 15. Sẵn sàng cho LOCAL-09

Có thể tiếp tục xây dựng bước tích hợp trên kết quả pairwise: registry/factory E2–E4 đã đồng tồn tại theo hai hướng và giữ nguyên đầu ra. Trước khi xác nhận runtime bốn expert hoặc huấn luyện MoE, cần chọn một cấu hình tài nguyên được phê duyệt và đo lại an toàn khi nạp E1/E3; không nên giả định GTX 1650 hiện tại có headroom từ các số đo pairwise. LOCAL-09 vẫn cần test input chuẩn hóa theo từng chuyên gia, output assembly thật nếu resource cho phép, và tích hợp gating riêng. Chưa tuyên bố full MoE ready-for-training.
