# LOCAL-10 — Báo cáo triển khai lõi gating MoE

## 1. Tóm tắt

**ĐẠT — ĐÃ KIỂM CHỨNG LÕI GATING TRAINABLE BẰNG KIỂM THỬ TỔNG HỢP.** Lõi gating giữ kiến trúc gốc, nhận đúng feature width 7936 từ registry hiện hành, tạo trọng số riêng theo từng mẫu và kết hợp bốn score theo tổng có trọng số của MOESCORE. Kiểm thử forward/backward xác nhận gradient đi vào gate, experts được đóng băng và ở eval.

Không khởi tạo chuyên gia thật, không nạp checkpoint, không chạy inference trên dữ liệu XACLE và không huấn luyện mô hình. Real expert output → gate integration được để lại cho LOCAL-10.1.

## 2. Phạm vi và trạng thái kho mã

Chỉ triển khai lõi gating nhẹ và lớp forward dùng adapter có sẵn. Không sửa expert source/adapter, loss, dataset, checkpoint hoặc mô hình gốc. Trước khi sửa, branch là `main`; remote giữ nguyên. Working tree đã có nhiều tệp chưa track từ các milestone trước cùng thay đổi có sẵn tại `expert1_finetune/Train_Expert1.py`; các nội dung đó được giữ nguyên.

Mốc regression trước LOCAL-10 theo báo cáo LOCAL-09.1 là 203/203. Lệnh baseline được khởi chạy trong phiên này trước phần triển khai, nhưng phần output thu được không chứa footer đếm test; vì vậy con số baseline bên dưới được ghi theo mốc đã báo cáo, còn kết quả đầy đủ sau thay đổi được xác nhận riêng ở mục 13:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Kết quả suite sau thay đổi được ghi ở mục 15 sau khi hoàn thành lần chạy cuối.

## 3. Kiến trúc gating

`model/moe_gating_core.py` cung cấp `GatingNetwork` với đúng chuỗi gốc:

```text
Linear(7936,512) → LayerNorm(512) → ReLU(inplace=True)
→ Dropout(0.1) → Linear(512,4) → Softmax(dim=-1)
```

Kích thước 7936 được lấy từ `EXPECTED_FEATURE_WIDTH` và số expert/thứ tự được lấy từ `EXPERT_ORDER`; không có một bảng registry thứ hai. Gate từ chối feature có sai chiều, dtype/device không tương thích hoặc giá trị không hữu hạn.

## 4. Hợp đồng forward và thứ tự

`MoEForwardIntegration` trong `model/moe_forward_integration.py` nhận bốn adapter đã tạo sẵn, kiểm tra chính xác key và metadata feature, gọi mỗi adapter đúng một lần theo thứ tự:

```text
expert1 → expert2 → expert3 → expert4
```

Nó tái sử dụng `validate_and_assemble_expert_outputs(...)` trong LOCAL-08. Tensor ghép là score `[B,4]` và features `[B,7936]`; kết quả là prediction `[B,1]` và weights `[B,4]`. Labels và metadata như model ID không được dùng làm đầu vào gate.

Prediction dùng nguyên công thức bốn hạng:

```python
score1 * weight1 + score2 * weight2 + score3 * weight3 + score4 * weight4
```

Không chuẩn hóa hoặc hiệu chỉnh score. Các kiểm thử B=1, B=2, B=4 xác nhận đúng thứ tự, contract, giá trị không bị sửa và parity với công thức tham chiếu.

## 5. Giải quyết 7168 so với 7936

`model/MOE_model.py` hiện ghi `gate_input_dim=7168`, trong khi registry và adapter đã xác minh tạo tổng feature 7936. LOCAL-10 không thay đổi file mô hình gốc đã được bảo vệ; integration nhẹ dùng registry để khởi tạo lớp đầu `[512,7936]`.

Vì shape trọng số khác, checkpoint gating cũ có lớp đầu `[512,7168]` không tương thích. Constructor không nhận 7168 và kiểm thử nạp state dict strict xác nhận tensor đầu vào 7168 bị từ chối do size mismatch. Không có `strict=False`, cắt hoặc đệm trọng số. Cần lưu ý `MOEModel` cũ chưa được chuyển sang adapter/feature contract mới; việc ghép production training pipeline thuộc milestone tích hợp tiếp theo.

## 6. Gating phụ thuộc mẫu

Một bài kiểm thử dùng tham số được kiểm soát tạo logits khác nhau khi tọa độ feature đầu tiên đổi dấu. Hai mẫu tạo weights khác nhau rõ rệt: mẫu A gần như chọn E1 (`[0.9999996, 1.15e-7, 1.15e-7, 1.15e-7]`), mẫu B gần như chọn E2 (`[1.15e-7, 0.9999996, 1.15e-7, 1.15e-7]`). Đây là chứng minh khả năng phụ thuộc đầu vào của kiến trúc trong test tổng hợp, không phải hành vi score thực nghiệm.

Trong eval, output của một mẫu khớp khi chạy riêng, trong batch chứa mẫu khác và khi đổi thứ tự batch, với sai khác tối đa quan sát được khoảng `2.98e-8` trong diagnostic CPU. LayerNorm hoạt động độc lập theo từng mẫu.

## 7. Trọng số và tính đúng của tổng

Với B=1/2/4, weights có shape `[B,4]`, hữu hạn, không âm; sai số tổng hàng lớn nhất đo được là `1.1920929e-7` (một ULP float32). Prediction khớp phép tính bốn hạng tham chiếu với sai số tuyệt đối lớn nhất/trung bình bằng `0.0`, trong `atol=1e-6, rtol=1e-5`; prediction nằm trong khoảng min/max score từng mẫu (convex combination). Score và features truyền vào không bị sửa đổi.

Điều này không khẳng định prediction luôn nằm trong `[-1,1]`; E3 có linear regression head không bị chặn theo source.

## 8. Đóng băng expert và gradient

Integration đặt mọi tham số expert thành `requires_grad=False`, chạy adapter forward trong `torch.no_grad()` và giữ adapter ở `eval`. Khi gọi `integration.train()`, gate chuyển sang train (Dropout 0.1 bật) nhưng toàn bộ cây expert vẫn eval. `integration.eval()` tắt dropout của gate và giữ expert eval.

Forward/backward tổng hợp cho thấy gate có gradient hữu hạn và có ít nhất một gradient khác zero; tham số expert không có gradient. Kiểm thử optimizer SGD đúng một bước tổng hợp xác nhận tham số gate thay đổi còn tham số expert giữ nguyên. Đây chỉ là kiểm tra khả năng cập nhật, không phải vòng huấn luyện.

## 9. Thang score của experts

Audit source, không khởi tạo model thật:

| Expert | Phép tính cuối | Kết luận từ source |
|---|---|---|
| E1 | Dot product giữa embedding audio/text LAION-CLAP đã chuẩn hóa L2 | Cosine-like, bị chặn `[-1,1]` nếu đầu ra embedding hữu hạn và chuẩn hóa đúng. |
| E2 | Dot product giữa embedding MSC đã chuẩn hóa L2 | Cosine-like, bị chặn `[-1,1]`. |
| E3 | Linear regression output, không có sigmoid/tanh/clamp cuối | Có thể không bị chặn. |
| E4 | Head hồi quy gốc kết thúc bằng tanh | Thuộc `(-1,1)` theo tính chất tanh. |

Không có bước score normalization chung trước tổng weighted sum và LOCAL-10 không bổ sung bước này. Softmax chỉ chuẩn hóa weights của gate. Do E3 không bị chặn theo kiến trúc, tổng convex không đảm bảo nằm trong `[-1,1]` nếu score E3 vượt khoảng.

Không đo min/max/mean/std score của expert thật. LOCAL-10.1 cần đo trên một tập mẫu train được phép, báo finite rate và phân bố riêng từng expert; không calibration trong phép đo.

## 10. Nhãn và loss

Unified Dataset lưu nhãn raw `0–10`; data pipeline gốc của MOESCORE chuyển target bằng `(score_raw - 5.0)/5.0`, còn inference chuyển ngược `prediction*5.0 + 5.0`. Vì vậy, raw label từ Unified Dataset chưa trực tiếp tương thích target loss gốc. LOCAL-10 giữ label ngoài đầu vào gating và không chuẩn hóa nhãn. LOCAL-11 phải đặt phép đổi tại ranh giới target/loss, tránh sửa thang nhãn của Dataset.

`ClippedMSELoss`, `ContrastiveLoss` và cách kết hợp loss không bị sửa. `Train_MOE.py` dùng diversity/entropy regularization với hệ số `0.2`; giá trị này được giữ nguyên.

## 11. State dict

Gating state dict strict save/load round-trip cho output eval giống hệt trong test. Shape của lớp đầu là `[512,7936]`, lớp cuối `[4,512]`. Bài test riêng xác nhận state dict kiểu cũ 7168 chiều không load strict được. Không lưu expert weights trong gating-only test state dict của `GatingNetwork`.

## 12. Synthetic test và diagnostic

- `tests/test_moe_gating.py`: **9/9 PASS**.
- `tests/test_moe_forward_integration.py`: **7/7 PASS**.
- `scripts/verify_moe_gating.py`: PASS trên CPU, seed 2027, B=1/2/4. Sai số tuyệt đối của weighted sum tham chiếu là 0 ở cả ba batch.
- JSON: `/tmp/moescore_local10_gating.json`.
- Diagnostic ghi rõ `model_construction=false`, `pretrained_model_construction=false`, `synthetic_adapter_stubs=true`, `checkpoint_deserialization=false`, `real_expert_inference=NOT_VERIFIED_LOCAL_10_1`, `real_score_distribution=NOT_MEASURED`, `full_moe_training=NOT_IMPLEMENTED`.

Lệnh diagnostic:

```bash
conda run -n nlp --no-capture-output python3 scripts/verify_moe_gating.py \
  --device cpu --batch-sizes 1,2,4 --seed 2027 --offline \
  --report-json /tmp/moescore_local10_gating.json
```

## 13. Kiểm thử hồi quy

- `tests/test_moe_gating.py`: **9/9 PASS**.
- `tests/test_moe_forward_integration.py`: **7/7 PASS**.
- `py_compile` cho hai module production, verifier và hai file test: **PASS** sau bản cập nhật cuối của verifier.
- Full suite sau thay đổi: **219/219 PASS**, 0 lỗi, 0 bỏ qua (`Ran 219 tests in 106.385s`). Lệnh:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Lần full suite đầu sau khi thêm test phát hiện phép kiểm tra tổng softmax đặt `atol=1e-7`, trong khi phép cộng float32 bốn trọng số lệch 1 ULP (`1.1920929e-7`). Ngưỡng được sửa thành `2e-7`, sát độ chính xác biểu diễn float32; không thay đổi mô hình hay logic softmax. Suite cuối đã chạy lại và đạt toàn bộ 219 test. Có `FutureWarning` sẵn có từ `timm.models.layers`; test import vẫn PASS.

## 14. Tệp thay đổi của LOCAL-10

Tạo mới:

- `model/moe_gating_core.py` — gate nhẹ, giữ cấu trúc gốc.
- `model/moe_forward_integration.py` — gọi adapter frozen, dùng assembler registry và tổng score gốc.
- `scripts/verify_moe_gating.py` — diagnostic synthetic/offline và JSON.
- `tests/test_moe_gating.py` — test kiến trúc, weights, sample dependence, gradient, state dict và legacy shape.
- `tests/test_moe_forward_integration.py` — test adapter stub, forward, freeze/eval và gradient.
- `docs/LOCAL-10/architecture-audit.md` — kiểm toán kiến trúc và score/label scales.
- `docs/LOCAL-10/implementation-report.md` — báo cáo triển khai.
- `docs/superpowers/plans/2026-11-02-local10-moe-gating.md` — kế hoạch thực thi nội bộ.

Không sửa các expert adapters, expert source, Dataset/Collate, loss, gate model gốc, checkpoint hoặc các tệp đã có trước trong working tree.

## 15. Giới hạn còn lại và sẵn sàng cho LOCAL-10.1

Đã xác minh kiến trúc và toán học bằng score/features synthetic cùng adapter stubs. Chưa xác minh được score distribution, output thật của bốn expert chạy vào gate, runtime với expert thật hoặc huấn luyện gate trên dữ liệu. Target scale vẫn cần phép chuyển đổi rõ ràng ở LOCAL-11. Checkpoint gating cũ 7168 chiều không tương thích.

**Sẵn sàng cho LOCAL-10.1 ở mức giao diện và toán học synthetic.** LOCAL-10.1 cần cung cấp output thật đã xác minh từ các experts (tuân thủ giới hạn tài nguyên hiện hành), đo phân bố score và chạy parity qua cùng gate; thành công ở stub không thay thế kiểm chứng đó.

