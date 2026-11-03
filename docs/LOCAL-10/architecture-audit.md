# LOCAL-10 — Kiểm toán kiến trúc gating và thang điểm

## 1. Kiến trúc gating gốc

Mã `GatingNetwork` trong [`model/MOE_model.py`](../../model/MOE_model.py#L11-L24) định nghĩa đúng chuỗi lớp:

```text
Linear(input_feature_dim, 512)
LayerNorm(512)
ReLU(inplace=True)
Dropout(0.1)
Linear(512, num_experts)
Softmax(dim=-1)
```

Trong `MOEModel`, số expert là 4. Gate nhận đặc trưng ghép, tạo trọng số riêng cho mỗi mẫu, sau đó mô hình trả `(final_score, weights)` ([`MOE_model.py`](../../model/MOE_model.py#L71-L89)).

Constructor gốc có mặc định `num_experts=3`, nhưng `MOEModel` truyền tường minh `num_experts=4`. Gate mới mặc định theo registry hiện tại là bốn và từ chối số expert khác để không lệch thứ tự adapter.

## 2. Hợp đồng đặc trưng và thứ tự

Registry hiện hành quy định thứ tự E1, E2, E3, E4 và kích thước tương ứng 1024, 2048, 3840, 1024 ([`registry.py`](../../model/expert_adapters/registry.py)). Tổng kích thước là:

```text
1024 + 2048 + 3840 + 1024 = 7936
```

Mã gốc hiện đặt `gate_input_dim = 7168` tại [`MOE_model.py`](../../model/MOE_model.py#L60-L65), nên không tương thích với adapter đã được xác minh. LOCAL-10 giữ nguyên đặc trưng chuyên gia và suy ra đầu vào gate từ `EXPECTED_FEATURE_WIDTH` của registry: `Linear(7936,512)`. Không cắt, chiếu, đệm hoặc đổi thứ tự đặc trưng.

Do ma trận lớp đầu đổi từ `(512,7168)` sang `(512,7936)`, checkpoint gate cũ có lớp đầu 7168 không tương thích trực tiếp. Việc nạp phải dùng kiểm tra kích thước nghiêm ngặt; LOCAL-10 không chuyển đổi một phần hay nạp với `strict=False`.

Assembler dùng lại `validate_and_assemble_expert_outputs(...)`, vốn xác thực contract rồi ghép score và feature theo `EXPERT_ORDER`; do đó score có dạng `[B,4]`, feature `[B,7936]`.

## 3. Công thức forward

Với đặc trưng `F` và điểm số gốc `s1...s4`:

```text
w = softmax(g_theta(F), dim=-1)
y = s1*w1 + s2*w2 + s3*w3 + s4*w4
```

Đầu ra là prediction `[B,1]` và weights `[B,4]`. Công thức bốn hạng giữ nguyên thứ tự và cách tính của mã gốc. Không có chuẩn hóa, hiệu chỉnh, clipping hoặc biến đổi score chung trước khi tính tổng.

Vì `w_i >= 0` và tổng trọng số bằng 1, với score hữu hạn, prediction nằm trong khoảng từ score nhỏ nhất tới score lớn nhất của cùng mẫu. Điều này không đảm bảo prediction thuộc `[-1,1]` nếu một expert có thể phát ra giá trị ngoài khoảng đó.

## 4. Ngữ nghĩa score của các expert

Đây là kết luận từ mã nguồn, không phải thống kê score thực tế:

| Expert | Phép tính cuối đã kiểm tra | Tính chất nguồn |
|---|---|---|
| E1 | Tích vô hướng hai embedding LAION-CLAP đã chuẩn hóa L2 | Tương tự cosine, thuộc `[-1,1]` khi embedding hữu hạn và chuẩn hóa đúng. Xem [`Train_Expert1.py`](../../expert1_finetune/Train_Expert1.py) và báo cáo LOCAL-04. |
| E2 | Tích vô hướng embedding MSC đã chuẩn hóa L2 | Tương tự cosine, thuộc `[-1,1]`. Xem [`expert2_model.py`](../../expert2_finetune/expert2_model.py). |
| E3 | Linear regression head không có sigmoid/tanh/clamp ở đầu ra | Không bị chặn theo cấu trúc. Xem [`moescore_expert3.py`](../../expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py#L244-L266). |
| E4 | Regression head kết thúc bằng `tanh` | Đầu ra thuộc khoảng mở `(-1,1)` về mặt toán học. Xem [`seq_coattn.py`](../../expert4_finetune/seq_coattn.py#L25-L32). |

Các đặc trưng embedding chuẩn hóa bên trong expert là hành vi riêng của từng expert. Softmax chuẩn hóa trọng số gating, không chuẩn hóa score. LOCAL-10 không đo phân bố score từ mô hình thật.

## 5. Nhãn và loss gốc

Unified Dataset giữ nhãn trung bình gốc ở thang `0–10`. Dataset cũ của MOESCORE ánh xạ target bằng `(mos - 5.0) / 5.0` ([`data/dataset.py`](../../data/dataset.py#L74-L87)); inference chuyển prediction về thang gốc bằng `pred * 5.0 + 5.0` ([`Inference_MOE.py`](../../Inference_MOE.py#L73-L80)). Vì vậy, nhãn Unified chưa trực tiếp ở thang target mà loss gốc chờ đợi. LOCAL-10 không nhận label trong gating core và không thực hiện phép đổi này. LOCAL-11 cần đặt chuyển đổi rõ ràng tại ranh giới tạo batch target cho loss.

Loss `ClippedMSELoss`, `ContrastiveLoss` và `CombineLosses` được giữ nguyên ([`loss_function.py`](../../loss/loss_function.py)). `Train_MOE.py` cộng diversity loss với hệ số `0.2` ([`Train_MOE.py`](../../Train_MOE.py#L21-L26), [`Train_MOE.py`](../../Train_MOE.py#L127-L132)); hệ số và công thức không bị thay đổi.

## 6. Đóng băng chuyên gia và gradient

Forward integration chạy các adapter một lần theo thứ tự registry trong `torch.no_grad()`, rồi cho gradient đi qua gating network. Mọi tham số adapter bị đóng băng và adapter luôn ở `eval`, kể cả khi gọi `integration.train()`. Gate vẫn chuyển sang train mode theo yêu cầu và giữ dropout gốc 0.1.

Kiểm thử synthetic xác nhận gradient gate hữu hạn và khác zero trong bài toán không suy biến; tham số chuyên gia không có gradient. Đây chưa phải huấn luyện trên dữ liệu XACLE.

## 7. Checkpoint và điểm cần xác minh tiếp

- Gate mới có input weight `[512,7936]`, output weight `[4,512]`.
- Checkpoint gate 7168 chiều phải bị từ chối khi nạp strict; không có chuyển đổi ngầm.
- LOCAL-10 chưa đo min/max/mean/std hay tỷ lệ score không hữu hạn của các expert thật.
- Kế hoạch LOCAL-10.1: dùng các ví dụ train được phép, chạy expert thật đã xác minh, ghi phân bố riêng từng expert (min, max, mean, std, finite rate), đối chiếu thang target và không sửa/calibrate score trong phép đo.
- Thống kê thực nghiệm không thay thế tính chất được suy ra từ phép tính trong source.

