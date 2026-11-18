# MOE-09.3 — Sửa paired-cross từ 5D sang 4D+1

Sáu expert dùng `build_paired_cross_features()` nay trả về:

```text
[audio, text, |normalize(audio)-normalize(text)|,
 product(normalize(audio), normalize(text)), cosine_scalar]
```

Cosine vẫn tính từ hai embedding L2-normalized, nhưng chỉ nối một scalar thay vì lặp lại `D` lần. Hai embedding đầu và phép tính score của expert không đổi. Expert 3 giữ nguyên feature 5D cùng head đã huấn luyện; Expert 4 và M2D-LoRA cũng giữ nguyên biểu diễn hiện tại.

| Expert | Feature width hiện tại |
|---|---:|
| `expert1` | 2049 |
| `expert2` | 4097 |
| `expert3` | 3840, giữ nguyên |
| `expert4` | 1024, giữ nguyên |
| `top2_m2d` | 3073 |
| `top2_msclap` | 4097 |
| `top2_laion` | 2049 |
| `top2_mga` | 4097 |
| `m2d_lora` | 3073, giữ nguyên |

Tổng chiều mới: `top1_4 = 11010`, `full_9 = 27399`. Cache identity của sáu expert đổi theo representation policy `paired_cross_4d_plus_1_v2` và helper mới; shard cũ của sáu expert không được dùng lại. Cache Expert 3, Expert 4 và M2D-LoRA có thể dùng lại nếu provenance riêng của chúng còn khớp. Assembly identity đổi vì feature dimensions đổi.

Feature store cũ có chiều cũ sẽ bị validator từ chối. Gate checkpoint cũ không tương thích với input width mới; cần huấn luyện lại gate. Không cần huấn luyện lại expert. Các báo cáo MOE-09.3/09.4 trước đó ghi chiều cũ là kết quả lịch sử và được ghi đè bởi trạng thái trong tài liệu này.

Không chạy test suite trong lượt triển khai này.
