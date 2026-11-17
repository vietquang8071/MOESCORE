# Báo cáo bốn adapter pretrained Top 2

**Cập nhật:** 17/11/2026  
**Phạm vi:** bốn wrapper Top 2 và xác minh inference thật; không tích hợp gate chín expert, không thêm M2D-LoRA.

## Kết quả hiện tại

Cả bốn adapter đã nạp checkpoint pretrained và chạy inference trên cùng một cặp audio–caption từ phần train XACLE 2027. Lượt `--adapter all` mới nhất đạt **4/4 PASS**. MS-CLAP dùng checkpoint local `CLAP_weights_2023.pth`; thử tải theo Hub mặc định trước đó bị chặn bởi DNS.

Chi tiết kết quả, score/shape quan sát được, checkpoint, môi trường và lệnh chạy nằm trong [báo cáo inference thật](top2_real_inference_report.md).

| Adapter | Preprocessing audio | Dimension audio/text đã quan sát | Kết quả inference thật |
|---|---:|---:|---|
| `top2_m2d` | 16 kHz | 768 / 768 | PASS |
| `top2_msclap` | Đọc WAV gốc; package dùng 44,1 kHz, đoạn 7 giây | 1024 / 1024 | PASS; score thay đổi theo crop ngẫu nhiên 7 giây |
| `top2_laion` | 48 kHz | 512 / 512 | PASS |
| `top2_mga` | 32 kHz | 1024 / 1024 | PASS |

Bốn adapter trả cosine score float32 `[1,1]`, feature float32 lần lượt `[1,1536]`, `[1,2048]`, `[1,1024]`, `[1,2048]`; model ở eval và frozen. Score MS-CLAP có thể thay đổi giữa lần chạy vì package chọn crop audio 7 giây ngẫu nhiên từ WAV dài 9.844 giây. Với MGA, loader khớp 415/415 tensor của model và báo một key checkpoint dư.

## Thay đổi và giới hạn

- MS-CLAP factory truyền checkpoint local qua `model_fp`; thử đường tải Hub mặc định không thành công do DNS.
- M2D factory giữ nguyên tên thư mục cha logic khi checkpoint được đưa qua symlink để đáp ứng metadata convention của `PortableM2D`.
- LAION nhận checkpoint native `laion-clap.pt` qua `CLAP_Module.load_ckpt`, dùng một checkpoint chung cho hai embedding đã projection vào không gian CLAP 512 chiều. Không dùng cặp pooled hidden-state từ hai biến thể khác nhau.
- Thêm script chạy tuần tự một cặp dữ liệu và phân biệt `BLOCKED` với `FAIL`.
- Không sửa source `EnsembleSVR-XACLE`, các adapter Top 1, registry, gate, feature store hay trainer. Thay đổi tracked có sẵn tại `expert1_finetune/Train_Expert1.py` được giữ nguyên.

Đăng ký bốn key `top2_m2d`, `top2_msclap`, `top2_laion`, `top2_mga` vào hệ chín expert vẫn thuộc milestone sau. Báo cáo này không mở rộng registry hoặc gating network hiện tại.
