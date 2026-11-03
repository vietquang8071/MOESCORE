# Kế hoạch LOCAL-10.1

## Mục tiêu

Xác minh đầu ra thật của bốn expert đã khôi phục checkpoint, ghép đúng theo `sample_id`, rồi chạy qua assembler và gating core LOCAL-10 mà không nạp đồng thời cả bốn mô hình.

## Các bước

1. Bổ sung kiểm thử trước cho API nhận đầu ra expert đã xác minh; chỉ sau khi test đỏ mới thêm API mỏng vào `MoEForwardIntegration`.
2. Viết harness nhẹ để chọn manifest train ổn định, kiểm tra provenance, lưu/mở NPZ an toàn, báo cáo lỗi theo từng expert và chạy subprocess một expert mỗi lần.
3. Chạy real inference B=1 trước, sau đó lấy tập nhỏ chung; đo RSS/VRAM, kiểm tra checkpoint, output, frozen/eval, nguồn module và tính ổn định.
4. Ghép đủ đầu ra thật bằng assembler hiện có; kiểm tra chính xác các lát feature, chạy gate thật và backward kiểm tra gradient trên tensor detached.
5. Thử cặp E2/E4 khi preflight tài nguyên đạt ngưỡng an toàn; không dựng đồng thời bốn expert.
6. Chạy test tập trung, full regression, py_compile; viết báo cáo và JSON bằng tiếng Việt.

## Quy tắc xác minh

- Manifest dùng cùng tập mẫu và hash cho mọi tiến trình.
- Chỉ dữ liệu train có nhãn được dùng; nhãn không đi vào gate.
- Artifact chỉ lưu score/feature float32 và JSON metadata, không pickle.
- Kết quả sequential không được gọi là runtime đồng thời bốn expert.
- Không sửa expert, checkpoint, dữ liệu, score hay cấu hình huấn luyện.
