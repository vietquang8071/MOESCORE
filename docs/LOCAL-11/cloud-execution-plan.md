# LOCAL-11 — Kế hoạch thực thi trên Google Colab và Kaggle

> Cập nhật sau LOCAL-11: quy trình Kaggle hiện hành nằm ở
> [hướng dẫn Kaggle bằng tiếng Việt](../KAGGLE_GATING_GUIDE_VI.md).
> Đã bổ sung extractor toàn tập theo shard, chốt loss `[1.0, 0.5]`, và xác minh
> nguồn UniLM đi kèm chỉ khác hai import. Các mô tả thiếu extractor hoặc bắt
> buộc clone UniLM riêng bên dưới phản ánh trạng thái lịch sử của LOCAL-11.

## 1. Phạm vi

LOCAL-11 chỉ chạy thử cổng trên bốn đầu ra expert đã xác minh và chạy test phần mềm cục bộ. Không có phiên Colab/Kaggle nào được khởi chạy. Các lệnh dưới đây là giao diện dự kiến cho CLOUD-01 đến CLOUD-04; cần sao chép cấu hình và thay đường dẫn mount theo phiên thực tế.

Pipeline gate đọc feature store theo ba nhánh độc lập: `train`, `validation`, `test`. Lệnh train chỉ mở `train` và `validation`; lệnh đánh giá chỉ mở `validation`; lệnh suy luận chỉ mở `test` và không đòi nhãn.

## 2. Phiên bản và kiểm tra môi trường

Môi trường LOCAL đã kiểm tra: Python `3.10.21`, PyTorch `2.14.1+cu130`, CUDA runtime `13.0`, NumPy `1.26.4`, SciPy `1.15.3`, SoundFile `0.14.0`, PyYAML `6.0.3`. Mã chuyên gia còn dùng các phụ thuộc gốc của bốn nhánh. Không giả định image Colab/Kaggle có cùng bản driver hoặc PyTorch; trước khi trích xuất thật, ghi lại phiên bản và chạy CLOUD-01. Không cài gói trong LOCAL-11.

Yêu cầu trước mỗi lượt cloud:

1. Xác nhận Python/PyTorch/CUDA có thể chạy các adapter đã xác minh; không nâng cấp thư viện tùy tiện.
2. Kiểm tra GPU thật bằng `nvidia-smi` và `torch.cuda.is_available()`; chọn batch nhỏ cho lần đầu.
3. Kiểm tra checkpoint, dữ liệu, UniLM source clone và feature store đã mount.
4. Chạy xác minh định danh UniLM E4 đúng commit `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` và fingerprint đã được LOCAL-07.1-FIX xác nhận.
5. Bật offline mode cho các thư viện Hub nếu dùng cache; không để tiến trình tải thay thế checkpoint hoặc tokenizer ngoài dự kiến.

## 3. Google Colab

### Mount và thư mục

- Gắn Google Drive chứa checkpoint đã được phê duyệt, metadata/WAV và thư mục kết quả bền vững.
- Đặt clone MOESCORE trong `/content/` để mã nguồn chạy trên đĩa phiên; đặt output/checkpoint định kỳ trong Drive. I/O Drive có thể chậm, nên không ghi từng batch lên Drive.
- Bản sao config nên khai báo `data.feature_store_root` và `output_dir` tương đối với tệp YAML hoặc dùng đường dẫn mount cụ thể trong config ngoài Git.
- Tạo thư mục UniLM từ đúng source clone đã xác minh; không dùng `expert4_finetune/unilm/` thay thế.

### Lệnh thao tác

Trong các lệnh sau, `<STORE>` là feature store có đủ split và provenance hợp lệ; `<ARTIFACTS>` là output numeric có báo cáo đã xác minh cho đúng split. LOCAL-10.1 hiện chỉ tạo artifact train giới hạn bốn mẫu, vì vậy không dùng chúng làm feature store đầy đủ.

```bash
python scripts/validate_moe_feature_store.py \
  --root <STORE> --split train --require-labels

python scripts/validate_moe_feature_store.py \
  --root <STORE> --split validation --require-labels

python scripts/validate_moe_feature_store.py \
  --root <STORE> --split test

python scripts/train_moe_gating.py \
  --config configs/moe_cloud.example.yaml

python scripts/train_moe_gating.py \
  --config configs/moe_cloud.example.yaml \
  --resume <OUTPUT_DIR>/latest_gate.pt

python scripts/evaluate_moe_gating.py \
  --config configs/moe_cloud.example.yaml \
  --checkpoint <OUTPUT_DIR>/best_gate.pt

python scripts/infer_moe_gating.py \
  --config configs/moe_cloud.example.yaml \
  --checkpoint <OUTPUT_DIR>/best_gate.pt \
  --output <OUTPUT_DIR>/submission.csv
```

Các lệnh train/evaluate/infer không dựng expert. Trước chúng, feature store phải được trích xuất bằng adapter thật với đúng checkpoint, split, preprocessing và provenance. Lệnh `scripts/build_moe_feature_store.py` hiện nhập artifact theo định dạng LOCAL-10.1 và dùng được để kiểm tra/nhập artifact sẵn có; nó không thay thế bộ trích xuất chunk tổng quát cho 7.500/3.000/3.000 mẫu. Tạo hoặc xác minh bộ trích xuất đó là tiền đề riêng của CLOUD-02.

### Lưu và khôi phục

- Giữ `latest_gate.pt` để resume; sao lưu sau các bước checkpoint interval và cuối epoch.
- Giữ `best_gate.pt` riêng để đánh giá/suy luận.
- Sao lưu manifest store, shard, báo cáo provenance và cấu hình cùng checkpoint gate.
- Khi runtime bị ngắt, xác minh hash/store identity trước khi tiếp tục; chỉ dùng `--resume` với đúng config, checkpoint expert và train/validation stores đã ghi trong checkpoint.

## 4. Kaggle

### Mount và thư mục

- Thêm dataset XACLE và checkpoint được phê duyệt làm Kaggle Input; input thường chỉ đọc.
- Lưu feature store đang tạo, log và checkpoint trong `/kaggle/working/`.
- Lưu CSV dự đoán trong `/kaggle/working/`; tải về hoặc đóng gói thành Kaggle Dataset theo quy trình challenge sau khi xác minh.
- Tạo bản config riêng với đường dẫn `/kaggle/input/...` và `/kaggle/working/...`; không sửa cấu hình production bằng đường dẫn cứng.
- Mount/copy đúng UniLM source clone đã xác minh vào input; kiểm tra commit/fingerprint trước khi trích xuất E4.

### Lệnh thao tác

```bash
python scripts/validate_moe_feature_store.py \
  --root <STORE> --split train --require-labels

python scripts/train_moe_gating.py \
  --config <KAGGLE_CONFIG.yaml>

python scripts/train_moe_gating.py \
  --config <KAGGLE_CONFIG.yaml> \
  --resume /kaggle/working/moe_gating/latest_gate.pt

python scripts/evaluate_moe_gating.py \
  --config <KAGGLE_CONFIG.yaml> \
  --checkpoint /kaggle/working/moe_gating/best_gate.pt

python scripts/infer_moe_gating.py \
  --config <KAGGLE_CONFIG.yaml> \
  --checkpoint /kaggle/working/moe_gating/best_gate.pt \
  --output /kaggle/working/submission.csv
```

Giới hạn phiên Kaggle có thể làm mất nội dung `/kaggle/working/` sau khi kết thúc. Lưu định kỳ checkpoint gate và manifest; trước khi dừng phiên, tải/commit output theo cơ chế Kaggle cho phép. Không coi feature cache chưa hoàn tất là split hợp lệ.

## 5. Pipeline trích xuất và huấn luyện cloud

### CLOUD-01 — Smoke môi trường

1. Xác minh import, GPU, dataset/checkpoint paths và UniLM identity.
2. Chạy một mẫu qua từng adapter hiện có với FP32.
3. Xác minh IDs, checkpoint provenance và output contract.
4. Dùng không quá bốn đầu ra train đã xác minh để chạy một đến ba bước cổng.
5. Không dùng test/validation label vào cập nhật tham số.

### CLOUD-02 — Trích xuất expert thành store

1. Chạy expert theo thứ tự ổn định và giữ nguyên preprocessing LOCAL-09.1.
2. Trích xuất riêng từng split; lưu shard giới hạn, ID theo thứ tự, score/features FP32 và provenance từng expert.
3. Resume theo manifest/checksum; từ chối shard trùng hoặc sai nguồn.
4. Chỉ finalize split sau khi đủ số record dự kiến và validation manifest đạt.
5. Không dùng bốn artifact mẫu của LOCAL-10.1 như toàn bộ split.

**Khoảng trống hiện tại:** verifier LOCAL-10.1 công khai hỗ trợ split `train` với tối đa vài mẫu; builder LOCAL-11 có thể nhập định dạng artifact đó. Chưa có CLI extraction tổng quát đã kiểm thử cho toàn bộ train/validation/test. Vì vậy, CLOUD-02 chưa sẵn sàng chỉ dựa trên các lệnh LOCAL-11; cần triển khai/kiểm thử extractor shard riêng trước khi chạy lớn.

### CLOUD-03 — Huấn luyện và chọn checkpoint

1. Kiểm tra train/validation stores có cùng thứ tự expert, provenance checkpoint và preprocessing contract.
2. Chỉ train loader được đưa vào optimizer; validation chạy eval/no-grad, không gọi optimizer.
3. Lưu `latest_gate.pt`; chọn `best_gate.pt` theo SRCC validation và báo undefined SRCC đúng trạng thái.
4. Trước khi chạy, giải quyết hệ số Contrastive: mã gốc đang dùng `+0.5`, cấu hình mẫu LOCAL-11 dùng `-0.5` theo yêu cầu milestone.
5. Không hiệu chỉnh score hoặc dùng test để chọn checkpoint.

### CLOUD-04 — Suy luận test

1. Kiểm provenance giữa gate checkpoint và test feature store.
2. Không yêu cầu hoặc tiêu thụ nhãn test.
3. Giữ thứ tự ID/tên file, kiểm đầu ra hữu hạn và không trùng.
4. Đối chiếu header/độ chính xác CSV với quy định nộp chính thức 2027 trước khi phát hành.

## 6. Yêu cầu dùng chung và nguyên tắc cuộc thi

Mọi môi trường phải dùng cùng registry `expert1, expert2, expert3, expert4`, feature width `7936`, kiến trúc gate LOCAL-10, target transform `(raw-5)/5`, hệ số diversity `0.2`, công thức score gốc và định dạng checkpoint. Không dùng trọng số cố định hoặc averaging; quyết định cuối phải đi qua gate trainable, phụ thuộc mẫu.

Challenge Rule 7 được mô tả là cho phép MoE nội bộ nhưng cấm ensembling thông thường. Trích xuất frozen experts offline thành feature cache không tự nó chứng minh phê duyệt hoặc vi phạm. Nếu cách triển khai cache/gating chưa được ban tổ chức xác nhận, cần hỏi rõ trước khi nộp và ghi lại kiến trúc inference cuối.

## 7. Dung lượng và lưu trữ

Chỉ ma trận đặc trưng FP32 7936 chiều ước tính khoảng 227 MiB cho train 7.500 mẫu, 91 MiB cho validation 3.000 và 91 MiB cho test 3.000; score, nhãn và overhead shard làm tăng thêm. Store đọc mmap và xử lý theo batch/shard, không bắt buộc nạp mọi split cùng lúc. Dung lượng VRAM cho gate nhỏ hơn nhiều expert, nhưng extractor expert cần GPU/RAM tương ứng với từng checkpoint; kiểm tra headroom theo từng expert, không cộng peak độc lập thành ước lượng simultaneous.

## 8. Trạng thái xác minh

Các lệnh trainer/evaluator/inference đã được kiểm bằng feature store tổng hợp nhỏ; smoke thực tế dùng bốn đầu ra thật LOCAL-10.1 trên CPU. Không có runtime Colab/Kaggle, full extraction, full training, full validation, test inference hay kết quả cuộc thi nào được xác minh trong LOCAL-11.
