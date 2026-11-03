# LOCAL-11 Pipeline Implementation Plan

> **For agentic workers:** Thực hiện tuần tự từng nhóm; mỗi nhóm viết test trước, xác nhận RED, triển khai tối thiểu rồi chạy GREEN. Không commit theo yêu cầu của LOCAL-11.

**Goal:** Xây dựng pipeline portable chỉ huấn luyện mạng cổng trên đầu ra expert đã xác minh, gồm feature store, loss/target boundary, checkpoint resume, validation và xuất dự đoán.

**Architecture:** Lưu dữ liệu từng split bằng các shard NPZ số học, đọc an toàn không pickle và ràng buộc bằng manifest JSON có fingerprint. Dùng lại `GatingNetwork`, `get_loss_function` và feature order hiện tại; đặt biến đổi nhãn, loss boundary, optimizer, checkpoint, metric và CSV ở các module riêng, không sửa mã expert hoặc loss gốc.

**Tech Stack:** Python 3.10, PyTorch, NumPy, PyYAML, unittest; không thêm dependency.

**Spec:** Yêu cầu LOCAL-11 trong chỉ dẫn người dùng ngày 2026-11-02.

## Ràng buộc chung

- Không tải expert/checkpoint, không chạy huấn luyện đầy đủ, validation đầy đủ hoặc test inference đầy đủ.
- Smoke tối đa 4 mẫu, 3 bước optimizer, CPU mặc định; chỉ dùng artifact LOCAL-10.1 sau khi xác minh manifest, danh tính, split, provenance và nhãn.
- Không sửa expert adapters, dataset gốc, loss gốc, mô hình MOE gốc, checkpoint, nhánh hay remote; không stage/commit/push.
- Giữ score/features FP32 và bất biến; labels thô 0–10 chỉ đổi ở ranh giới loss bằng `(raw-5)/5`.
- Cấu hình LOCAL-11 ghi rõ hệ số contrastive `-0.5` theo yêu cầu. Mã `configs/config.yaml` đang có `+0.5`; giữ nguyên tệp đó và báo sai khác để chủ dự án quyết định trước khi chạy cloud.

## Cấu trúc tệp

- `data/moe_feature_store.py`: schema, writer shard nguyên tử, validator, reader lười và chuyển artifact LOCAL-10.1.
- `data/moe_training_dataset.py`: Dataset/collate cho score `[B,4]`, feature `[B,7936]`, ID và nhãn thô nếu split cho phép.
- `training/target_transform.py`: đổi thang nhãn thuận/nghịch và chẩn đoán tính khả thi của score interval.
- `training/moe_trainer.py`: objective gốc, loss boundary `[B]`, diversity, AdamW, scheduler và một bước train/eval.
- `training/checkpoint_manager.py`: checkpoint chỉ chứa gate/optimizer/scheduler/trạng thái tái lập, lưu nguyên tử, nạp strict và kiểm provenance.
- `training/moe_evaluator.py`: metric, suy luận không nhãn, đổi score và CSV ổn định.
- `scripts/{validate_moe_feature_store,train_moe_gating,evaluate_moe_gating,infer_moe_gating,smoke_test_moe_training}.py`: CLI thực thi API dùng chung.
- `configs/moe_cloud.example.yaml`: đường dẫn và siêu tham số portable cho Colab/Kaggle.
- `tests/test_moe_{feature_store,training_dataset,training_pipeline,checkpoint_resume,validation_inference}.py`: test hợp đồng và luồng nhỏ.
- `docs/LOCAL-11/{training-architecture,cloud-execution-plan,implementation-report}.md`: báo cáo tiếng Việt.

## Các bước thực hiện

### Bước 1 — Chuyển đổi nhãn, loss và tính khả thi

1. Viết test cho normalize/denormalize, score interval, `num_class` khi CBL bật/tắt, loss `[B,1]` so với `[B]`, pairwise matrix B=1/2/4 và loss gốc theo weights cấu hình.
2. Chạy test để xác nhận RED vì các API chưa tồn tại.
3. Cài `target_transform.py` và adapter loss trong trainer, làm phẳng prediction đúng tại loss boundary; giữ công thức gốc, hệ số diversity 0.2 và không sửa `loss/loss_function.py`.
4. Chạy lại test và kiểm tra parity với tham chiếu độc lập.

### Bước 2 — Feature store và Dataset

1. Viết test RED cho shard NPZ, manifest hash, FP32, danh tính, split, provenance, labels bắt buộc cho train, test không nhãn, shard resume và lỗi dữ liệu hỏng.
2. Cài writer/reader an toàn và Dataset/collate; kiểm tra trùng ID xuyên shard, schema, thứ tự expert và fingerprint trước khi mở dữ liệu.
3. Thêm kiểm thử nạp bốn artifact LOCAL-10.1 nếu artifact còn hợp lệ, không dựng expert.

### Bước 3 — Trainer và checkpoint resume

1. Viết test RED cho một bước train chỉ cập nhật gate, finite loss/gradient, validation không cập nhật, scheduler theo epoch và resume xác định.
2. Cài trainer dùng AdamW/scheduler theo `Train_MOE.py`, sampler thứ tự xác định và trạng thái RNG; chỉ gate có optimizer.
3. Viết test RED cho checkpoint strict, gate-only, optimizer/scheduler/step/store identity/provenance; cài lưu nguyên tử và nạp an toàn.
4. Chạy thử tiếp tục 3 bước liền mạch so với 1 bước + lưu/nạp + 2 bước.

### Bước 4 — Đánh giá và suy luận

1. Viết test RED cho SRCC undefined, MSE/MAE, inverse target transform, suy luận không nhãn và CSV `wav_file_name,pred_score`.
2. Cài evaluator/inference dùng `GatingNetwork` thực, eval/no-grad, giữ IDs và thứ tự.
3. Chạy fixtures có kết quả đã biết, không chạy split challenge đầy đủ.

### Bước 5 — CLI, smoke, cấu hình và báo cáo

1. Viết test RED cho config validation/CLI, trạng thái artifact real-vs-synthetic, giới hạn smoke và JSON.
2. Cài CLI kiểm store/train/evaluate/infer/smoke; mọi entry point gọi cùng API và fail rõ khi thiếu store/provenance.
3. Kiểm tra artifact `/tmp/moescore_local101`; nếu hợp lệ lấy tối đa bốn mẫu train và chạy tối đa ba bước CPU. Nếu thiếu/hỏng thì smoke synthetic, đánh dấu real smoke chưa chạy.
4. Viết đủ ba báo cáo tiếng Việt, ghi discrepancy `+0.5`/`-0.5`, chính sách cache E1, Rule 7 và hướng dẫn Colab/Kaggle.
5. Chạy py_compile, test mới, các test LOCAL-10 liên quan, chẩn đoán JSON và toàn bộ suite.

## Tiêu chí nghiệm thu

- Test RED/GREEN cho mọi API mới; kết quả thực real artifact và synthetic được tách rõ.
- B=1/2/4 hợp lệ; loss scalar không broadcast sai; gate-only checkpoint strict round-trip/resume xác định.
- Validation không cập nhật; test inference không cần nhãn; CSV giữ đúng mapping.
- Không khẳng định đã train/validate/infer cloud hoặc chất lượng mô hình.

## Trọng tâm tự rà soát

- Manifest có thực sự ràng buộc shard nội dung và provenance, không chỉ tin filename/hash khai báo?
- CBL bật có thất bại rõ khi thiếu `num_class`, tránh tự tạo thống kê từ validation/test không?
- Khi resume, sampler và dropout RNG có tiếp tục cùng thứ tự/mask như luồng không ngắt?
- Test inference có loại nhãn khỏi đầu vào ngay cả khi store chứa nhãn?
- Các chỉ số SRCC constant-input trả undefined, không tự biến thành 0?
