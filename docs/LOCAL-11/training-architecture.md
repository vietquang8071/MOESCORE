# LOCAL-11 — Kiến trúc huấn luyện cổng MoE

## 1. Toán học huấn luyện gốc

Mã hiện hành nằm trong `Train_MOE.py`, `loss/loss_function.py`, `data/dataset.py` và `configs/config.yaml`. `Train_MOE.train()` chỉ đưa tham số của `model.gating_network` vào AdamW; expert được giữ cố định. Mỗi bước tính tổng loss hồi quy với phạt đa dạng từ entropy của trọng số cổng.

```text
loss_tổng = loss_hồi_quy + 0.2 × loss_đa_dạng
loss_đa_dạng = mean(1 - entropy(weights) / log(4))
```

`ReduceLROnPlateau` chạy ở cuối epoch, với `mode=min`, `factor=0.5`, `patience=5`. Checkpoint tốt nhất được chọn theo SRCC trên validation. Mã gốc dùng AdamW, learning rate `cfg["lr"]` (cấu hình là `1e-5`) và weight decay `1e-4`; batch train là 16, validation là 1, và train loader bỏ batch cuối không đủ kích thước.

## 2. Kiểm toán cấu hình loss

Các tham số được yêu cầu cho cấu hình LOCAL-11 là:

| Thành phần | Giá trị |
|---|---:|
| Trọng số Clipped MSE | `1.0` |
| Trọng số Contrastive | `-0.5` theo yêu cầu milestone |
| Ngưỡng Clipped MSE `tau` | `0.25` |
| Chế độ Clipped MSE | `utt` |
| `beta` | `0.99` |
| Biên Contrastive | `0.1` |
| Class balancing (`cbl`) | `false` |
| Phạt đa dạng | `0.2` |

Có sai khác cần quyết định trước khi chạy huấn luyện đầy đủ: tệp đang có `configs/config.yaml` thực tế ghi `loss_weights: [1.0, 0.5]`, trong khi yêu cầu LOCAL-11 chỉ định `[1.0, -0.5]`. Không sửa tệp cấu hình gốc. Cấu hình mẫu `configs/moe_cloud.example.yaml` theo giá trị `-0.5` được nêu trong yêu cầu và ghi chú sai khác ngay trong YAML. Bộ huấn luyện giữ nguyên chính xác trọng số được truyền qua cấu hình, không tự đổi dấu. Cần chốt một cấu hình chuẩn trước CLOUD-03; không dùng smoke nhỏ để giải quyết khác biệt về mục tiêu.

`ClippedMSELoss` tính MSE theo từng mẫu nhưng chỉ giữ mẫu có sai số tuyệt đối lớn hơn `tau`. `ContrastiveLoss` tạo ma trận sai khác cặp mẫu, dùng biên `0.1`, lấy phần vượt biên và chia đôi trung bình. Công thức loss gốc không bị sửa. Kiểm thử xác nhận cả miền bị cắt (loss bằng 0) lẫn miền hoạt động.

## 3. Hạng của tensor loss và broadcast

Gating hiện trả dự đoán `[B,1]`, còn collate nhãn gốc trả `[B]`. Nếu truyền nguyên dự đoán `[B,1]` vào `ContrastiveLoss`, phép `unsqueeze` tạo tensor dự đoán `[B,B,1]`; phép trừ với ma trận nhãn `[B,B]` có thể broadcast thành `[B,B,B]`. Ranh giới huấn luyện vì vậy đổi riêng dự đoán thành `prediction.reshape(-1)` trước loss. Gating vẫn trả `[B,1]`; mục tiêu đã đổi có dạng `[B]`.

Kiểm thử B=1, B=2 và B=4 xác nhận loss là scalar và ma trận Contrastive B=2/B=4 khớp tham chiếu `[B,B]`. Không dùng phép `.mean()` tùy tiện để che lỗi hạng.

## 4. `num_class` và class balancing

Dataset cũ của MOESCORE đọc `num_class`; CSV average của XACLE 2027 không bảo đảm trường này. Với cấu hình hiện hành `cbl=false`, hai loss gốc không dùng `num_class`, do đó pipeline truyền `None`, không tự dựng số lớp.

Nếu bật class balancing mà batch không có `num_class` dương kiểu int64 cho từng mẫu, pipeline báo lỗi. Audit mã loss còn phát hiện nhánh `ClippedMSELoss(cbl=True)` mở rộng trọng số thành `[B,1]` rồi nhân với loss `[B]`, dẫn đến broadcast `[B,B]` trước khi rút gọn. Pipeline từ chối nhánh đó ngay cả khi có metadata để không chạy một phép tính khác ý định. Không sửa `loss/loss_function.py`; muốn bật CBL cần một quyết định riêng về sửa source và kiểm toán toán học mới.

## 5. Ranh giới nhãn

Unified Dataset và feature store giữ nhãn gốc `float32` trong `[0,10]`. Chỉ trainer đổi nhãn tại ranh giới loss:

```python
target_normalized = (label_raw - 5.0) / 5.0
prediction_raw = prediction_normalized * 5.0 + 5.0
```

Batch được gắn `labels_scale="raw_0_to_10"`; trainer/evaluator từ chối nhãn thiếu dấu hoặc có dấu thang khác. Nhãn không được đưa vào gating, cũng không được ghi đè trong store. Phép đổi ngược không clip dự đoán.

## 6. Store đầu ra expert đã xác minh

`data/moe_feature_store.py` ghi từng split ở thư mục riêng: `train/`, `validation/`, `test/`. Mỗi split có `manifest.json` và các shard nguyên số NPY; JSON chứa thứ tự expert, fingerprint manifest, định danh checkpoint, định danh preprocessing, sample IDs, tên WAV, kích thước, dtype và SHA-256 nội dung/tệp.

Contract mỗi shard là `scores: float32 [N,4]`, `features: float32 [N,7936]`; nhãn raw là bắt buộc cho train, có thể vắng mặt ở test. `num_class` chỉ lưu khi nguồn có đầy đủ giá trị. Tải mảng dùng `allow_pickle=False`, mmap và cache tối đa một số shard cấu hình được. Ghi shard nguyên tử, ghi lại shard giống hệt có tính lặp an toàn; shard khác nội dung, ID trùng, split không khớp, dtype sai, NaN/Inf, thứ tự expert sai hay hash sai đều bị từ chối.

`scripts/build_moe_feature_store.py` nhập định dạng đầu ra LOCAL-10.1, kiểm tra thứ tự ID của bốn expert, fingerprint manifest, SHA artifact, báo cáo khôi phục factory, trạng thái frozen/eval, metadata checkpoint trước/sau, rồi so WAV/caption/nhãn với Unified Dataset khi có `--dataset-root`. Metadata checkpoint mà importer đối chiếu là đường dẫn, kích thước và thời gian sửa đổi; nó **không** tạo bằng chứng SHA nội dung checkpoint. Các tệp đầu ra được giữ theo split; builder hiện không chạy trích xuất full split.

## 7. Tập dữ liệu, DataLoader và ranh giới split

`MoeTrainingDataset` chỉ đọc store split được chỉ định, không đọc WAV và không khởi tạo model. `collate_moe_features` giữ thứ tự hàng, trả `scores [B,4]`, `features [B,7936]`, ID, tên WAV và `labels_raw [B]` khi có. Tensors là CPU FP32; `labels_scale` nêu rõ thang gốc. B=1, B=2, B=4 đã được kiểm tra.

Store yêu cầu ID có tiền tố đúng split; collate từ chối batch trộn split; trainer chỉ nhận `train`; evaluator chỉ nhận `validation`; suy luận chỉ nhận `test`. Mã ID model, tên model, caption, WAV path và nhãn không đi vào gating. Train/validation dùng chung provenance expert và cùng contract preprocessing; checkpoint ràng buộc fingerprint của cả hai store. Test không cần nhãn.

## 8. Trainer và cô lập gradient

`training/moe_trainer.py` dùng đúng `GatingNetwork` LOCAL-10 và loss gốc. Optimizer chỉ chứa tham số gate, kiểm tra bằng danh tính đối tượng; score/features expert được detach trước forward. Chuyên gia không được tạo trong trainer. Gate chuyển sang train khi cập nhật; validation/inference dùng eval và no-grad. Dự đoán `[B,1]` chỉ được làm phẳng tại ranh giới loss.

Mỗi bước trả các thành phần Clipped MSE, Contrastive, loss hồi quy kết hợp, entropy diversity, tổng loss, chuẩn gradient và số tensor tham số được cập nhật. Tổng phải là scalar hữu hạn; gradient gate phải hữu hạn. Dữ liệu expert và nhãn không bị sửa đổi. Smoke trên 4 đầu ra thật thực hiện 2 bước với B=2, dưới giới hạn 3 bước; đây chỉ là kiểm tra thực thi.

## 9. Trình tối ưu và lịch học

Mặc định trong mã trainer: AdamW, learning rate `1e-5`, weight decay `1e-4`; có thể cấu hình. Lịch ReduceLROnPlateau giữ `mode=min`, hệ số giảm `0.5`, patience `5`; chỉ được gọi sau validation mỗi epoch. Mẫu Colab/Kaggle giữ batch train 16, 50 epoch tối đa, patience dừng sớm 50, validation batch 1. Các giá trị này là mặc định cấu hình, chưa được đánh giá chất lượng.

## 10. Checkpoint và resume

`training/checkpoint_manager.py` ghi checkpoint gate-only nguyên tử, gồm state dict gate, optimizer, scheduler, epoch, global step, cấu hình và fingerprint, fingerprint store/provenance, trạng thái sampler, seed, RNG CPU/CUDA, patience và SRCC tốt nhất. Không lưu mô hình expert. Đọc dùng `torch.load(weights_only=True)`, kiểm tra schema version, feature width 7936, số expert, fingerprint và state dict strict. Checkpoint gating cũ có input 7168 bị từ chối; không dùng `strict=False` hoặc cắt/đệm trọng số.

`StatefulRandomSampler` lưu hoán vị, epoch và cursor. Unit test so sánh ba bước liên tục với một bước, lưu/khôi phục rồi chạy hai bước còn lại; loss và tham số khớp chính xác trên fixture CPU có dropout. Checkpoint `latest_gate.pt` dùng để resume; `best_gate.pt` phục vụ chọn model. Checkpoint mới hoàn tất epoch mang sampler sang epoch tiếp theo; mã runner cũng xử lý trạng thái sampler đã cạn khi khôi phục.

## 11. Validation và suy luận

`MoeGateEvaluator` báo SRCC, MSE, MAE trên thang gốc 0–10; SRCC không xác định khi prediction/target hằng được ghi `null` cùng trạng thái, không giả thành 0. Loss validation, nếu yêu cầu, đổi target về thang chuẩn hóa và không thực hiện optimizer step. Dự đoán được đổi về thang gốc bằng công thức ngược, không clip.

Suy luận test cần store không nhãn và gate checkpoint phù hợp provenance expert; batch không được dùng nhãn. CSV hiện theo hợp đồng của `Inference_MOE.py`: `wav_file_name,pred_score`, một dòng mỗi tệp, giữ thứ tự và làm tròn hai chữ số. Định dạng nộp chính thức XACLE 2027 chưa được xác minh trong repository; cần đối chiếu hướng dẫn challenge trước CLOUD-04.

## 12. Khả năng biểu diễn theo khoảng điểm expert

Với softmax, dự đoán mỗi mẫu là tổ hợp lồi nên nằm trong `[min(score experts), max(score experts)]`. Smoke trên bốn mẫu train LOCAL-10.1 cho cả 4/4 nhãn chuẩn hóa nằm ngoài khoảng đó: 3 nhãn thấp hơn cận dưới, 1 nhãn cao hơn cận trên. Khoảng cách tuyệt đối tối thiểu trung bình là `0.3688909`, lớn nhất `0.7085800`; cận dưới MSE trung bình trên bốn mẫu là `0.1883419` ở thang chuẩn hóa.

Đây là quan sát trên bốn mẫu đã chọn, không phải thống kê toàn tập và không phải lỗi phần mềm. Nó cho thấy phép trộn hiện tại không thể biểu diễn chính xác các nhãn đó nếu chỉ dùng điểm expert hiện có. Không thêm calibration, bias, clip hay phép biến đổi score trong LOCAL-11. Cần xem xét kết quả trên tập train đầy đủ và quyết định kiến trúc riêng sau khi giải quyết hệ số loss.

## 13. Chính sách cache và crop Expert 1

Feature store chứa đầu ra expert đã trích xuất bằng một lượt preprocessing cố định. Chúng không tương đương với augmentation stochastic trong lúc huấn luyện expert. Riêng Expert 1 dùng crop ngẫu nhiên ở chế độ train và crop giữa ở eval; LOCAL-10.1 artifacts được tạo theo eval deterministic. Huấn luyện gate trên các đặc trưng đã cache vì vậy là một giao thức cố định, không mô phỏng mỗi epoch crop mới.

## 14. Tuân thủ challenge và quyết định còn mở

Gating dùng trọng số trainable, phụ thuộc feature từng mẫu và phép tổng có trọng số nội bộ; không dùng average cố định, không đưa `model_id`, `model_name` hay label vào gate. Train chỉ nhận split train; validation dùng để đo/chọn checkpoint; test là suy luận không nhãn. Tuy nhiên, do hệ thống trích xuất cache expert offline, cần xác nhận diễn giải Rule 7 của ban tổ chức nếu chưa có phê duyệt rõ ràng.

Trước huấn luyện đám mây đầy đủ cần chốt ít nhất: dấu trọng số Contrastive (`configs/config.yaml` hiện `+0.5`, yêu cầu milestone là `-0.5`), khả năng biểu diễn nhãn theo điểm expert, schema nộp chính thức và bộ trích xuất shard tổng quát cho validation/test. LOCAL-11 không chạy các thí nghiệm cloud này.
