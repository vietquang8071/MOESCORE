# Fine-tune Expert 2 MGA-CLAP

Trainer này dùng **cosine similarity trực tiếp**, giữ hai embedding MGA-CLAP đã L2-normalize, rồi tối ưu clipped MSE và pairwise score loss của MOESCORE. Mặc định mở hai block HTSAT cuối, hai layer BERT cuối, projection và codebook. `freeze_policy: full` là đối chứng mở mọi transformer block; frontend audio vẫn giữ thống kê BatchNorm cố định.

Dữ liệu mặc định là XACLE 2027 development dataset tại `../../data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset`. Train/validation dùng `train_average.csv` (7.500 cặp) và `validation_average.csv` (3.000 cặp); không dùng file đánh giá riêng từng listener. Đây là bộ dữ liệu 2027, vì vậy metric của paper XACLE 2026 chỉ là mốc tham khảo, không bảo đảm so sánh trực tiếp.

Chuẩn bị môi trường có PyTorch + CUDA tương thích GPU, sau đó cài dependencies trong `requirements-expert2.txt`. Source MGA được đặt ở `MGA-CLAP/`, checkpoint ở `weights/mga-clap.pt`. Các patch nhỏ trong source MGA: Expert 2 tạo BERT từ config vì checkpoint MGA chứa BERT weights, còn code pretrain vẫn dùng BERT pretrained theo mặc định; bỏ import chỉ cần cho pretrain, và dùng đúng device trong Sparsemax. Loader chỉ bỏ buffer `position_ids` cũ nếu nó là dãy vị trí chuẩn mà Transformers hiện tại không còn khai báo; mọi trọng số khác vẫn được nạp với `strict=True`.

Kiểm tra dữ liệu:

```bash
python Train_Expert2.py --check-data
```

Finetune:

```bash
python Train_Expert2.py --config configs/expert2_finetune.yaml
```

Đánh giá zero-shot MGA ban đầu (P0) hoặc nạp lại checkpoint đã finetune trên validation:

```bash
python Train_Expert2.py --eval-only
python Train_Expert2.py --eval-only --checkpoint weights/expert2_best.pt
```

Sau khi chọn `expert2_best.pt` bằng validation SRCC, đánh giá **một lần** trên `test_average.csv` và `wav/test/`:

```bash
python Train_Expert2.py --eval-only --split test --checkpoint weights/expert2_best.pt
```

Lệnh này tính SRCC/LCC/KTAU/MSE trên 3.000 cặp test có nhãn và ghi `weights/expert2_run/test_evaluation_metrics.json`, `weights/expert2_run/test_evaluation_predictions.csv`. Không dùng test SRCC để chọn lại checkpoint. Notebook Colab có cell tương ứng sau phần train.

Đổi dataset bằng `--data-root /path/to/XACLE_dataset`. Đổi policy, batch hoặc thiết bị trong YAML. Resume từ `weights/expert2_run/last.pt` bằng `--resume weights/expert2_run/last.pt`.

Output `weights/expert2_best.pt` là raw `Expert2.state_dict()` để nạp vào wrapper MoE cùng cấu trúc `Expert2.model.model=ASE`. Metrics và prediction validation tốt nhất nằm trong `weights/expert2_run/`; lệnh đánh giá ghi file `pretrained_*` hoặc `evaluation_*` cùng thư mục. Prediction có cột `wav_file_name,pred_score`. Trainer chỉ train Expert 2; script `Train_MOE.py` hiện tại là giai đoạn gate riêng và còn các lỗi batch key/config cho toàn MoE.

Tham số mặc định: Adam, batch 8, learning rate 1e-5, warm-up 10% và linear decay, tối đa 50 epochs, chọn best theo validation SRCC. Ngưỡng loss `tau=0.25`, `margin=0.1` nằm trên thang điểm chuẩn hóa [-1,1]; hệ số hai loss là 1.0 và 0.5. Đây là kết hợp thông số paper với các giá trị còn thiếu được ghi rõ trong [phương án](mga-clap_finetune.md).

Máy kiểm tra hiện có GTX 1650 4 GB. Forward một mẫu và backward batch 2 đã chạy; batch 8 và cả epoch chưa được kiểm chứng trên GPU này. Nếu giảm batch để vừa VRAM, loss pairwise thay đổi do số cặp trong batch khác, nên cần báo đây là thí nghiệm khác với thiết lập paper.

## Google Colab

Mở [Colab_Expert2.ipynb](Colab_Expert2.ipynb) trên Colab và chọn GPU runtime. Tải bản sao code hiện tại (gồm source `MGA-CLAP/` đã được patch và `weights/mga-clap.pt`) lên `MyDrive/MOESCORE/`. Đóng gói dataset từ máy local thành một ZIP và tải lên `MyDrive/XACLE_dataset.zip`; notebook tự giải nén vào `/content` để giảm số lần đọc WAV qua Drive. Nếu Drive đã có dataset dạng thư mục, notebook cũng nhận `MyDrive/XACLE_dataset/`, nhưng đọc trực tiếp sẽ chậm hơn.

Ví dụ tạo ZIP trên máy local, từ thư mục cha của dataset thực tế:

```bash
cd ~/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset
zip -r XACLE_dataset.zip XACLE_Challenge_2027_development_dataset
```

Notebook tạo config riêng cho Colab với đường dẫn tuyệt đối, kiểm tra dữ liệu, cài dependencies, train và hiển thị metric. Best checkpoint ở `MyDrive/MOESCORE/weights/expert2_best.pt`; checkpoint tiếp tục ở `MyDrive/MOESCORE/weights/expert2_run/last.pt`. Nếu phiên Colab ngắt sau một epoch, chạy lại notebook với `RESUME = True`. Chỉ có epoch đã hoàn tất được lưu. Drive cần đủ chỗ cho checkpoint gốc 1,6 GB và các checkpoint mới. GPU Colab và thời gian chạy thay đổi theo phiên; giữ batch size 8 nếu muốn đúng cấu hình đã chốt.
