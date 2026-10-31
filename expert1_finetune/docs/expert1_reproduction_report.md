# MOESCORE Expert 1: báo cáo kiến trúc và cấu hình fine-tuning

## Phạm vi

Báo cáo này tổng hợp trainer riêng trong [`Train_Expert1.py`](Train_Expert1.py), notebook Colab [`Colab_Expert1.ipynb`](Colab_Expert1.ipynb), loss trong [`loss/loss_function.py`](loss/loss_function.py), cấu hình [`config.yaml`](config.yaml), log lần chạy batch 8 ở `../expert_1_training_result.txt`, và Expert 1 tích hợp trong [`model/Basic_comp.py`](model/Basic_comp.py).

Các thông số được chia thành **giá trị mặc định**, **cấu hình trong notebook**, và **giá trị của lần chạy batch 8 đã ghi log**. Chúng không luôn giống nhau: notebook hiện đặt batch size 2, còn lệnh chạy thực tế được cung cấp trong hội thoại dùng batch size 8.

## Kiến trúc LAION-CLAP được fine-tune

Trainer khởi tạo `laion_clap.CLAP_Module(enable_fusion=False, ...)` và nạp checkpoint do `--checkpoint` chỉ định. Với cấu hình mặc định của LAION-CLAP, mô hình gồm:

- **Audio encoder:** HTSAT-tiny, nhận waveform mono 48 kHz. HTSAT tạo biểu diễn audio; projection head đưa biểu diễn sang không gian chung 512 chiều.
- **Text encoder:** RoBERTa-base, với tokenizer CLAP giới hạn đầu vào 77 token. Text projection cũng đưa biểu diễn sang không gian chung 512 chiều.
- **Đầu ra hai nhánh:** embedding audio và text được chuẩn hóa L2 trong API LAION-CLAP. Vì vậy tích vô hướng của hai embedding tương đương cosine similarity.
- **Không dùng fusion branch:** `enable_fusion=False`.
- Mô hình CLAP có các logit scale cho contrastive pretraining, nhưng Expert 1 tính score trực tiếp từ hai embedding chuẩn hóa; không nhân thêm logit scale.

Cấu hình HTSAT-tiny của LAION-CLAP khai báo audio backbone embedding 768 chiều; projection của CLAP chuyển embedding này về joint embedding 512 chiều. Text backbone RoBERTa-base có hidden width 768 và cũng được chiếu về 512 chiều. Xem [cấu hình HTSAT-tiny chính thức](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model_configs/HTSAT-tiny.json) và [implementation CLAP chính thức](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model.py).

Forward của trainer lấy hai embedding, tính:

```text
score = sum(audio_embedding * text_embedding, dim=-1)
```

Kết quả là một score cho mỗi cặp audio-text, shape `(batch,)`. Loss nhận score trên thang gần `[-1, 1]`; phép đổi về thang 0–10 chỉ được dùng khi tính SRCC validation và lưu báo cáo metric.

### Phần nào được mở để cập nhật gradient

Trong `CLAPAudioTextEncoder.__init__`, trainer trước hết đóng gradient cho toàn bộ tham số, sau đó mở lại các tham số khớp những quy tắc tên sau:

1. Trong toàn bộ LAION-CLAP model, tham số có tên chứa `projection` hoặc `mlp`.
2. Trong `audio_branch`, tham số có tên chứa `layers.2`, `layers.3`, `head` hoặc `norm`.

Quy tắc thứ nhất có thể mở các MLP trong nhiều tầng HTSAT, không chỉ hai tầng cuối. Quy tắc `norm` cũng mở mọi tham số audio branch có tên chứa chuỗi này. Phần transformer chính của RoBERTa không được mở theo các điều kiện đó; projection của text được mở vì tên tham số chứa `projection`. Lần chạy được ghi log báo **28,378,768 tham số trainable**.

Việc này là fine-tuning một phần của LAION-CLAP: không đóng băng toàn bộ encoder, cũng không cập nhật mọi tham số của hai backbone.

## So sánh với `Expert1` trong `model/Basic_comp.py`

| Thành phần | Trainer `Train_Expert1.py` | `model/Basic_comp.py` – `Expert1` | Tác động |
|---|---|---|---|
| Module CLAP | `CLAPAudioTextEncoder` chứa `laion_clap.CLAP_Module` | `CLAP_audio_text_encoder` chứa cùng `CLAP_Module` | Cùng loại backbone LAION-CLAP. |
| Fusion | Hardcode `enable_fusion=False` | Đọc `cfg.clap_encoder.enable_fusion` | Muốn cùng kiến trúc thì config tích hợp phải đặt `false`. |
| Checkpoint | Đọc từ CLI `--checkpoint` | Đọc `cfg.clap_encoder.ckpt_path` | Khác nguồn cấu hình, không nhất thiết khác trọng số. |
| Freeze mask | Freeze tất cả, mở `projection`/`mlp`; mở audio `layers.2`, `layers.3`, `head`, `norm` | Dùng cùng các điều kiện tên | Chính sách chọn tham số trainable tương ứng nhau. |
| Điểm dự đoán | Tích vô hướng audio-text, shape `(B,)` | Tích vô hướng audio-text, `keepdim=True`, shape `(B, 1)` | Giá trị score như nhau; chỉ khác shape. |
| Fused features | Không tạo; trainer chỉ cần score cho loss | Nối hai embedding thành `fused_feats` shape `(B, 1024)` để gating network nhận | Phần nối này phục vụ MoE, không phải một encoder/nhánh CLAP mới. |
| Đầu ra forward | Chỉ trả score | Trả `(mos_hat, fused_feats)` | Khác giao diện phục vụ mục đích train riêng so với tích hợp MoE. |
| Cấu trúc tên module | `Expert1.model.model...` | `Expert1.model.model...` | Trainer giữ cùng nesting để state dict có thể nạp vào Expert 1 tích hợp, nếu key và shape tương ứng. |

Tóm lại, hai lớp Expert 1 dùng cùng cách lấy embedding CLAP, cùng phép tính cosine score và cùng quy tắc chọn tham số trainable. Khác biệt chính là trainer standalone không tạo feature nối cho gating network và nhận checkpoint/config qua CLI. Trong `Basic_comp.py`, `enable_fusion` phải có trong config; file `config.yaml` hiện trong repo không khai báo khóa đó, nên cần cung cấp khóa khi khởi tạo mô hình tích hợp.

## Tiền xử lý dữ liệu của trainer

`Expert1Dataset` trong `Train_Expert1.py` thực hiện các bước sau:

1. Đọc WAV, trộn các channel thành mono.
2. Resample về 48,000 Hz nếu sample rate ban đầu khác 48 kHz.
3. Chuẩn hóa độ dài về 10 giây: cắt ngẫu nhiên đoạn 10 giây khi train; lấy đoạn giữa khi validation; audio ngắn được lặp rồi đệm phần còn thiếu.
4. Chuẩn hóa nhãn bằng `(average_score - 5) / 5`, đưa score 0–10 về `[-1, 1]`.
5. Dùng `collate_batch` để ghép waveform và score thành batch; caption được giữ thành danh sách string để LAION-CLAP tokenize.

Đây là preprocessing trong trainer standalone. Nó không gọi `data/dataset.py` của pipeline MoE.

## Cấu hình fine-tuning

### Giá trị mặc định trong `Train_Expert1.py`

| Tham số | Mặc định |
|---|---:|
| Checkpoint | Đường dẫn CLI; notebook mặc định `weights/laion-clap.pt` |
| Số epoch tối đa | 50 |
| Batch size | 4 |
| Learning rate đỉnh | `1e-5` |
| Optimizer | AdamW, `weight_decay=1e-4` |
| Warm-up | 10% tổng số optimizer steps (`--warmup-ratio 0.1`) |
| Early stopping patience | 10 epoch không tăng SRCC |
| DataLoader workers | 0 |
| Seed | 42 |
| Device | CUDA |
| Log interval | 100 batch |

Scheduler là linear warm-up rồi linear decay, được tiến một lần sau mỗi `optimizer.step()`. Tổng step được tính theo `len(train_loader) × epochs` dự kiến.

### Loss trong config

Trainer đọc `config["loss"]` để tạo loss; các tham số MoE khác như top-level `batch_size: 16` không điều khiển run này vì batch và epoch được truyền qua CLI. Loss trong config hiện tại là:

| Thành phần | Cấu hình |
|---|---:|
| Hệ số clipped MSE | `1.0` |
| Hệ số contrastive | `0.5` |
| Clipped MSE `tau` | `0.25` trên score đã chuẩn hóa |
| Clipped MSE `mode` | `utt` |
| Contrastive margin | `0.1` |
| Class-balanced loss | Tắt (`cbl: false`) cho cả hai loss |

Các trường `beta: 0.99` không làm class balancing hoạt động khi `cbl` là `false`. Trainer dừng với lỗi nếu bật `cbl`, vì script yêu cầu xác minh `num_class` trước khi dùng.

### Cấu hình lần chạy batch 8 có log

Lệnh shell được dùng trong lần chạy này truyền batch size 8, epochs 50, patience 10 và warm-up ratio 0.1. Log ghi nhận:

- 7,500 mẫu train, 938 batch mỗi epoch; 3,000 mẫu validation, 375 batch mỗi epoch.
- 46,900 optimizer steps dự kiến; 4,690 bước warm-up và 42,210 bước decay.
- Optimizer AdamW, peak LR `1e-5`, weight decay `1e-4`.
- 28,378,768 tham số trainable.
- Best validation SRCC `0.60666` tại epoch 3; early stopping sau epoch 13.

Trong hội thoại, người dùng xác nhận checkpoint gốc là `630k-audio-best.pt`; notebook dùng tên bản sao cục bộ `weights/laion-clap.pt`. Log xác nhận checkpoint bản sao được nạp, nhưng không lưu hash để đối chiếu byte-for-byte.

Notebook hiện lưu `BATCH_SIZE = 2`, trong khi run trên dùng batch 8 theo lệnh shell. Nếu chạy lại nguyên notebook như đang lưu, giá trị batch sẽ là 2 trừ khi sửa cell hoặc override biến đó.

## Môi trường Colab ghi trong notebook

- Python 3.12.
- `numpy==1.26.4`.
- `laion-clap==1.1.7`.
- `transformers>=4.51.3,<5`.
- `pyyaml`; PyTorch và torchaudio lấy từ runtime Colab và được notebook in phiên bản khi chạy.

## Giới hạn tái lập

- Paper nêu batch size 8, Adam với LR `1e-5`, loss kết hợp clipped MSE và contrastive, cùng linear warm-up/decay. Paper không nêu warm-up ratio; 0.1 là giá trị triển khai trong trainer/notebook.
- Trainer hiện dùng AdamW với weight decay `1e-4`, nên khác mô tả Adam trong paper.
- Log phản ánh một lần chạy với seed 42 và batch 8; SRCC không đảm bảo lặp lại chính xác nếu runtime, thư viện, dữ liệu hoặc checkpoint khác.
- `weights/laion-clap.pt` là tên file cục bộ dùng trong notebook. Danh tính của checkpoint cần được hiểu theo nguồn/weights thực tế; tên file không tự xác minh hash.

## Tệp tham chiếu

- [`Train_Expert1.py`](Train_Expert1.py): dataset, CLAP wrapper, freeze policy, optimizer, scheduler, validation và checkpoint.
- [`Colab_Expert1.ipynb`](Colab_Expert1.ipynb): cài môi trường và các biến chạy trên Colab.
- [`model/Basic_comp.py`](model/Basic_comp.py): Expert 1 trong mô hình MoE.
- [`loss/loss_function.py`](loss/loss_function.py) và [`config.yaml`](config.yaml): loss và tham số loss.
- [`MOESCORE.pdf`](MOESCORE.pdf), [`Sun_NPU_Sun_NPU.pdf`](Sun_NPU_Sun_NPU.pdf): mô tả phương pháp và thiết lập paper.
- [LAION-CLAP model implementation](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model.py), [HTSAT-tiny model config](https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model_configs/HTSAT-tiny.json): cấu hình backbone, projection và embedding của thư viện upstream.
