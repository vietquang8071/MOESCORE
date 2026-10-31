# Báo cáo reproduction và finetune MOESCORE Expert 2 — MGA-CLAP

Ngày rà soát: 2026-10-31  
Phạm vi: implementation Expert 2 riêng, khởi tạo từ checkpoint MGA-CLAP và finetune trên XACLE average-score pairs. Không train MoE gate hay các expert khác.

## 1. Tóm tắt implementation

Implementation nằm trong [expert2_model.py](expert2_model.py), [expert2_data.py](expert2_data.py), [expert2_loss.py](expert2_loss.py), [Train_Expert2.py](Train_Expert2.py) và [configs/expert2_finetune.yaml](configs/expert2_finetune.yaml). Backbone được dựng từ source MGA-CLAP trong thư mục [MGA-CLAP](MGA-CLAP), theo kiến trúc trong [ASE](MGA-CLAP/models/ase_model.py) và cấu hình [inference_example.yaml](MGA-CLAP/settings/inference_example.yaml).

Model giữ đường biểu diễn MGA-CLAP gồm HTSAT, BERT, hai projection và Modality-Shared Codebook (MSC). Hai embedding đầu ra được L2-normalize, sau đó lấy dot product trực tiếp làm cosine score. Không có alignment MLP hay regression head mới. Đây là lựa chọn đã chốt để bám theo score computation của Expert 2 trong code MOESCORE.

Metric Expert 2 được công bố trong MOESCORE validation table là SRCC 0.6297, LCC 0.6562, KTAU 0.4550, MSE 3.7208. Đây là mốc đối chiếu, không phải kết quả đã đạt được của implementation này. Chưa có full finetuning run được xác nhận trong workspace.

## 2. Kiến trúc MGA-CLAP được finetune

### 2.1. Nhánh audio

- WAV được chuyển thành mono, resample về 32 kHz, rồi crop hoặc zero-pad về 10 giây (320.000 samples).
- Frontend trong cấu hình MGA dùng STFT với n_fft 1024, hop_length 320, 64 mel bins, f_min 50 Hz và f_max 14.000 Hz.
- Audio encoder là HTSAT-Swin Transformer: embed_dim 96, 4 stage có depth [2, 2, 6, 2] và số attention head [4, 8, 16, 32]. Runtime tạo biểu diễn frame với chiều 768.
- Frame features đi qua frame_proj: LayerNorm(768) → Linear(768, 1024) → GELU → LayerNorm(1024) → Linear(1024, 1024).
- Đường finetune dùng frame embeddings, không dùng global audio/classification output của HTSAT.

### 2.2. Nhánh text

- Caption được giữ dưới dạng chuỗi và chuyển cho tokenizer BERT-base-uncased trong MGA-CLAP.
- Tokenizer pad theo caption dài nhất trong batch, truncate ở tối đa 30 tokens và tạo special-token mask.
- BERT-base-uncased trả token features chiều 768. Code bỏ token đầu trước khi projection và dùng mask để MSC bỏ qua token đặc biệt/padding.
- Word features đi qua word_proj có cấu trúc tương tự frame_proj: LayerNorm(768) → Linear(768, 1024) → GELU → LayerNorm(1024) → Linear(1024, 1024).

### 2.3. Modality-Shared Codebook và score

ASE có codebook dùng chung cho hai modality với shape 4096 × 1024. MSC thực hiện dot-product attention trên frame/token features, scale theo căn bậc hai embedding dimension, chia temperature nội bộ T=1000, max-pool theo thời gian, áp Sparsemax trên 4096 code, rồi tổng hợp code vectors bằng trọng số Sparsemax. Embedding text có mask; audio embedding không cần mask chuỗi.

Sau MSC, cả hai vector được L2-normalize:

    z_audio = normalize(MSC(frame_features, codebook))
    z_text  = normalize(MSC(word_features, codebook, text_mask))
    score_norm = sum(z_audio * z_text)

Mỗi vector có shape [B, 1024]. Vì đã chuẩn hóa L2, dot product bằng cosine similarity. Model trả score shape [B, 1] và feature ghép [z_audio, z_text] shape [B, 2048] để giữ interface tương thích MoE. Temperature contrastive pretraining của ASE (temp=0.07) không nhân vào score finetune; MSC vẫn dùng T=1000 như định nghĩa upstream.

### 2.4. Khởi tạo và checkpoint

Trainer dựng ASE từ MGA-CLAP chính thức với audio pretrained flag tắt và BERT được khởi tạo từ config; sau đó nạp checkpoint local weights/mga-clap.pt từ checkpoint["model"]. Mọi weight được load strict. Loader chỉ bỏ buffer BERT position_ids cũ nếu tensor đúng là dãy vị trí chuẩn và runtime Transformers hiện tại không có buffer tương ứng; các missing/unexpected weights khác gây lỗi.

SpecAugment được tắt cho lượt finetune hiện tại. Checkpoint MGA local có cấu trúc khớp với HTSAT, BERT-base-uncased, frame/word projection 1024 chiều và codebook 4096 × 1024. File [mga-clap_finetune.md](mga-clap_finetune.md) ghi lại hash checkpoint và các vấn đề reproduction đã được phân tích.

## 3. So sánh với Expert 2 trong model/Basic_comp.py

Cả hai implementation đều là MGA-CLAP và cùng phép tính lõi: HTSAT tạo frame features, BERT tạo token features, projection đưa về không gian 1024 chiều, MSC dùng shared codebook để tạo embedding toàn cục, L2-normalize hai embedding, rồi tính dot product làm score. Cả hai trả score và phép nối audio/text features; không thêm MLP vào nhánh Expert 2.

| Thành phần | Expert 2 trong Basic_comp.py | Expert 2 finetune hiện tại |
|---|---|---|
| Mục đích | Expert runtime được MOE_model dùng khi train/evaluate gating network | Trainer độc lập để cập nhật MGA-CLAP theo average relevance score |
| ASE/import | Import qua tên mga_model.ASE trong code MOESCORE cũ | Import models.ase_model.ASE trực tiếp từ source MGA-CLAP được vendored |
| Nạp checkpoint | Nạp checkpoint["model"] với strict=False; sau đó gọi self.model.eval() | Load strict=True, chỉ tương thích riêng buffer position_ids chuẩn đã bị Transformers bỏ |
| Input audio | Expert2.forward đọc batch["wavs"]; encoder lấy trung bình channel bằng audio.mean(1) | Dataset tạo mono 32 kHz/10 giây; collate dùng batch["mga_wavs"] dạng [B,1,320000], encoder squeeze channel |
| Input text | batch["caption_tokens"], chuỗi được tokenizer bên trong MGA-CLAP | Cùng chuỗi raw và tokenizer upstream; dùng thống nhất batch key caption_tokens |
| Embeddings và MSC | encode_text → word_proj → MSC; encode_audio → frame_proj → MSC; L2-normalize | Cùng luồng feature và MSC |
| Score/feature | Dot product cosine shape [B,1]; concat audio/text feature shape [B,2048] | Giữ nguyên score và shape feature để tương thích MoE |
| Trainability | Wrapper trong Basic_comp.py gọi eval(), nhưng không tự đặt requires_grad mask; MOE_model đóng băng experts ở giai đoạn train gate | Freeze mặc định partial được định nghĩa tường minh theo tên module; chỉ các module được chọn tham gia optimizer |

Vì vậy, khác biệt chính không phải là một kiến trúc scoring khác: finetune giữ nguyên computational path của Expert 2 trong Basic_comp.py. Các thay đổi nằm ở import/source rõ ràng, kiểm tra checkpoint nghiêm ngặt, preprocessing riêng đúng yêu cầu 32 kHz của MGA, batch schema thống nhất, cùng freeze policy và trainer dùng loss cho score prediction. Sau khi finetune, state_dict có thể được nạp vào wrapper có cùng cấu trúc Expert2.model.model=ASE.

## 4. Freeze policy trong cấu hình hiện tại

Cấu hình đang chọn freeze_policy=partial. Model đóng băng toàn bộ parameter trước rồi chỉ mở:

| Module | Trạng thái |
|---|---|
| Hai HTSAT block cuối: audio_encoder.audio_enc.layers.3.blocks.0 và blocks.1 | Train |
| LayerNorm cuối của HTSAT: audio_encoder.audio_enc.norm | Train |
| Hai BERT layer cuối: text_encoder.text_encoder.encoder.layer.10 và layer.11 | Train |
| frame_proj và word_proj | Train |
| Shared codebook | Train |
| Các HTSAT block khác, các BERT layer 0–9, embedding/frontend không nằm trong danh sách trên | Freeze |
| temp và các head không tham gia đường cosine | Freeze/không dùng trong score |

Khi gọi model.train(), wrapper để các encoder eval trước rồi bật train mode cho đúng hai HTSAT blocks, norm và hai BERT layers cuối; các projection/codebook ở train mode. Điều này giữ dropout/stochastic behavior của phần backbone đóng băng ở eval trong khi cho phép gradient cập nhật các module partial. Tùy chọn freeze_policy=full có trong code nhưng không phải cấu hình mặc định; nó mở toàn bộ HTSAT transformer layers, norm, BERT, projection và codebook.

Cách ánh xạ “hai lớp cuối” thành hai block HTSAT cuối ở stage 3 là lựa chọn code-level cụ thể. Paper không mô tả chính xác tên stage/block tương ứng với HTSAT của checkpoint này.

## 5. Dataset và tiền xử lý

- Root mặc định trong YAML: ../../data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset (tương đối với repo root).
- Train: meta_data/train_average.csv và wav/train; cấu hình kỳ vọng 7.500 cặp.
- Validation: meta_data/validation_average.csv và wav/validation; cấu hình kỳ vọng 3.000 cặp.
- Test có 3.000 cặp và chỉ được đọc khi gọi eval-only --split test, không tham gia tối ưu hoặc chọn checkpoint.
- Hỗ trợ tên cột wav_file_name hoặc wave_file_name, cùng text và average_score hoặc average_semantic_alignement_score.
- Kiểm tra score hữu hạn trong [0,10], caption/file name hợp lệ, file WAV hiện hữu, WAV không rỗng và không trùng filename trong split. Trainer kiểm tra giao nhau giữa train và validation.
- WAV integer được scale thành float32; stereo được downmix bằng trung bình kênh; resample_poly chuyển mọi sample rate về 32 kHz; âm thanh crop/pad thành 10 giây.
- Score target được chuẩn hóa y=(score_raw−5)/5 về [-1,1]. Khi report prediction, chuyển ngược score_raw=5·score_norm+5.

Các count trong YAML là kiểm tra kỳ vọng theo dataset hiện tại. MOESCORE paper là benchmark XACLE 2026; dữ liệu local có tên XACLE Challenge 2027 development. Vì split/version không mặc nhiên đồng nhất, kết quả trên data 2027 phải được ghi rõ và không xem là reproduction ngang bằng hoàn toàn với benchmark 2026.

## 6. Cấu hình finetune đã dùng

| Tham số | Giá trị hiện tại | Nguồn/ghi chú |
|---|---:|---|
| Freeze policy | partial | Lựa chọn implementation; layer cuối được ánh xạ tường minh trong expert2_model.py |
| Device | cuda | YAML mặc định; cần CUDA khả dụng khi train |
| Seed | 42 | YAML; seed Python, NumPy, PyTorch và CUDA được đặt |
| Train batch size | 8 | Một forward/loss trên tối đa 8 cặp; không gradient accumulation |
| Validation batch size | 8 | Không shuffle |
| DataLoader workers | 0 | YAML |
| Epoch tối đa | 50 | YAML; kết thúc sớm nếu đủ patience |
| Early stopping patience | 10 epochs | Dựa trên validation SRCC không cải thiện |
| Optimizer | Adam | betas=(0.9, 0.999), weight_decay mặc định của Adam bằng 0 |
| Peak learning rate | 1e-5 | YAML và TASALIGN report |
| Scheduler | Linear warm-up rồi linear decay | Warm-up ratio=0.1 của tổng optimizer steps dự kiến |
| Gradient clipping | Global norm 2.0 | YAML, sau backward trước optimizer step |
| Mixed precision | Không bật trong trainer hiện tại | Forward/backward chạy precision mặc định của PyTorch |
| SpecAugment | Tắt | Được đặt trong config ASE trước khi model dựng |
| Loss CBL | Tắt/không dùng | expert2_loss.py triển khai clipped MSE và pairwise objective, không class-balanced weighting |
| Chọn checkpoint | Validation SRCC cao nhất | Test split không dùng để chọn model |

Dataset có 7.500 train pairs; batch size 8 tạo 938 train batches/epoch vì DataLoader giữ batch cuối không đầy (batch cuối 4 cặp). Không đặt drop_last=True. Không dùng gradient accumulation; vì thế các quan hệ pairwise loss được tính trong phạm vi batch thực tế của từng optimizer step.

### Loss

Prediction và target được flatten/assert thành vector [B], score chuẩn hóa [-1,1]. Clipped MSE bỏ qua sai số không vượt tau=0.25 và lấy bình phương sai số còn lại:

    error_i = pred_i − target_i
    L_mse = mean(1[|error_i| > 0.25] · error_i²)

Pairwise term so sánh mọi hiệu score có hướng trong batch, áp margin 0.1, lấy mean trên B×B rồi chia 2:

    L_pair = mean(max(0, |(pred_i − pred_j) − (target_i − target_j)| − 0.1)) / 2
    L_total = 1.0 · L_mse + 0.5 · L_pair

Đây là đúng công thức trong [expert2_loss.py](expert2_loss.py), với ngưỡng mặc định tau=0.25 và margin=0.1. Trên score scale 0–10, tau tương đương 1.25 điểm và margin tương đương chênh lệch 0.5 điểm.

### Metrics và lưu trạng thái

Sau mỗi epoch, validation được tính trên thang điểm 0–10: SRCC, Pearson LCC, Kendall tau-b (ghi là KTAU) và MSE.

- Checkpoint tốt nhất được ghi ở weights/expert2_best.pt khi SRCC tăng; đây là raw Expert2 state_dict.
- weights/expert2_run/validation_predictions.csv và best_metrics.json được cập nhật cùng checkpoint tốt nhất.
- weights/expert2_run/last.pt được ghi sau mỗi epoch, chứa model, optimizer, scheduler, metric, best SRCC, early-stop state và RNG state để resume.
- Log train/validation hiện được in ra stdout; trainer chưa ghi file training_log riêng.
- Chạy eval-only sẽ ghi prediction CSV và metric JSON trong expert2_run, với prefix pretrained/evaluation hoặc test_pretrained/test_evaluation.

Trainer hiện không dùng AMP và không có gradient scaler. Checkpoint resume cũng không chứa scaler state vì không có mixed precision.

## 7. Căn cứ paper và giới hạn reproduce

1. MOESCORE paper mô tả Experts 1–3 dự đoán bằng similarity; Expert 2 code trong Basic_comp.py thực hiện dot product trên embedding đã normalize. Vì vậy bản finetune dùng cosine trực tiếp.
2. Figure 1(c) của MOESCORE có thể được diễn giải là có thêm alignment features/head cho expert; nhánh Expert 2 trong code được cung cấp không hiện thực head đó. Implementation này ưu tiên code score path và quyết định cosine đã chốt, không tự thêm MLP.
3. MOESCORE config/script cho giai đoạn MoE có optimizer/scheduler và batch khác với fine-tuning một expert độc lập; Train_MOE.py là train gating network sau khi experts đã được tạo/freeze, không được dùng làm bằng chứng cho optimizer của Expert 2.
4. Adam, batch size 8, LR 1e-5 và linear warm-up/decay được lấy theo TASALIGN report. Warm-up ratio 0.1, epochs 50, patience 10, seed 42, validation batch 8 và grad clip 2.0 là các giá trị của cấu hình implementation/giả định vì recipe Expert 2 riêng không được công bố đủ.
5. Freeze mask hiện tại diễn giải chỉ dẫn “hai lớp cuối” thành hai HTSAT blocks cuối của stage cuối và BERT layers 10–11. Đây là ánh xạ có thể kiểm tra từ runtime, nhưng không có bằng chứng paper chỉ đích danh chính xác các module đó.
6. Checkpoint weights/mga-clap.pt được kiểm tra tương thích với source và nạp strict; chưa xác minh nó giống từng byte checkpoint được dùng trong lần chạy tạo Table 2.
7. Mốc công bố Expert 2 trên validation là SRCC 0.6297, LCC 0.6562, KTAU 0.4550, MSE 3.7208. Chưa chạy full training trong kết quả workspace hiện có, nên không khẳng định đạt mốc.

Nguồn đối chiếu: [MOESCORE.pdf](MOESCORE.pdf), [TASALIGN paper](Sun_NPU_Sun_NPU.pdf), [model/Basic_comp.py](model/Basic_comp.py), [mga-clap_finetune.md](mga-clap_finetune.md), source MGA-CLAP trong thư mục [MGA-CLAP](MGA-CLAP).

## 8. Lệnh chạy

Từ thư mục gốc MOESCORE:

    python Train_Expert2.py --check-data
    python Train_Expert2.py --config configs/expert2_finetune.yaml
    python Train_Expert2.py --config configs/expert2_finetune.yaml --resume weights/expert2_run/last.pt
    python Train_Expert2.py --eval-only --checkpoint weights/expert2_best.pt
    python Train_Expert2.py --eval-only --split test --checkpoint weights/expert2_best.pt

Có thể đổi data root bằng --data-root. Xem [Expert2_README.md](Expert2_README.md) và [requirements-expert2.txt](requirements-expert2.txt) để biết môi trường, Colab và dependencies.
