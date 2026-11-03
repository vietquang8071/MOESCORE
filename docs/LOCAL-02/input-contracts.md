# LOCAL-02.3 — Kiểm toán hợp đồng đầu vào và tiền xử lý

**Phạm vi:** phân tích tĩnh mã nguồn cục bộ; không chạy mô hình, không đọc checkpoint/dataset nhị phân.
**Trạng thái:** **PASS — STATIC AUDIT COMPLETE**. Trạng thái này xác nhận truy vết mã nguồn, không xác nhận tương thích runtime hay parity dự đoán.

## 1. Tóm tắt điều hành

Bốn pipeline fine-tune không dùng chung cách chuẩn bị audio: Expert 1 nhận 48 kHz, Expert 2 nhận 32 kHz, Expert 3 và 4 nhận 16 kHz. Cả bốn chuyển âm thanh thành mono, nhưng khác decoder, resampler, chuẩn hóa biên độ, cắt và đệm. Không thể tạo một waveform duy nhất rồi đưa nguyên trạng cho cả bốn expert.

Cả bốn expert nhận caption thô dạng chuỗi tại ranh giới fine-tuned model và tự tokenize bên trong. Khóa caption_tokens ở Expert 1/2 chỉ là tên cho list[str], không phải tensor token IDs. Tokenizer khác nhau; không thể dùng chung token IDs. Expert 3 dùng encoder/tokenizer suy ra từ M2D checkpoint; cấu hình hiện tại chọn họ BERT-base, nhưng nhánh gọi truncate=False, nên max_length=512 không áp đặt giới hạn token.

Dataset MOESCORE gốc chưa tạo batch tương thích: thiếu clap_wavs, training trả captions thay cho caption_tokens, audio chưa resample/mono hóa, và nhánh resample MGA có lỗi khi sample rate đầu vào là 32 kHz. Ngoài ra wav_lens có thể vượt chiều dài đã cắt. Báo cáo chỉ ghi nhận; không sửa mã nguồn.

## 2. Expert 1 — LAION-CLAP

### Dataset và đường xử lý đang dùng

Entry point Train_Expert1.py khởi tạo Expert1Dataset cho train và validation, cùng collate_batch; không có tiền xử lý khác được dùng ở hai DataLoader này. CSV bắt buộc wav_file_name, text, average_score; audio được nối với wav_dir sau khi bỏ dấu / đầu tên file (L29-L40, L45-L80, L199-L212).

### Audio contract

- torchaudio.load(path) nạp waveform channel-first. Code không truyền normalize=; theo mặc định thông dụng của torchaudio, output là float32 đã chuyển PCM integer sang miền float. Mã này không có bước peak normalization riêng; version torchaudio cần được xác nhận khi parity test.
- Chuyển mono bằng trung bình các channel (mean(dim=0)), sau đó resample bằng torchaudio.functional.resample đến 48.000 Hz.
- Độ dài đầu ra cố định 10 giây = 480.000 samples. Audio dài hơn: train lấy crop ngẫu nhiên; validation lấy crop giữa. Audio ngắn: lặp waveform nguyên số lần rồi zero-pad phần còn lại. Audio rỗng bị từ chối; không có ngưỡng duration tối thiểu khác.
- Item có clap_wav dạng [1, 480000]; collate tạo clap_wavs [B,1,480000]. forward squeeze channel thành [B,480000], rồi gọi get_audio_embedding_from_data(..., use_tensor=True). move_batch chuyển tensor audio/score lên device; danh sách caption Python được giữ nguyên (L140-L144).
- Ngoài các bước thấy trong trainer, LAION-CLAP có thể thực hiện frontend nội bộ. Source fine-tune chỉ gọi API; chi tiết mel/audio normalization phụ thuộc phiên bản laion_clap, chưa được xác nhận tại đây.

### Text contract và batch

Caption lấy từ text, được kiểm tra không rỗng nhưng giữ nguyên chuỗi, không strip/lowercase tại dataset. collate_batch đặt danh sách chuỗi vào khóa tên caption_tokens; Expert1.forward truyền trực tiếp danh sách đó cho get_text_embedding. Không có tensor token IDs được tạo trong dataset. Chiều dài, padding/truncation, tokenizer config nội bộ của API này không khai báo trong source fine-tune, do đó UNVERIFIED; cần xác nhận bằng đúng bản laion_clap/pretrained resource trước parity test. Tensor score là float32; num_class là long; model nhận audio và list caption, không cần mask hay length.

### Train / validation / inference

Train và validation dùng cùng dataset/collate; khác biệt audio có chủ đích là random crop so với crop giữa. Audio ngắn xử lý giống nhau. Trong các source Expert 1 đã kiểm tra không tìm thấy entry point inference độc lập; không được mặc định cách crop inference nếu không gọi dataset validation hiện có.

**Bằng chứng:** Expert1Dataset.__getitem__, collate_batch, CLAPAudioTextEncoder.forward, Expert1.forward, DataLoader tại expert1_finetune/Train_Expert1.py:L25-L95,L98-L144,L199-L212; validation tại L267-L285. **Trạng thái:** STATIC_VERIFIED cho preprocessing nội bộ repo; UNVERIFIED cho mặc định/tokenization/frontend của dependency ngoài.

## 3. Expert 2 — MGA-CLAP

### Dataset và đường xử lý đang dùng

expert2_finetune/Train_Expert2.py dùng Expert2Dataset và collate_expert2 cho cả train, validation và evaluate_checkpoint trên validation/test. CSV chấp nhận wav_file_name hoặc wave_file_name; score chấp nhận average_score hoặc average_semantic_alignement_score; cột caption là text. Tên file phải là basename; dataset từ chối tên WAV trùng lặp (expert2_finetune/expert2_data.py:L18-L44). Thư mục/split được nối trong get_splits/get_eval_split (expert2_finetune/Train_Expert2.py:L46-L78).

### Audio contract

- Đọc WAV bằng scipy.io.wavfile.read.
- Integer PCM được chuyển float32 và chia cho max(abs(dtype.min), dtype.max); float input chỉ cast float32, không scale lại. Stereo được lấy trung bình theo channel.
- Nếu sample rate khác 32 kHz, resample bằng scipy.signal.resample_poly với tỉ số rút gọn theo GCD; không có waveform z-score/peak normalization trong dataset. Audio rỗng bị từ chối; không đặt minimum duration khác.
- Cắt phần đầu sau 10 giây ở 32 kHz, tức tối đa 320.000 samples; audio ngắn được zero-pad bên phải tới đúng chiều dài. Không có random crop. Dataset item [T]; collate thêm channel để mga_wavs có [B,1,320000]; MGAEncoder.forward squeeze channel trước khi gọi ASE, nên ASE nhận [B,320000].
- HTSAT tạo spectrogram/log-mel bên trong. audio_args cấu hình sr=32000, n_fft=1024, hop_length=320, n_mels=64, max_length=10, mono; AudioFeature dùng torchlibrosa Spectrogram/LogmelFilterBank. create_expert2 tắt training.spec_augmentation trước khi tạo ASE. Không tái tạo frontend này ở dataset.

mga_wavs là representation chuyên biệt của Expert 2, không thể thay bằng clap_wavs: khác sample rate, cắt/đệm và waveform frontend. Dataset/collate của Expert 2 sinh riêng representation đó.

### Text contract và batch

Dataset strip khoảng trắng đầu/cuối, giữ nguyên hoa/thường ở source, rồi collate đưa raw strings vào caption_tokens (list chiều dài B). MGAEncoder truyền chuỗi trực tiếp tới ASE.encode_text; TextEncoder nội bộ dùng BertTokenizer của bert-base-uncased, padding=longest, truncation=True, max_length=30, trả IDs/mask qua model trên thiết bị của encoder. Token IDs/mask là dữ liệu nội bộ, không phải đầu vào batch bên ngoài. Trainer chuyển mga_wavs và scores lên device; caption strings ở dạng Python (expert2_finetune/Train_Expert2.py:L81-L93,L180-L188). ASE bỏ token đầu khi chiếu word embeddings; special_tokens_mask được đảo thành chỉ báo token hợp lệ cho MSC. Không dùng tokenizer Expert 1.

### Train / validation / inference

Train, validation và test evaluation cùng Expert2Dataset/collate; xử lý audio/caption xác định, không phụ thuộc chế độ train. Validation/test dùng shuffle=False; train shuffle. YAML kiến trúc upstream đặt spec augmentation true, nhưng factory fine-tune ghi đè false; đây là lựa chọn thực tế của wrapper đang dùng, không phải khác biệt input giữa các split.

**Bằng chứng:** expert2_finetune/expert2_data.py:L14-L85; expert2_finetune/Train_Expert2.py:L46-L78,L81-L115,L127-L146,L180-L205; expert2_finetune/expert2_model.py:L12-L23,L68-L71,L86-L99; expert2_finetune/MGA-CLAP/models/text_encoder.py:L38-L74; expert2_finetune/MGA-CLAP/models/ase_model.py:L62-L93,L95-L144; expert2_finetune/MGA-CLAP/models/feature_extractor.py:L12-L38; expert2_finetune/MGA-CLAP/settings/inference_example.yaml:L6-L30. **Trạng thái:** STATIC_VERIFIED; chưa chạy model/tokenizer.

## 4. Expert 3 — M2D-CLAP

### Dataset và đường xử lý đang dùng

expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py gọi MoescoreExpert3Dataset với cấu hình train/validation; inference gọi cùng dataset/collate cho validation, test hoặc custom set. Config bắt buộc 16 kHz và 10 giây; train batch 8, shuffle, drop_last=True, validation batch 1, không có waveform augmentation (train_expert3.py:L68-L90,L108-L176). Inference giữ dataset/preprocessing tương tự; batch_size chỉ điều chỉnh batching (expert3_finetune/XACLE_2027_baselinemodel/inference_expert3.py:L18-L65).

### Audio contract

- Đọc bằng soundfile.read(..., dtype='float32', always_2d=True), lấy trung bình channel; resample bằng scipy.signal.resample_poly sang 16 kHz nếu cần. Không có bước chuẩn hóa biên độ riêng trong dataset.
- Cắt phần đầu ở 10 giây = 160.000 samples, zero-pad bên phải nếu ngắn. Mọi mẫu dataset do đó có waveform cố định. Audio rỗng bị từ chối; không có minimum duration khác.
- Item [1,160000]; collate [B,1,160000]; MoescoreExpert3.encode_modalities chấp nhận [B,T] hoặc [B,1,T], squeeze channel singleton, từ chối các shape còn lại. _move_batch chuyển tensor lên device nhưng giữ list captions ở dạng Python; tokenizer chuyển token inputs lên device của text encoder.
- PortableM2D lấy sample rate/frontend từ tên thư mục checkpoint. MoescoreExpert3 đặt checkpoint vào thư mục m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025; hậu tố xác định model 16 kHz và text family B. Frontend M2D tạo log-mel nội bộ, chuẩn hóa theo mean/std của checkpoint; 16 kHz dùng FFT/window 400, hop 160, 80 mel bins, fmax 8000. Encoder chia mel frames thành các chunk theo input_size, padding patch nội bộ; không nhận wav_lens/mask. Không tự tạo spectrogram ngoài model.

### Text contract và batch

Dataset strip chuỗi, collate trả list raw captions. Model kiểm tra kiểu list/tuple chuỗi và batch size khớp audio. Checkpoint layout đã mã hóa B (BERT-base); get_text_encoder chọn BertXEncoder, tải tokenizer/model google-bert/bert-base-uncased; checkpoint có hậu tố TI được dùng để nạp text weights. Tokenizer dùng padding=longest. encode_clap_text gọi encoder với truncate=False; vì vậy max_length=512 ở chữ ký BertXEncoder không được bật thành giới hạn truncation. Độ dài batch thực tế phụ thuộc caption; giới hạn vị trí của BERT có thể làm input dài quá lỗi runtime. Cần test riêng caption dài; không tự truncate trong adapter. Text encoder lấy vector CLS, rồi text_proj nếu checkpoint có. Device transfer tokenizer diễn ra trong encoder.

### Train / validation / inference

Cùng config và cùng dataset/collate cho cả ba; không random crop hoặc augmentation. Khác biệt là train shuffle/drop-last và batch 8, validation/inference giữ thứ tự, inference cho phép batch size do CLI chỉ định. Audio/caption không được sắp xếp độc lập. concat_assignment ảnh hưởng thứ tự cross-feature sau encode, không phải tiền xử lý input; phải giữ đúng config checkpoint.

**Bằng chứng:** expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py:L17-L116; expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py:L68-L137,L157-L176,L241-L250; expert3_finetune/XACLE_2027_baselinemodel/inference_expert3.py:L18-L65; expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py:L15-L51,L210-L266; expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py:L149-L158,L246-L257,L266-L298,L301-L326,L375-L422,L449-L467,L545-L566,L569-L601. **Trạng thái:** STATIC_VERIFIED về nhánh và lời gọi; DERIVED về token length dynamic; chưa kiểm tra checkpoint/tokens runtime.

## 5. Expert 4 — BEATs + CLAP + SeqCoAttn

### Dataset và đường xử lý đang dùng

Training tạo Expert4Dataset mà không truyền sample-rate/duration override; dataset default 16 kHz, 10 giây. Validation và inference cũng dùng cùng class, collate_expert4 và default đó (expert4_finetune/trainer.py:L216-L223; expert4_finetune/inference.py:L62-L76).

### Audio contract

- Đọc bằng scipy.io.wavfile.read. Integer PCM được chuyển float32: unsigned được center quanh 0 và scale; signed chia theo biên độ dtype. Float WAV chỉ cast float32, không rescale. Stereo được lấy trung bình.
- scipy.signal.resample_poly chuyển sang 16 kHz. Cắt phần đầu tối đa 160.000 samples; dataset không pad từng item. Mẫu rỗng/rate không dương bị từ chối; không có min duration lớn hơn một sample.
- Collate lưu wav_lens là số samples hợp lệ sau resample/crop, trước pad, dtype long [B]; pad zero tới 160.000 rồi tạo wavs float32 [B,1,160000].
- AudioEncoder.forward yêu cầu đúng [B,1,N] và lengths [B], kiểm tra 0 < lengths <= N. Lengths tính bằng samples ở sample rate 16 kHz; trainer/inference chuyển cả wavs và wav_lens lên device.
- BEATs chuẩn hóa mỗi waveform bằng mean/std trên toàn chiều padded N, sau đó tạo mask arange(N) >= length (True = padding). Do z-score chạy trên waveform đã pad, zero padding góp phần vào mean/std; wav_lens chỉ tạo mask sau đó. Frontend BEATs sinh fbank nội bộ; padding mask được chuyển qua forward_padding_mask trước encoder. Khi tích hợp phải giữ cả độ dài thật và quy tắc pad như source hiện tại.

### Text contract và mask

captions là list raw strings đã strip trong dataset; model không nhận token tensor. TextEncoder dùng tokenizer clap.tokenize từ LAION-CLAP checkpoint; padding max_length, truncation true, max_length=77. Text branch tạo hidden states từ IDs/mask; projection áp dụng sau khi bỏ token đầu tiên, mask cũng bỏ cùng vị trí. Tokenizer attention mask có 1=valid, 0=pad; chuyển thành text_mask = attention[:,1:].eq(0), vì thế True=padding. SeqCoAttn dùng trực tiếp các mask này dưới dạng key_padding_mask, trong đó True bị loại khỏi key positions. Mask audio/text là bool [B,T]; mỗi mẫu phải còn ít nhất một vị trí hợp lệ. Text encoder nội bộ chuyển IDs/attention lên device.

### Train / validation / inference

Ba giai đoạn cùng dataset và collate; không có random crop. Split khác ở shuffle và batch size, không ở audio/text preprocessing. Training/evaluation/inference chuyển wavs và wav_lens lên device; chuỗi Python được giữ lại. Chế độ model có thể ảnh hưởng dropout/encoder behavior, nhưng preprocessing input không đổi.

**Bằng chứng:** expert4_finetune/dataset.py:L23-L115; expert4_finetune/trainer.py:L193-L245,L275-L289; expert4_finetune/inference.py:L14-L33,L50-L76; expert4_finetune/audio_encoder.py:L20-L76; expert4_finetune/text_encoder.py:L39-L60,L111-L124; expert4_finetune/model.py:L8-L24; expert4_finetune/seq_coattn.py:L6-L22. **Trạng thái:** STATIC_VERIFIED; sequence lengths sau fbank phụ thuộc runtime BEATs, không kiểm tra ở đây.

## 6. Ma trận tương thích audio giữa experts

| Thuộc tính | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|---|---|---|---|---|
| Sample rate vào model | 48 kHz | 32 kHz | 16 kHz | 16 kHz |
| Loader | torchaudio | scipy wavfile | soundfile | scipy wavfile |
| Mono | mean channel | mean channel | mean channel | mean channel |
| Resample | torchaudio resample | scipy polyphase | scipy polyphase | scipy polyphase |
| Biên độ ở dataset | mặc định torchaudio; không normalize thêm | PCM int scale; float giữ nguyên | soundfile float32; không normalize thêm | PCM int scale; float giữ nguyên |
| Giới hạn | 10 s | đầu 10 s | đầu 10 s | đầu 10 s |
| Audio ngắn | repeat rồi zero pad | zero pad | zero pad | zero pad tại collate |
| Train crop | random nếu >10s | không | không | không |
| Val/inference crop | center nếu >10s | đầu | đầu | đầu |
| Input batch | [B,1,480000] | [B,1,320000] | [B,1,160000] | [B,1,160000] + lengths |
| Frontend trong model | LAION-CLAP API; chi tiết phiên bản ngoài source | torchlibrosa log-mel + HTSAT | nnAudio log-mel + M2D norm | BEATs fbank |
| Trạng thái | STATIC_VERIFIED trừ API details | STATIC_VERIFIED | STATIC_VERIFIED | STATIC_VERIFIED |

E3 và E4 cùng 16 kHz, channel averaging và resample_poly với tỉ số tương đương; có thể thử dùng chung bước decode/mono/resample sau khi so sánh PCM integer, WAV float và biên xử lý. Padding của E3 xảy ra trong dataset; E4 pad trong collate và chuẩn hóa sau padding, vì vậy hai tensor cuối không được coi là cùng ngữ nghĩa nếu không tái tạo chính xác length/pad boundary. E1/E2 không nên chia sẻ waveform cuối do sample rate, resampler/crop và amplitude rules khác nhau.

**Đồ thị đề xuất:** CSV ghép sample_id + raw caption + score + audio_path → đọc file/giữ metadata chung → nhánh preprocess riêng E1 (48k) / E2 (32k) / E3 (16k) / E4 (16k + valid length) → collate theo expert → adapter truyền đúng key và device. Chỉ hợp nhất decode/mono/resample trung gian sau khi parity kiểm chứng; frontend mel/fbank/tokenization tiếp tục nằm trong model như hiện nay.

## 7. Ma trận tương thích text

| Expert | External input | Tokenizer/model | Padding/truncation | Mask |
|---|---|---|---|---|
| 1 | list[str], khóa caption_tokens | LAION-CLAP get_text_embedding; tokenizer nội bộ chưa được khai báo trong trainer | UNVERIFIED | Không truyền external mask |
| 2 | list[str], khóa caption_tokens | BERT bert-base-uncased trong ASE | longest, truncation, max 30 | Internal special-token validity mask cho MSC |
| 3 | list[str], khóa captions | checkpoint family B → Google BERT-base-uncased | padding longest; gọi truncate=False; max 512 không giới hạn thực tế | Internal attention mask cho BERT; không trả external mask |
| 4 | list[str], khóa captions | tokenizer LAION-CLAP gắn với checkpoint | padding max_length 77, truncation true | True=pad, bỏ vị trí token đầu |

Caption raw có thể được giữ chung; token IDs không thể chia sẻ. Expert 2 và 3 đều có BERT nhưng tokenizer/source khác nhau và độ dài/mask khác nhau. Không chuẩn hóa casing ở unified dataset; chỉ giữ quy tắc strip theo từng expert. Riêng Expert 1 không strip trước CLAP API.

## 8. Nhất quán train / validation / inference

| Expert | Train so với validation | Inference hiện có | Đánh giá |
|---|---|---|---|
| 1 | Random crop so với center crop khi audio >10s; ngắn thì cùng repeat+pad | Không thấy inference riêng; validation loader là đường đánh giá hiện hữu | Random/center là khác biệt có chủ đích; future inference cần chọn center nếu cần parity validation |
| 2 | Cùng deterministic crop đầu/pad/tokenize | evaluate_checkpoint dùng cùng dataset/collate cho validation/test | Nhất quán input |
| 3 | Cùng dataset; train shuffle/drop-last và batch 8, validation batch 1 | expert3_finetune/XACLE_2027_baselinemodel/inference_expert3.py cùng dataset/collate | Nhất quán input; batch có thể thay đổi độ dài text padding=longest nhưng đầu ra clip embedding |
| 4 | Cùng dataset/collate; chỉ thay shuffle/batch/model mode | inference.py dùng cùng dataset/collate | Nhất quán input; phải giữ padding/length để không đổi z-score |

Các khác biệt này được xác nhận từ đường code; parity số học không được chạy.

## 9. Kiểm toán dataset MOESCORE hiện tại

### Schema và xử lý

XACLEDataset dùng pandas và đòi các cột wav_file_name, average_score, text, num_class; đường dẫn nối wav_dir và tên file (bỏ slash đầu). torchaudio.load đọc waveform nhưng dataset không lấy sample rate làm đầu vào cho conversion và không mono hóa. wav được pad/cắt theo max_sec * sr; mga_wav được tạo bằng một Resample(orig_freq=16000,new_freq=32000) cố định nếu sr != 32000. Hai rủi ro xác nhận tĩnh: nếu file 32 kHz thì mga_wav chưa được gán; nếu file không phải 16 kHz thì transform vẫn giả định đầu vào 16 kHz. Không sửa các lỗi này trong task audit.

collate_fn tạo wavs và mga_wavs theo channel-first [B,C,T]; lengths được lấy từ chiều dài trước khi truncate, nên có thể lớn hơn tensor đã cắt và bị AudioEncoder Expert 4 từ chối. Training collate trả captions nhưng không có caption_tokens; inference collate trả caption_tokens nhưng không có captions, không có wav_lens và không tạo mga_wavs. Cả hai giữ raw captions, chưa tokenize. Train_MOE.train chuyển batch lên device và dùng một batch cho cả experts; MOEModel.forward gọi cả bốn expert trên cùng dict.

### Ma trận tương thích hiện tại

| Yêu cầu | MOESCORE hiện tại | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|
| Sample rate | sr từ caller; không resample wav; effective runtime rate chưa rõ | 48k | 32k | 16k | 16k |
| Audio preprocess | torchaudio decode; không mono; nhánh MGA 16→32 cố định có lỗi | Cần mono/resample/crop riêng | Cần mga_wavs đúng 32k | Cần mono 16k, first 10s, pad | Cần mono 16k, first 10s, pad + lengths đúng |
| Audio layout | wavs [B,C,T]; mga_wavs [B,C,T] nếu nhánh thành công | Có thể map nếu mono và shape đúng | Legacy E2 dùng wavs, fine-tuned E2 cần mga_wavs | Fine-tuned chấp nhận [B,1,T]/[B,T]; legacy dùng caption_tokens | Keys phù hợp train, nhưng sample rate/padding/length cần đúng |
| Text representation | train captions; inference caption_tokens; cả hai raw string | cần list[str] dưới tên caption_tokens | cần list[str] dưới caption_tokens | fine-tuned chấp nhận captions raw; legacy E3 lấy caption_tokens | cần captions raw |
| Tokenization | không tokenize | internal CLAP | internal BERT | internal theo checkpoint | internal CLAP tokenizer |
| Audio lengths | train có wav_lens trước truncate; inference không có | không cần | không cần trong fine-tuned path | không cần | cần samples [B]; hiện không luôn hợp lệ |
| Mask | không tạo | không external | internal | không external | Audio/text mask nội bộ, True=padding |
| Kết luận | — | ADAPTER_MAPPING_REQUIRED + DATASET_CHANGE_REQUIRED | DATASET_CHANGE_REQUIRED | ADAPTER_MAPPING_REQUIRED + DATASET_CHANGE_REQUIRED | DATASET_CHANGE_REQUIRED |

configs/config.yaml chứa top-level sample_rate và các section clap_encoder/mga_encoder; trong khi Train_MOE.train đọc cfg['audio_encoder']['sample_rate']. utils.load_config mặc định đọc JSON. Cấu hình runtime được truyền cho train(cfg) không được xác định từ source hiện có; vì vậy không khẳng định sample rate hiện tại trong run MOESCORE. Khi dùng legacy Basic_comp, Expert 1 đòi clap_wavs; Expert 2/E3 đòi caption_tokens; E4 đòi captions và wav_lens. Các key batch train/inference hiện tại không đồng nhất với toàn bộ yêu cầu này.

**Bằng chứng:** data/dataset.py:L10-L31,L34-L87,L89-L135,L138-L183; Train_MOE.py:L40-L78,L117-L125,L159-L171; model/MOE_model.py:L28-L89; model/Basic_comp.py:L244-L336; configs/config.yaml:L13-L44; utils.py:L7-L12,L32-L65. **Trạng thái:** STATIC_VERIFIED về code và mismatch key; UNVERIFIED về config được truyền thật và sample rate thực tế của từng file.

## 10. Batch contract thống nhất được đề xuất

Giữ chung identity, caption và label; audio representations phải tách theo expert. Dataset/collate tạo tensor trên CPU; adapter chuyển tensor cần thiết lên device. Không tạo generic caption_tokens tensor. Các tên trường dưới đây là đề xuất cho LOCAL-02.4, chưa implement.

| Key | Producer → Consumer | Type / shape (sample → batch) | Dtype / device | Audio rate / tokenizer / mask | Stage và evidence |
|---|---|---|---|---|---|
| sample_ids | unified dataset → tất cả, dùng để align | chuỗi → list[str] [B] | Python string / CPU | ID xác định duy nhất cặp audio-caption | CSV/dataset refs mục 12 |
| captions | unified dataset → adapter E1–E4 | chuỗi → list[str] [B] | Python string / giữ nguyên CPU đến tokenizer | raw text; strip riêng trong E2–E4; không pretokenize chung | E1 expert1_finetune/Train_Expert1.py:L71-L87; E2 expert2_finetune/expert2_data.py:L68-L85; E3 expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py:L78-L116; E4 expert4_finetune/dataset.py:L73-L115 |
| expert1_wavs | expert-specific preprocessing → E1 | [1,480000] → [B,1,480000] | float32 / model device | 48 kHz; mono; train random crop, eval center; short repeat+pad | Train_Expert1.py:L25-L80,L84-L87 |
| expert2_wavs | expert-specific preprocessing → E2 | [1,320000] → [B,1,320000] | float32 / model device | 32 kHz; mono; first crop + zero-pad; không thay bằng E1 tensor | expert2_finetune/expert2_data.py:L14-L16,L49-L85 |
| expert3_wavs | expert-specific preprocessing → E3 | [1,160000] → [B,1,160000] | float32 / model device | 16 kHz; mono; first crop + zero-pad; M2D mel do model tạo | expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py:L17-L24,L78-L116 |
| expert4_wavs | expert-specific preprocessing + collate → E4 | raw [T]; batch [B,1,160000] | float32 / model device | 16 kHz; mono; cắt đầu; zero-pad giống dataset trước z-score | expert4_finetune/dataset.py:L23-L26,L73-L115; audio_encoder.py:L50-L76 |
| expert4_wav_lens | collate → E4 | sample length → long [B] | int64 / đưa cùng audio lên model device | số samples 16 kHz, trước pad; 0 < len <= 160000 | expert4_finetune/dataset.py:L101-L106; expert4_finetune/audio_encoder.py:L50-L57 |
| scores_raw | dataset → metric/alignment | scalar → float32 [B] | CPU hoặc device theo consumer | thang 0–10 | mỗi expert dataset; root data/dataset.py:L74-L87,L119-L135 |
| scores | dataset/collate → objective dùng trong fine-tune | scalar → float32 [B] | CPU trước batch move | (score-5)/5, score gốc [0,10] | E1 expert1_finetune/Train_Expert1.py:L68-L80; E2 expert2_finetune/expert2_data.py:L68-L74; E3 expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py:L98-L101; E4 expert4_finetune/dataset.py:L107-L115 |

Key thực tế cho E1/E2 có thể được adapter map từ captions sang caption_tokens, nhưng giá trị vẫn là chuỗi thô. E3/E4 nhận raw captions. Chỉ thêm input_ids hoặc attention masks nếu một adapter gọi trực tiếp tokenizer; không dùng chung giữa họ tokenizer khác nhau. Tên chính thức và policy sharing chỉ chốt sau test parity.

## 11. Kiến trúc preprocessing đề xuất

Đồ thị dữ liệu:

- CSV row → sample_id + raw caption + score + audio_path.
- Metadata, caption, label và file identity có thể dùng chung.
- Nhánh E1: decode torchaudio → mono → 48 kHz → crop train/eval riêng → 10s.
- Nhánh E2: decode scipy → PCM scaling/mono → 32 kHz → cắt đầu/pad → 10s.
- Nhánh E3: decode soundfile → mono → resample_poly 16 kHz → cắt đầu/pad → 10s.
- Nhánh E4: decode scipy → PCM scaling/mono → resample_poly 16 kHz → cắt đầu → lưu valid length → collate pad → BEATs.
- Từng expert tạo batch representation riêng; mel/fbank/tokenization giữ bên trong model.

Chỉ hợp nhất decode/mono/resample trung gian sau khi parity kiểm chứng; đặc biệt không bỏ qua crop/pad và z-score Expert 4. Không tạo mel/fbank/token IDs chung bên ngoài các expert.

## 12. Sample alignment và label contract

Mỗi prediction phải ánh xạ về cùng một dòng audio-caption, không chỉ cùng audio filename. Expert 2/3 từ chối filename trùng; Expert 1 không lưu ID trong batch; Expert 4 hỗ trợ cùng filename với caption khác nhau nhưng cấm trùng cặp (name,text). MOESCORE batch có wav_paths nhưng không mang ID cặp tường minh. Do train shuffle, không thể ghép predictions bằng thứ tự list độc lập.

Đề xuất sample_id ổn định từ (split, normalized relative wav filename, exact caption); nếu source có thể lặp nguyên cặp, thêm row key bất biến. Tất cả nhánh audio/text/score phải phát sinh từ cùng row và collate phải bảo toàn cùng permutation. Không deduplicate chỉ theo filename nếu nhiều caption hợp lệ.

Các fine-tuned dataset đều dùng score [0,10] và tạo target (score-5)/5; Expert 1 chỉ kiểm tra num_class tùy chọn rồi mặc định 1, E2 không trả num_class, E3 tính bins (dù CBL tắt theo config), E4 không dùng num_class trong dataset. Unified pipeline phải giữ scores_raw và target convention tách biệt; không thay đổi score hoặc grouping trong task này.

## 13. Yêu cầu mapping đầu vào cho Expert Adapters

| Adapter | Chuẩn hóa batch → model fine-tuned | Điều kiện bắt buộc |
|---|---|---|
| E1 | expert1_wavs và raw captions → clap_wavs, caption_tokens | giữ crop eval giữa; gọi nguyên CLAP API/tokenization |
| E2 | expert2_wavs và raw captions → mga_wavs, caption_tokens | đảm bảo 32 kHz, singleton channel đúng, không feed wavs MOESCORE 16k/48k |
| E3 | expert3_wavs, raw captions → wavs, captions | 16 kHz/10s; giữ checkpoint-configured tokenizer và concat_assignment; không tự trunc |
| E4 | expert4_wavs, expert4_wav_lens, raw captions → wavs, wav_lens, captions | length samples đúng sau cắt/resample; giữ padding trước z-score; không đảo mask True=pad |

Các adapter cần validate batch size/identity đồng nhất trước forward. Không chia sẻ input key nếu tên giống nhưng sample rate/semantic khác. Không triển khai adapter ở LOCAL-02.3.

## 14. Rủi ro và bất tương thích ưu tiên

| Ưu tiên | Rủi ro / bằng chứng | Tác động | Phép phát hiện | Giảm thiểu về sau |
|---|---|---|---|---|
| Cao | Root dataset không resample wav; caller sample-rate config chưa xác định (data/dataset.py:L34-L72, Train_MOE.py:L41-L58) | model nhận waveform khác rate so với fine-tune | đối chiếu sample rate file và tensor theo nhánh | giữ branch resample riêng, fail-fast nếu rate sai |
| Cao | mga_wav không khởi tạo khi source 32 kHz; resampler root cố định 16→32 bất kể rate thật (data/dataset.py:L60-L72) | lỗi UnboundLocalError hoặc tốc độ/độ dài MGA sai | fixture WAV 16/32/48 kHz | dùng conversion dựa trên source rate và kiểm tra output |
| Cao | Audio root không mono; rate/length có thể không phù hợp Expert4 (data/dataset.py:L65-L104, expert4_finetune/audio_encoder.py:L50-L55) | sai input hoặc model từ chối length > padded N | WAV stereo/dài hơn max, assert length<=N | tạo mono/length sau resample-crop theo source Expert4 |
| Cao | Input root thiếu hoặc đổi key theo train/inference (data/dataset.py:L126-L135,L179-L183; expert forwards) | KeyError; model nhận nhầm tensor/caption | contract test từng expert | adapter mapping rõ ràng |
| Cao | Dùng một caption_tokens chung | tokenization sai giữa CLAP/BERT/M2D; E1 API max chưa biết | so sánh tokenizer/reference IDs | giữ raw strings và tokenization nội bộ |
| Cao | Expert1 random train crop vs center eval; audio ngắn repeat+pad (Train_Expert1.py:L57-L66) | prediction parity lệch nếu random path bật khi gating/inference | cùng ví dụ >10s nhiều seed, kiểm tra deterministic | adapter luôn dùng policy eval cho frozen inference |
| Cao | M2D truncate=False mặc dù encoder có max_length=512 (expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py:L456-L462,L556-L564) | caption dài có thể vượt giới hạn vị trí encoder | kiểm tra caption token length/runtime | giữ nguyên source, định nghĩa xử lý chỉ sau khi xác thực checkpoint reference |
| Cao | Expert4 z-score gồm padding (expert4_finetune/audio_encoder.py:L50-L57) | thay pad/duration làm đổi cả phần valid waveform đã chuẩn hóa | so sánh z-score tensor với standalone | tái tạo cùng padded waveform/length boundary |
| Trung bình | Decoder/scaling khác giữa torchaudio, scipy, soundfile | lệch mẫu âm thanh dù cùng WAV | so sánh waveform sau từng biên preprocessing | chỉ dùng decode chung sau parity test |
| Trung bình | Sample ID không nhất quán/thiếu ở Expert1 và root | trộn score/predictions giữa caption cùng audio | join theo sample_id thay vì batch order | giữ ID cặp qua mọi nhánh |
| Trung bình | Audio frontend nội bộ library phụ thuộc checkpoint/version | khác mel/fbank/audio embedding | parity features/outputs khi resources sẵn sàng | khóa phiên bản và gọi model API nguyên trạng |

Không có rủi ro nào được sửa trong phạm vi audit.

## 15. Kế hoạch kiểm chứng sau này

Các test dưới đây chưa chạy:

1. **Dataset, không cần checkpoint:** tạo WAV fixture PCM signed/unsigned và float, mono/stereo, sample rates 16/32/44.1/48 kHz, audio rỗng/ngắn/đúng 10s/dài; kiểm tra mỗi dataset output shape, dtype, resampling target, crop/pad values, caption và score.
2. **Collation, không cần checkpoint:** xác nhận [B,1,T], wav_lens bằng số samples thực sau cắt, lengths không vượt padded length, order/identity không đổi; test mask E4 True đúng vùng padded.
3. **Text tokenizer:** cần đúng tokenizer/cache/pretrained resources. So sánh IDs/masks standalone với adapter cho caption ASCII, unicode, khoảng trắng, caption dài; không gửi Hub request tự động.
4. **Preprocessing parity:** với cùng WAV, so sánh outputs ở biên model: E1 center-crop eval; E2 32k; E3 16k + padding; E4 waveform/length trước BEATs. Exact compare khi cùng library/deterministic; tolerance chỉ với phép resampling/float có căn cứ.
5. **Model input/forward:** cần pretrained resources; sau khi có fine-tuned checkpoint thì load đúng như LOCAL-02.2. Kiểm tra dtype/shape, finite prediction, batch correspondence, prediction standalone so với adapter trên cùng example.
6. **Label/identity:** join standalone/unified bằng sample_id, đối chiếu raw score và normalized score ở từng row; không dựa vào shuffled batch order.
7. **Chế độ frozen:** sau khi restore, eval và khóa params; test input path không bật augmentation/crop ngẫu nhiên; kiểm tra gating backward không cập nhật experts (thuộc tích hợp sau).

Không có checkpoint, dataset hoặc model nào được tải/khởi tạo trong audit này. Tests 1–2 cần fixture; 3–5 cần tokenizer/pretrained resources, và phần parity prediction cần fine-tuned checkpoint.

## 16. Đề xuất cho LOCAL-02.4

- Xác định forward() từng fine-tuned wrapper nhận key nào, shape/dtype và score/features rank; Expert 1 hiện tại chỉ trả score, cần xử lý output feature ở task output-contract riêng.
- Ghi rõ feature dimension checkpoint-dependent của Expert 1/3; không suy từ legacy gate_input_dim.
- Chốt adapters giữ raw captions, gọi tokenizer nội bộ và không đổi sequence/mask.
- Kiểm tra interface Expert 4 mask/length sau fbank; không đưa mask ngược convention.
- Khóa cùng sample IDs và row ordering cho cả bốn dự đoán; thống nhất score shape chỉ tại adapter boundary.
- Giữ nguyên các preprocessing policies tại mục 10–13; LOCAL-02.3 không triển khai contract.

## 17. Tham chiếu nguồn

- Expert 1: expert1_finetune/Train_Expert1.py:L25-L95,L98-L144,L147-L180,L199-L212,L267-L285.
- Expert 2 data/training: expert2_finetune/expert2_data.py:L14-L85; expert2_finetune/Train_Expert2.py:L46-L115,L127-L146,L180-L205.
- Expert 2 model: expert2_finetune/expert2_model.py:L12-L23,L68-L99; expert2_finetune/MGA-CLAP/settings/inference_example.yaml:L1-L36; expert2_finetune/MGA-CLAP/models/ase_model.py:L62-L93,L95-L144; expert2_finetune/MGA-CLAP/models/audio_encoder.py:L12-L72; expert2_finetune/MGA-CLAP/models/text_encoder.py:L38-L74; expert2_finetune/MGA-CLAP/models/feature_extractor.py:L12-L38.
- Expert 3: expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json:L1-L55; expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py:L17-L116; expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py:L68-L176,L241-L250; expert3_finetune/XACLE_2027_baselinemodel/inference_expert3.py:L18-L65; expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py:L15-L51,L210-L266; expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py:L149-L158,L246-L257,L266-L326,L375-L422,L449-L467,L545-L601.
- Expert 4: expert4_finetune/dataset.py:L23-L115; expert4_finetune/trainer.py:L193-L245,L275-L289; expert4_finetune/inference.py:L14-L33,L50-L76; expert4_finetune/audio_encoder.py:L20-L76; expert4_finetune/text_encoder.py:L39-L60,L111-L124; expert4_finetune/model.py:L8-L24; expert4_finetune/seq_coattn.py:L6-L22.
- MOESCORE legacy/current: data/dataset.py:L10-L31,L34-L135,L138-L183; Train_MOE.py:L40-L78,L117-L125,L159-L171; model/MOE_model.py:L28-L89; model/Basic_comp.py:L244-L336; configs/config.yaml:L13-L44; utils.py:L7-L12,L32-L65.
- Báo cáo trước: docs/LOCAL-02/expert-architecture-audit.md; docs/LOCAL-02/checkpoint-compatibility.md.

## Kết luận

**PASS — STATIC AUDIT COMPLETE.** Đã truy vết source-level đường preprocessing và contract đầu vào của cả bốn expert, đánh giá dataset MOESCORE, và đề xuất contract có nhánh audio riêng cùng caption thô chung. Các chi tiết thuộc dependency ngoài, checkpoint/tokenizer cụ thể và prediction parity được đánh dấu UNVERIFIED; không có kết quả runtime nào được tuyên bố. Không sửa code sản xuất.
