# MOE27-02 — Whisper Top 2 feature extractor

## Trạng thái triển khai

- Worktree: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE-Whisper`
- Nhánh: `dev/moe27-whisper`
- Backbone yêu cầu: `openai/whisper-large-v2`
- Stable import path cho E6:

  ```python
  from model.feature_extractors import WhisperTop2FeatureExtractor
  ```

Đây là extractor component. Nó không trả score, không ghép hai vector, không thêm expert vào registry và không sửa E6 residual head.

## Nguồn tham chiếu và quy tắc tương thích

Đã đọc các file của Top 2 reference repository:

- `features/extract_features.py`, hàm `Whisper()`
- `load_pretrained_models/load_model.py`, nhánh `load_model("Whisper", ...)`
- `features/all_feature_dict.py`, cách gọi loader/extractor
- `datasets/fetch_data.py`, cách tạo waveform cố định 10 giây

Extractor dùng đúng `AutoFeatureExtractor`, `WhisperTokenizer` và `WhisperModel`. Với mỗi batch:

1. Chuyển từng waveform về CPU NumPy và flatten; gọi feature extractor với `sampling_rate=16000`, `return_tensors="pt"` và không thêm tùy chọn padding/truncation. Đầu vào được kiểm tra là `[B,80,3000]`.
2. Gọi tokenizer trực tiếp trên `captions_raw` với `return_tensors="pt", padding=True`. Caption gốc được giữ nguyên, gồm cả whitespace; chỉ kiểm tra chuỗi rỗng sau khi strip.
3. Truyền `input_features` và `decoder_input_ids` vào `WhisperModel`. Không truyền attention mask, không gọi `generate()` và không tạo transcript.
4. Lấy trung bình không mask theo trục sequence của `encoder_last_hidden_state` và `last_hidden_state`. Decoder pooling vì vậy bao gồm các vị trí tokenizer padding, đúng như reference.

Mô hình bị freeze và luôn ở evaluation mode. Forward chạy trong `torch.inference_mode()`. `close()` giải phóng các tham chiếu model/tokenizer/frontend và xóa CUDA allocator cache nếu đang dùng CUDA.

## Input và output

Input theo `data.unified_collate.collate_unified`:

```python
batch = {
    "expert3_wavs": torch.Tensor[B, 1, 160000],  # CPU float32, 16 kHz, 10 giây
    "captions_raw": list[str],                  # một caption cho mỗi waveform
}
```

Unified dataset chuyển audio thành mono, resample về 16 kHz, cắt ở 160.000 mẫu và zero-pad nếu ngắn hơn; caption được giữ nguyên từ CSV. Extractor không chạy lại resampling/padding. Có thể truyền thêm `sample_rate=16000` để xác thực; unified batch hiện tại đã mã hóa sample rate trong contract và không cần thêm trường này.

```python
whisper_audio, whisper_text = whisper_extractor(batch)
# mỗi tensor: [B, 1280], float32, finite, trên device đã cấu hình
```

Hai tensor được trả riêng, không concatenate. E6 có thể nối chúng với `laion_features` và dùng cache này trong huấn luyện residual mà không nạp Whisper.

## Cấu hình và checkpoint

Ví dụ nằm tại `configs/whisper_top2.example.yaml`. Mặc định:

- `model_name_or_path: openai/whisper-large-v2`
- `sample_rate: 16000`
- `runtime.offline: true`
- `runtime.device: auto`

`WhisperTop2FeatureExtractor.from_config(config)` nhận đường dẫn model Hugging Face hoặc thư mục local hoàn chỉnh. Thư mục local phải xác định rõ large-v2, có `config.json`, `preprocessor_config.json`, tokenizer assets, và weight file/index cùng các shard. Extractor từ chối ID Whisper khác. Offline mode truyền `local_files_only=True` cho cả ba lớp.

Đã rà `MOESCORE/weights/pretrained` đệ quy. Các file hiện có là `BEATs_iter3_plus_AS2M.pt`, `CLAP_weights_2023.pth`, `checkpoint-30.pth`, `laion-clap.pt` và `mga-clap.pt`; không có Whisper checkpoint, tokenizer, feature extractor config hay Hugging Face model directory. Hugging Face cache chỉ có `models--openai--whisper-tiny/refs/main`, không có snapshot `whisper-large-v2`. Không tải hoặc thay bằng checkpoint khác.

Khi model local/cache đầy đủ xuất hiện, checkpoint identity chứa SHA-256 của config, frontend/tokenizer assets và từng checkpoint weight/index file. Preprocessing identity ghi lại sample rate, waveform/log-Mel shapes, chính xác lời gọi tokenizer/model và quy tắc pooling.

## Cache component

`data.whisper_feature_cache.WhisperFeatureCacheWriter` ghi riêng từng split, không dùng global per-expert cache có trường score:

```text
<output-dir>/<split>/
  manifest.json
  shard-00000000.npz
  shard-00000001.npz
```

Mỗi shard giữ `sample_ids`, `whisper_audio_features [N,1280]` và `whisper_text_features [N,1280]` dưới dạng NumPy float32. Manifest schema v1 giữ split, checkpoint identity, preprocessing identity, extraction version, dimension `{audio:1280,text:1280,total:2560}`, ID theo thứ tự và SHA-256 của từng shard. Shard và manifest được ghi nguyên tử. Khi mở lại cache, chương trình xác minh metadata, hash, dimensions, finite values và duplicate ID; sai checkpoint/preprocessing/split sẽ bị từ chối. Extraction có thể tiếp tục: ID đã lưu được bỏ qua, chỉ các hàng còn thiếu mới được đọc và tính đặc trưng.

Thư mục output mặc định là `result/whisper_feature_cache`; script tạo thư mục con riêng cho split. Cache chỉ chứa đặc trưng, không chứa model weights.

## Chạy thử và trích xuất

Smoke test chạy một item với batch size 1 và không ghi cache:

```bash
python scripts/extract_whisper_features.py \
  --config configs/whisper_top2.example.yaml \
  --metadata-csv /data/meta_data/validation_average.csv \
  --audio-dir /data/wav/validation \
  --split validation \
  --smoke-test
```

Trích xuất một split, mặc định batch size 1:

```bash
python scripts/extract_whisper_features.py \
  --config configs/whisper_top2.example.yaml \
  --metadata-csv /data/meta_data/train_average.csv \
  --audio-dir /data/wav/train \
  --split train \
  --output-dir result/whisper_feature_cache
```

Có thể đặt `--batch-size` lớn hơn khi đủ VRAM. Script chỉ xử lý một split mỗi lần và không nạp LAION. Chỉ dùng `--allow-download` nếu muốn cho phép Hugging Face truy cập mạng; mặc định script offline.

## Kiểm thử và parity

Test mock được thêm tại `tests/test_whisper_top2.py` và `tests/test_whisper_feature_cache.py`. Chúng khóa input contract, lời gọi frontend/tokenizer/model, pooling gồm padding, freeze/eval/inference mode, cấu hình offline local, cache alignment/resume, duplicate IDs, identity mismatch, finite values và round-trip.

Kết quả kiểm tra tại worktree:

- `python3 -m py_compile model/feature_extractors/whisper_top2.py data/whisper_feature_cache.py scripts/extract_whisper_features.py tests/test_whisper_top2.py tests/test_whisper_feature_cache.py`: PASS.
- `python3 scripts/extract_whisper_features.py --help`: PASS.
- `tests/test_whisper_feature_cache.py`: PASS, 8 tests, dùng Python 3.12 cùng pytest đã có trong uv cache.
- `tests/test_kaggle_dependencies.py` (test hồi quy hiện có, không cần PyTorch): PASS, 4 tests.
- `tests/test_whisper_top2.py`: chưa thu thập được vì system Python không có `torch` (`ModuleNotFoundError`).
- `pytest -q --tb=no`: chưa chạy được suite; pytest dừng ở collection với 49 import errors do thiếu `torch` trong môi trường.
- Chưa chạy được real-model smoke và numerical parity với Top 2: không có checkpoint `openai/whisper-large-v2` hoặc runtime PyTorch/Transformers trong environment này. Vì vậy không tuyên bố đã xác nhận parity số học.

Khi dependencies/checkpoint sẵn sàng, chạy:

```bash
pytest tests/test_whisper_top2.py tests/test_whisper_feature_cache.py -q
pytest -q
```

Để xác nhận parity nguồn, cần dùng cùng snapshot large-v2, waveform 16 kHz/10 giây, caption, tokenizer/frontend, device và dtype; so sánh log-Mel `[B,80,3000]`, decoder IDs, hai vector pooled với tolerance theo dtype. Batch caption có độ dài khác nhau có thể làm decoder mean thay đổi do padding, đây là hành vi nguồn cần giữ.

## Tích hợp E6

E6 import `WhisperTop2FeatureExtractor` từ `model.feature_extractors`, gọi với unified batch và nhận `(whisper_audio, whisper_text)`. Lúc huấn luyện residual, E6 có thể đọc cache riêng theo `sample_ids` rồi concatenate `[laion_features, whisper_audio, whisper_text]`; không cần nạp lại Whisper. Không cần thay đổi adapter dùng chung, registry hoặc global feature store.
