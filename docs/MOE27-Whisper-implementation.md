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

Khi model local/cache đầy đủ xuất hiện, checkpoint identity chứa SHA-256 của config, frontend/tokenizer assets và từng checkpoint weight/index file. Với Hub cache, fingerprint dùng revision đã resolve nếu Transformers cung cấp. Preprocessing identity ghi lại sample rate, waveform/log-Mel shapes, chính xác lời gọi tokenizer/model, quy tắc pooling, device/dtype và mã preprocessing unified.

## Cache component

`data.whisper_feature_cache.WhisperFeatureCacheWriter` ghi riêng từng split, không dùng global per-expert cache có trường score:

```text
<output-dir>/<split>/
  manifest.json
  shard-00000000.npz
  shard-00000001.npz
```

Mỗi shard giữ `sample_ids`, `whisper_audio_features [N,1280]` và `whisper_text_features [N,1280]` dưới dạng NumPy float32. Manifest schema v1 giữ split, checkpoint identity, preprocessing identity, dataset identity, extraction version, dimension `{audio:1280,text:1280,total:2560}`, ID theo thứ tự và SHA-256 của từng shard. Dataset identity là digest có thứ tự của sample ID, hash caption thô và hash nội dung WAV; vì thế sửa caption/audio hoặc đổi thứ tự CSV sẽ từ chối cache cũ. Preprocessing identity cũng ghi hash mã preprocessing unified và batch size.

Shard và manifest được ghi nguyên tử. Khi mở lại cache, chương trình xác minh metadata, hash, dimensions, finite values và duplicate ID; sai checkpoint/preprocessing/dataset/split sẽ bị từ chối. Resume chỉ chấp nhận cache IDs là prefix theo đúng thứ tự dataset và kết thúc ở ranh giới batch. Điều này giữ cùng nhóm caption đã padding; đổi batch size/device/dtype làm identity khác. `--limit-samples` trong extraction cache phải kết thúc ở ranh giới batch (trừ khi lấy toàn bộ split). Các điều kiện này cần thiết vì decoder mean gồm vị trí padding nên phụ thuộc caption dài nhất trong mỗi batch.

Thư mục output mặc định là `result/whisper_feature_cache`; script tạo thư mục con riêng cho split. Cache chỉ chứa đặc trưng, không chứa model weights.

## Chạy thử và trích xuất

Smoke test chạy một item với batch size 1 và không ghi cache:

```bash
python3 scripts/extract_whisper_features.py \
  --config configs/whisper_top2.example.yaml \
  --metadata-csv /data/meta_data/validation_average.csv \
  --audio-dir /data/wav/validation \
  --split validation \
  --smoke-test
```

Trích xuất một split, mặc định batch size 1:

```bash
python3 scripts/extract_whisper_features.py \
  --config configs/whisper_top2.example.yaml \
  --metadata-csv /data/meta_data/train_average.csv \
  --audio-dir /data/wav/train \
  --split train \
  --output-dir result/whisper_feature_cache
```

Có thể đặt `--batch-size` lớn hơn khi đủ VRAM. Script chỉ xử lý một split mỗi lần và không nạp LAION. Chỉ dùng `--allow-download` nếu muốn cho phép Hugging Face truy cập mạng; mặc định script offline.

Khi dùng `--limit-samples` trong chế độ ghi cache, giới hạn phải là bội số của `--batch-size` nếu chưa lấy hết split. Nếu chỉ muốn kiểm tra một mẫu, dùng `--smoke-test`.

## Kiểm thử và parity

Test mock được thêm tại `tests/test_whisper_top2.py` và `tests/test_whisper_feature_cache.py`. Chúng khóa input contract, lời gọi frontend/tokenizer/model, pooling gồm padding, freeze/eval/inference mode, cấu hình offline local, cache alignment/resume, duplicate IDs, identity mismatch, finite values và round-trip.

Kết quả kiểm tra tại worktree:

- Trong môi trường `conda nlp` (Python 3.10.21, PyTorch 2.14.1+cu130, Transformers 4.57.6): `conda run -n nlp env PYTHONPATH=/home/vitquay1708/.cache/uv/archive-v0/UfEXJ4HVLkLakVBe/lib/python3.12/site-packages python -m pytest tests/test_whisper_top2.py tests/test_whisper_feature_cache.py -q`: PASS, 26 tests. Pytest được lấy từ uv cache vì env `nlp` không cài pytest.
- Cùng môi trường, `tests/test_kaggle_dependencies.py`: PASS, 4 tests.
- Toàn suite `pytest -q --tb=no`: 394 passed, 28 failed, 1 skipped. Các lỗi ngoài Whisper chủ yếu do test import package `MOESCORE` không tồn tại dưới tên worktree `MOESCORE-Whisper` (23 tests), ba test cần notebook Expert4 vắng mặt, một test không thấy package `expert4_repro`, và một test subprocess diagnostic thất bại mà không in stderr. Không có test Whisper nào thất bại.
- `conda run -n nlp python -m py_compile ...` và `conda run -n nlp python scripts/extract_whisper_features.py --help`: PASS.
- Test mock bao gồm identity sai biến thể Whisper và fingerprint symlink trong Hugging Face snapshot; các test này PASS trong 26 test phía trên.
- Chưa chạy được real-model smoke và numerical parity với Top 2: môi trường `nlp` có PyTorch/Transformers nhưng không có checkpoint `openai/whisper-large-v2`. Vì vậy không tuyên bố đã xác nhận parity số học.

Khi dependencies/checkpoint sẵn sàng, chạy:

```bash
conda run -n nlp env PYTHONPATH=/home/vitquay1708/.cache/uv/archive-v0/UfEXJ4HVLkLakVBe/lib/python3.12/site-packages python -m pytest tests/test_whisper_top2.py tests/test_whisper_feature_cache.py -q
conda run -n nlp env PYTHONPATH=/home/vitquay1708/.cache/uv/archive-v0/UfEXJ4HVLkLakVBe/lib/python3.12/site-packages python -m pytest -q
```

Để xác nhận parity nguồn, cần dùng cùng snapshot large-v2, waveform 16 kHz/10 giây, caption, tokenizer/frontend, device và dtype; so sánh log-Mel `[B,80,3000]`, decoder IDs, hai vector pooled với tolerance theo dtype. Batch caption có độ dài khác nhau có thể làm decoder mean thay đổi do padding, đây là hành vi nguồn cần giữ.

## Tích hợp E6

E6 import `WhisperTop2FeatureExtractor` từ `model.feature_extractors`, gọi với unified batch và nhận `(whisper_audio, whisper_text)`. Lúc huấn luyện residual, E6 có thể đọc cache riêng theo `sample_ids` rồi concatenate `[laion_features, whisper_audio, whisper_text]`; không cần nạp lại Whisper. Không cần thay đổi adapter dùng chung, registry hoặc global feature store.
