# Báo cáo MOE-09.2: cache theo expert và MS-CLAP center crop

**Ngày thực hiện:** 17-11-2026  
**Repository/branch:** `MOESCORE`, `main`  
**Phạm vi:** tách định danh cache theo expert; truyền WAV gốc cho MS-CLAP; thay random crop bằng center crop xác định. Không tích hợp gate chín expert và không trích xuất toàn bộ dataset.

## Kết quả

Đã tách hai lớp định danh. Mỗi expert có cache riêng theo nội dung checkpoint, code liên quan, preprocessing, input/output contract, chính sách inference và phiên bản dependency. Feature store hoàn chỉnh có assembly identity riêng, bao gồm profile, thứ tự expert, các cache identity, chiều feature và định danh manifest nguồn. Vì vậy, thêm expert tạo assembly mới nhưng không đổi cache identity của expert cũ nếu các đầu vào ảnh hưởng đến phép suy luận của expert đó không đổi.

Worker giữ cách chạy tuần tự và khôi phục từ shard. Với mọi shard được yêu cầu, worker xác minh metadata và nội dung trước khi bỏ qua việc khởi tạo model. Shard còn thiếu được tính lại; shard có metadata, hash, identity hoặc tensor không hợp lệ bị từ chối. Shard được ghi nguyên tử. Batch size không thuộc expert identity; shard size chỉ ảnh hưởng phân hoạch shard.

MS-CLAP nhận `wav_paths` được phân giải từ các record của unified dataset và manifest đã xác thực. Adapter tiếp tục nhận tensor `expert2_wavs` theo contract cũ, nhưng audio embedding của MS-CLAP được tạo từ WAV gốc.

## Định danh và xác thực cache

`data/moe_expert_cache.py` định nghĩa schema 2, serialization JSON canonical (`sort_keys`, dấu phân cách ổn định, từ chối NaN/Infinity) và SHA-256. Expert identity không chứa thứ tự/profile toàn cục, checkpoint expert khác, adapter không liên quan hoặc batch size. Hash nội dung checkpoint được dùng thay vì chỉ dựa vào đường dẫn/thời gian sửa đổi. Các dependency liên quan và, khi cần, snapshot model/tokenizer Hugging Face cũng được nhận diện theo nội dung/ref.

Sample/shard identity gắn với split, thứ tự sample ID, SHA-256 WAV, caption nguyên văn, preprocessing/input identity, số lượng và shape score/feature mong đợi. Metadata ghi nhận identity, file hashes, dtype/shape và trạng thái model đã khôi phục/frozen/eval. Khi load, hệ thống kiểm tra hash file, identity, shape, `float32` và tính hữu hạn của mọi giá trị.

Cache của expert nằm dưới `feature_cache/experts/<expert>/<cache_id>/<split>/`; assembly nằm dưới `feature_cache/assemblies/<profile>/<assembly_id>/`. Hai profile có thứ tự expert khác nhau tạo assembly identity khác nhau. Một store có thứ tự expert không khớp sẽ bị từ chối.

### Mở rộng ví dụ

Giả sử shard `expert1`–`expert4` đã được xác thực trong profile `top1_4`. Khi thêm `top2_m2d`, `top2_msclap`, `top2_laion`, `top2_mga` và `m2d_lora`, các expert cũ giữ nguyên cache ID nếu checkpoint, code, preprocessing và contract của chúng không đổi. Worker chỉ chạy cho shard mới hoặc shard bị invalidation. Sau đó profile/thứ tự mới tạo assembly ID khác; không cần tính lại bốn cache cũ. Phần đăng ký profile và extraction chín expert vẫn thuộc milestone sau.

## WAV path cho MS-CLAP

`ExpertInputContract` có trường tùy chọn `requires_wav_paths=False`; contract riêng của `top2_msclap` bật trường này. Các expert khác không bị buộc nhận thêm input. Worker đối chiếu record với unified dataset, lấy đường dẫn audio từ row nguồn đã xác thực và kiểm tra:

- split, sample ID, tên file, caption và thứ tự batch khớp record;
- đường dẫn sau resolve vẫn nằm trong thư mục WAV của split đã chọn;
- không có đường dẫn trùng hoặc traversal;
- file tồn tại/đọc được và SHA-256 bằng manifest.

Batch MS-CLAP có dạng `expert2_wavs`, `captions_raw`, `wav_paths`, cùng metadata `wave_file_names`/`split`. Không suy ra path từ caption hay từ một tên file không được xác thực.

## Center crop và bảo toàn preprocessing MS-CLAP

Đã kiểm tra package trong môi trường Python thực thi: `msclap==1.3.4`. Call graph thực tế là `get_audio_embeddings()` → `preprocess_audio()` → `load_audio_into_tensor()` → `read_audio()`. Loader cài đặt chọn vị trí bằng random khi audio dài hơn giới hạn.

Adapter tạo một subclass cục bộ của lớp MS-CLAP đã cài và override riêng `load_audio_into_tensor`; không sửa `site-packages` hoặc monkey-patch class toàn cục. `read_audio()` của upstream vẫn đảm nhiệm WAV decode, resample và cách xử lý channel. Adapter xác nhận cấu hình model đã nạp là 44.100 Hz và 7 giây. Chính sách được ghi vào model API để đưa vào cache identity:

```json
{
  "audio_crop_policy": "center",
  "audio_duration_seconds": 7,
  "sample_rate_hz": 44100,
  "short_audio_policy": "repeat_then_truncate"
}
```

Với audio dài, crop bắt đầu tại `(n_samples - target_samples) // 2`; audio đúng độ dài được giữ nguyên. Audio ngắn tiếp tục được lặp rồi cắt đúng như upstream, không đổi sang zero padding. Đầu ra loader vẫn là tensor `float32` trên CPU trước khi upstream chuyển vào model. Khác biệt toán học duy nhất có chủ đích so với upstream là chọn đoạn trung tâm thay vì vị trí ngẫu nhiên cho audio dài. Thay crop policy, sample rate, thời lượng hoặc short-audio policy làm đổi cache identity của MS-CLAP.

## Inference thực tế và độ ổn định

Đã chạy lại bằng model/checkpoint thật, không phải mock:

| Mục | Kết quả |
|---|---|
| Python | `/home/vitquay1708/miniconda3/envs/nlp/bin/python` |
| Device | CPU; CUDA không được dùng trong lần smoke test này |
| Split/sample | `train`, `00000.wav` |
| WAV | `.../XACLE_Challenge_2027_development_dataset/wav/train/00000.wav` |
| Caption | `A water vehicle travels through the water with wind noise and a man talking` |
| Checkpoint | `weights/pretrained/CLAP_weights_2023.pth` |
| Crop | center 7 giây ở 44.100 Hz; file gốc dài khoảng 9,844 giây |
| Score | `0.6499422788619995`, shape `(1, 1)`, `float32` |
| Features | shape `(1, 2048)`, `float32` |
| Model state | eval, toàn bộ tham số frozen |

Hai lần suy luận lặp trên cùng cặp audio-caption cho cùng score `0.6499422788619995`; feature lặp giống nhau trong cùng cấu hình CPU. Kiểm tra inference thực tế bổ sung với hai mẫu cho thấy: đổi thứ tự batch không đổi kết quả sau khi đảo lại; tiến trình mới cho cùng kết quả; thay batch size tạo sai khác feature tối đa khoảng `1.4305115e-06`, nằm trong tolerance `rtol=1e-5, atol=2e-6`. Đây là tính ổn định số trong tolerance trên CPU, không phải khẳng định deterministic bitwise trên CUDA.

Lần nạp đầu trong môi trường này cần `NUMBA_CACHE_DIR=/tmp/moescore-numba-cache` vì Numba không tạo được JIT cache mặc định trong `site-packages`. Không cài lại package và không sửa nội dung package.

## Kiểm thử

Đã chạy bằng Python trong môi trường `nlp`:

| Lệnh/bộ test | Kết quả |
|---|---:|
| `test_top2_adapters.py` | 7/7 PASS |
| `test_kaggle_feature_extraction.py` | 4/4 PASS |
| `test_moe_feature_store.py` | 6/6 PASS |
| `test_moe09_per_expert_cache.py` | 18/18 PASS |
| `py_compile` các module/test đã sửa | PASS |
| Inference MS-CLAP bằng checkpoint thật | PASS |

Test mới bao gồm: giữ cache ID khi đăng ký expert khác; invalidation theo checkpoint/crop; manifest, thứ tự và assembly identity; batch size; sample ID/caption/WAV hash/shape; ghi shard nguyên tử và phát hiện corruption; cache Hugging Face; WAV path tin cậy, traversal, file thiếu, hash/order mismatch; crop short/exact/long và odd offset; giữ resample; batch/order/repeated/fresh-process; và worker truyền đúng WAV path.

## Tương thích và giới hạn

- Registry hiện tại và extraction CLI vẫn chỉ hỗ trợ profile bốn expert Top-1; chưa tuyên bố profile `full_9` có thể chạy. Các contract/cache primitive đã sẵn sàng để nối vào profile sau.
- Các shard legacy gắn global extraction identity không được tự động gán lại schema 2. Global identity cũ không chứng minh đầy đủ checkpoint bytes, preprocessing và code dependency của từng expert. Vì vậy không an toàn để migrate; cần tái trích xuất vào cache namespace mới. Legacy shard/store không bị xóa hoặc ghi đè.
- Store mới được version hóa theo assembly identity dưới `feature_cache/assemblies/...`; dùng `feature_store_root` mà extraction command in ra khi cấu hình reader. Các artifact cũ giữ nguyên.
- Không chạy extraction toàn dataset, train gate, hoặc chạy inference CUDA.
- Bước tiếp theo để có `full_9`: đăng ký các expert mới trong profile/factory riêng, tạo input contracts và checkpoint provenance tương ứng, rồi assemble theo đúng thứ tự chín expert. Các cache Top-1 chỉ được reuse khi expert identity của chúng còn khớp.

## Files

**Tạo mới:**

- `data/moe_expert_cache.py`
- `tests/test_moe09_per_expert_cache.py`
- `docs/Ex1/moe09_per_expert_cache_msclap_report.md`

**Đã sửa:**

- `model/expert_adapters/registry.py`
- `model/expert_adapters/top2_msclap_adapter.py`
- `scripts/extract_moe_feature_store.py`
- `tests/test_kaggle_feature_extraction.py`
- `tests/test_top2_adapters.py`

Không thay đổi Top-1 expert math, model weights, gate, trainer, feature-store schema cũ hay source repository EnsembleSVR-XACLE. Không commit/push.
