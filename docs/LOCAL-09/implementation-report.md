# LOCAL-09 — Báo cáo Unified Dataset và preprocessing theo expert

## 1. Tóm tắt

**PASS — UNIFIED DATASET IMPLEMENTATION VERIFIED.** Đã thêm Dataset và Collate nhẹ, đọc audio theo từng expert, giữ nguyên caption gốc và nhãn 0–10, giữ thứ tự CSV/batch và độ dài hợp lệ của Expert 4. Các test synthetic kiểm tra metadata, split, audio ngắn/đúng 10 giây/dài hơn 10 giây, stereo, resampling, crop/pad và các adapter hiện có.

Preflight trên CSV validation tại máy xác nhận header và thư mục split tồn tại. Chỉ đọc header CSV; không quét các bản ghi hay decode audio. **Real-WAV preprocessing parity chưa chạy và được dành cho LOCAL-09.1.** Không dựng model hoặc nạp checkpoint.

## 2. Phạm vi

Task này triển khai dữ liệu CPU độc lập với model. Không chạy inference, huấn luyện, tính metric hoặc thay đổi adapter/model. Synthetic WAV chỉ được tạo trong thư mục tạm để test phần mềm; không được đưa vào dữ liệu challenge.

## 3. Baseline repository

- Branch: `main`.
- Trước thay đổi, full suite đạt **159/159** test.
- Working tree đã có thay đổi và tệp untracked từ milestone trước, cùng `expert1_finetune/Train_Expert1.py` đang modified. Tệp Expert 1 có sẵn đó được giữ nguyên.
- Môi trường `nlp`: Python 3.10.21, PyTorch 2.14.1+cu130, TorchAudio 2.11.0+cu130, SciPy 1.15.3, soundfile 0.14.0, NumPy 1.26.4. CUDA khả dụng; task không cấp tensor/model lên GPU.

## 4. Schema XACLE

Header validation CSV thực tế được preflight đọc:

```text
wav_file_name,text,average_score,model_ID,model_name
```

Header này khác cách ghi trong tài liệu challenge ở tên điểm: nguồn thực tế dùng `average_score`; các native loader cũng có những alias `average_semantic_alignement_score` (giữ nguyên lỗi chính tả nguồn) và `average_semantic_alignment_score`. Dataset nhận diện trường theo tên, không theo vị trí, và từ chối trường bắt buộc bị thiếu hoặc alias bị mơ hồ.

`model_ID` được nhận diện không phân biệt hoa thường và trả ra thành metadata `model_id`; `model_name` cũng được giữ lại nếu có. Hai trường này không được truyền vào expert adapter hoặc dùng làm feature gating. Preflight không đọc nội dung từng dòng, nên chưa xác nhận nhãn, sample overlap hay mapping filename của toàn split.

## 5. Audit preprocessing gốc

Audit dựa trên các lớp/hàm nguồn bên dưới và adapter contract đã xác minh trong LOCAL-04–08.

| Expert | Dataset và decoder gốc | Resampling / mono | Audio ngắn | Đúng 10 giây | Audio dài | Caption ở nguồn gốc |
|---|---|---|---|---|---|---|
| E1 | `Expert1Dataset` trong `expert1_finetune/Train_Expert1.py`; `torchaudio.load` | Channel mean; `torchaudio.functional.resample` tới 48 kHz | Lặp `floor(target/L)` lần rồi zero-pad phải | Giữ nguyên | Train: random crop; eval: crop giữa | Giữ chuỗi thô; adapter E1 giữ nguyên |
| E2 | `Expert2Dataset` trong `expert2_finetune/expert2_data.py`; `scipy.io.wavfile.read` | Integer PCM đổi float32 và scale theo dtype; channel mean; `scipy.signal.resample_poly` theo GCD tới 32 kHz | Zero-pad phải | Giữ nguyên qua phép cắt `[:320000]` | Crop đầu tiên | Dataset gốc `.strip()`; adapter cũng strip ở biên gọi model |
| E3 | `MoescoreExpert3Dataset` trong `expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py`; `soundfile.read(dtype="float32", always_2d=True)` | Mean channel float32; `resample_poly` theo GCD tới 16 kHz | Zero-pad phải | Giữ nguyên | Crop đầu tiên | Dataset gốc `.strip()`; adapter strip trước original forward |
| E4 | `Expert4Dataset` trong `expert4_finetune/dataset.py`; `scipy.io.wavfile.read` | Integer PCM signed/unsigned theo xử lý riêng của nguồn; channel mean; `resample_poly` với `Fraction` tới 16 kHz | Item chưa pad; `collate_expert4` pad phải | Giữ nguyên | Crop đầu tiên | Dataset gốc strip; adapter strip raw caption trước model |

Các đích adapter đã xác nhận trong `model/expert_adapters/registry.py`: E1 `[B,1,480000]`, E2 `[B,1,320000]`, E3/E4 `[B,1,160000]`; riêng E4 có thêm `expert4_wav_lens` int64 `[B]`.

## 6. Unified Dataset

Đã thêm `data/unified_dataset.py` với `UnifiedXACLEDataset`.

- Constructor nhận CSV, thư mục audio của đúng split, `split` bắt buộc (`train`, `validation`, `test`) và seed.
- `from_dataset_root()` dùng layout `meta_data/{split}_average.csv`, `wav/{split}`.
- `from_config()` nhận config đã qua `resolve_moe_paths`; `test_csv` được cấu hình riêng, không trộn split.
- Rows giữ nguyên thứ tự CSV; mỗi item có `sample_id={split}:{filename}`, `wave_file_name`, `split`, `captions_raw` và 4 waveform.
- Audio decode lazy theo item, không cache toàn bộ dataset. Mỗi expert dùng decoder/conversion/resampler riêng theo nguồn; không có model/checkpoint import hoặc GPU allocation.
- Kiểm tra CSV, caption, nhãn, filename basename, duplicate filename, path traversal/symlink escape, file audio tồn tại/đọc được và audio finite. Audio corrupt được báo cùng filename/split.
- Nhãn được trả dưới dạng scalar CPU `float32` trên thang gốc `[0,10]`; không đưa về `[-1,1]` và không dịch nhãn. Train/validation mặc định cần nhãn; test có thể không có nhãn, không tạo nhãn giả.
- `audit_split_overlaps()` có thể so filename và cặp `(filename, caption)` chỉ từ metadata CSV; không decode audio. Chưa chạy overlap audit trên các split thật vì hiện preflight chỉ xác minh header validation.

### Audio cho từng nhánh

- E1 tái tạo policy train/eval. Random crop train dùng RNG cục bộ xác định bởi `(seed, epoch, index)`; không thay đổi global RNG. `set_epoch()` cho phép caller đặt epoch. Crop vẫn uniform trong khoảng hợp lệ; kết quả không phụ thuộc thứ tự truy cập item.
- E2/E3 crop từ đầu và zero-pad phải tới độ dài cố định.
- E4 crop tối đa 160000 mẫu, ghi `expert4_wav_lens` trước khi pad, rồi pad phải tới 160000. Vì mọi item của unified batch có width cố định, bước pad trong item tương đương giá trị với `collate_expert4` gốc.
- Không tính attention mask trong Dataset; BEATs/SeqCoAttn tiếp tục tự xử lý length theo code gốc.

## 7. Unified Collate

Đã thêm `data/unified_collate.py:collate_unified(samples)`. Collate chỉ kiểm tra và stack; không resample, crop, pad waveform, strip caption, shuffle hoặc chuyển device.

```text
expert1_wavs       float32 CPU [B,1,480000]
expert2_wavs       float32 CPU [B,1,320000]
expert3_wavs       float32 CPU [B,1,160000]
expert4_wavs       float32 CPU [B,1,160000]
expert4_wav_lens   int64 CPU   [B]
captions_raw       list[str]  (raw, giữ whitespace và thứ tự)
labels             float32 CPU [B] (chỉ khi tất cả item có label)
wave_file_names    list[str]
sample_ids         list[str]
split              một split duy nhất
model_ids/names    chỉ có nếu trường tương ứng có mặt ở mọi item
```

Batch rỗng, tensor thiếu/sai shape/dtype/device, giá trị không finite, valid length ngoài `[1,160000]`, caption sai, batch trộn split hoặc trộn item có/không có label/metadata đều bị từ chối. Không có nhãn hoặc metadata được tự tạo.

## 8. Caption

Dataset giữ nguyên chuỗi CSV trong `captions_raw`, kể cả whitespace ở đầu/cuối. Adapter chịu trách nhiệm normalization gốc: E1 giữ nguyên; E2/E3/E4 strip theo source-compatible adapter boundary. Dataset chỉ từ chối chuỗi không phải text hoặc rỗng sau khi kiểm tra whitespace; không lower-case, tokenize, cắt câu hay Unicode-normalize.

## 9. Nhãn và split integrity

Chỉ một split được mở cho mỗi Dataset. Không có API nối train/validation/test; caller phải truyền split cụ thể. `test_csv` là trường optional backward-compatible trong `configs/moe_config.py`; `train_csv`, `validation_csv`, `audio_root` vẫn bắt buộc. Config mẫu có path `metadata/test.csv`. Nếu trường test CSV không cấu hình, `from_config(..., split="test")` báo lỗi rõ ràng.

Nhãn score phải hữu hạn trong `[0,10]`; không scale hoặc shift. Validation/test vẫn mang nhãn khi CSV có chúng để phục vụ đánh giá sau này, nhưng LOCAL-09 không có trainer nên không dùng các nhãn đó để huấn luyện. Model-generation metadata chỉ dùng cho trace/audit.

## 10. Adapter compatibility nhẹ

`tests/test_unified_integration.py` dùng chính bốn adapter hiện có với original-model stubs. B=1, B=2, B=4 được kiểm tra. Test xác nhận E1 nhận waveform 48 kHz và caption không strip; E2/E3 nhận đúng waveform/caption đã strip; E4 nhận đúng waveform cùng lengths `[80000,160000]`. Mỗi original stub được gọi một lần. Bốn adapters đều frozen/eval; output checker có thứ tự registry tạo scores `[B,4]` và features `[B,7936]`. Đây là kiểm tra interface với stub, **không phải inference bốn expert thật**.

## 11. Unit tests synthetic

`test_unified*.py` đạt **35/35** test. Bao phủ:

- Header/alias/order, caption thô, nhãn gốc, optional metadata, test không nhãn.
- Split rõ ràng, config/root paths, filename duplicate/path traversal/missing file.
- Audio short/exact/long; E1 repeat/crop, E2/E3/E4 crop/pad, stereo và nguồn 8 kHz; E4 length trước pad.
- WAV hỏng, waveform NaN, label sai, caption rỗng, overlap audit.
- Collate B=1/2/4, order/value, CPU, shape/dtype/finite checks, lengths, optional labels/metadata.
- Adapter integration với stubs và output assembly.
- JSON preflight status và đảm bảo real parity vẫn `NOT_VERIFIED_LOCAL_09_1`.

Thêm một test tương thích config cho optional `test_csv`; toàn bộ `tests/test_moe_config.py` đạt 22/22.

## 12. Dataset diagnostic

Đã thêm `scripts/check_unified_dataset.py`. Chạy:

```bash
conda run -n nlp --no-capture-output python3 scripts/check_unified_dataset.py \
  --config configs/moe_local.example.yaml \
  --split validation \
  --metadata-csv /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/meta_data/validation_average.csv \
  --audio-dir /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/wav/validation \
  --offline \
  --report-json /tmp/moescore_local09_preflight.json
```

Kết quả: `status=PASS`, header yêu cầu PASS, metadata và audio directory tồn tại; script chỉ đọc header, không đọc/scan rows hay decode audio. Synthetic B=1/2/4 đều PASS; report ghi `model_construction=false`, `checkpoint_deserialization=false`, `real_waveform_parity=NOT_VERIFIED_LOCAL_09_1`, `real_caption_parity=NOT_VERIFIED_LOCAL_09_1`. JSON được tạo ở `/tmp/moescore_local09_preflight.json`.

## 13. Regression tests

- Baseline trước thay đổi: 159/159 PASS.
- Focused unified suite: 35/35 PASS.
- Config suite: 22/22 PASS.
- Full suite sau thay đổi: **195/195 PASS**, 0 failed, 0 skipped (`Ran 195 tests in 121.847s`).
- `py_compile` các module Dataset, Collate, preflight, tests và config: PASS.
- Full suite có một cảnh báo deprecation từ `timm.models.layers`; không có test failure.

## 14. Hiệu năng và bộ nhớ

Không khởi tạo pretrained model, checkpoint, CUDA context cho inference, Dataset-wide waveform cache hoặc worker pool. Mỗi `__getitem__` đọc một file qua bốn đường decoder độc lập để giữ source behavior; đây có thể là chi phí I/O/CPU đáng kể nhưng ưu tiên fidelity. Không chạy benchmark CPU/RAM hoặc full dataset scan trong LOCAL-09.

## 15. Rủi ro numerical parity còn lại

- Chưa so waveform unified với bốn native dataset trên WAV thật.
- Chưa xác minh codec/sample-rate thực tế cho toàn train/validation/test.
- Resampler/library implementation khác nhau giữa experts theo source; không thể thay bằng một resampler chung.
- E1 random crop train đổi cách rút số so với global `random` của native loader, nhưng giữ phân phối uniform và cô lập trạng thái RNG; parity train crop cần kiểm thử có kiểm soát ở LOCAL-09.1.
- Chưa chạy real caption/label alignment hoặc split-overlap scan trên full metadata.
- CSV validation thật được xác nhận qua header; không decode WAV và không đánh giá model.

## 16. Kế hoạch LOCAL-09.1

Trên một sample WAV được chọn từ mỗi split hiện có, gọi native dataset và Unified Dataset cho đúng filename/caption/label rồi so từng nhánh tensor trước collate. Dùng decoder/resampler gốc và so exact equality khi đường xử lý giống hệt; nếu khác, báo max absolute/relative difference và xác định bước gây khác trước khi chọn tolerance. Kiểm tra short, đúng 10 giây, long, stereo, non-target sample rate khi có sample hợp lệ trong dữ liệu đã được phép dùng; không tạo sample đánh giá challenge mới. Kiểm tra caption raw và caption sau normalization ở adapter boundary, label 0–10, B=1 và B=2 mixed-length E4; sau đó đối chiếu `collate_expert4` với unified collate. Không nạp checkpoint, inference model hoặc tính SRCC trong phần preprocessing parity.

## 17. Files tạo/thay đổi

- `data/unified_dataset.py` — Dataset, decode/transforms theo expert, split overlap và header helper.
- `data/unified_collate.py` — collate/validation batch CPU.
- `scripts/check_unified_dataset.py` — preflight nhẹ và JSON.
- `tests/test_unified_dataset.py`, `tests/test_unified_collate.py`, `tests/test_unified_integration.py`, `tests/test_unified_preflight.py`.
- `configs/moe_config.py`, `configs/moe_local.example.yaml` — thêm optional `test_csv`.
- `tests/test_moe_config.py` — kiểm tra config `test_csv` tương thích ngược.
- `docs/LOCAL-09/implementation-report.md` — báo cáo này.

Không sửa expert source, adapter, checkpoint, mô hình MoE, loss hoặc gating. Không nạp model, train, tải resource, cài package, stage, commit hay push.

## 18. Sẵn sàng cho LOCAL-09.1

Dataset/Collate API và adapter boundary đã sẵn sàng cho parity test trên real WAV. Dù vậy, trước khi đưa vào training pipeline, LOCAL-09.1 cần xác nhận giá trị waveform/caption/label với native loaders trên mẫu thực và xử lý mọi sai khác đã đo. Hiện không tuyên bố real preprocessing parity, real four-expert inference hoặc MoE training readiness.
