# LOCAL-11 — Báo cáo triển khai pipeline huấn luyện MoE

## 1. Tóm tắt

**Kết quả: PASS — pipeline gate-only cho huấn luyện, validation, checkpoint và suy luận đã được kiểm chứng cục bộ bằng test tổng hợp và smoke trên đầu ra expert thật đã xác minh.** Store giữ nguyên score/feature FP32 và sample identity; trainer chỉ cập nhật gate; validation không cập nhật tham số; suy luận không cần nhãn.

Smoke thật dùng bốn đầu ra train LOCAL-10.1 đã có sẵn, batch 2 và hai bước optimizer trên CPU. Không khởi tạo chuyên gia hoặc nạp lại checkpoint expert trong LOCAL-11. Không chạy full train, validation hoặc test inference.

Các giới hạn quan trọng trước cloud đầy đủ: `configs/config.yaml` trong repository ghi hệ số Contrastive `+0.5`, còn yêu cầu LOCAL-11 chỉ định `-0.5`; feature store importer hiện nhận artifact LOCAL-10.1 nhưng chưa có extractor shard tổng quát đã kiểm thử cho full validation/test; khả năng biểu diễn weighted-sum trên bốn mẫu thật rất hạn chế. Các điểm này được giữ nguyên, không hiệu chỉnh score hay sửa loss gốc.

## 2. Repository và môi trường

- Branch: `main`; remote không đổi.
- Working tree trước task đã có thay đổi và tệp chưa theo dõi từ các milestone trước, gồm `M expert1_finetune/Train_Expert1.py`. Không xóa hoặc ghi đè chúng; không stage/commit/push.
- Baseline được báo cáo trước LOCAL-11: **228 test đạt**. Lệnh baseline trong phiên trả exit code 0; output summary ban đầu bị cắt. Lượt cuối ghi log đầy đủ có footer ở `/tmp/moescore_local11_full_regression.log`.
- Python `3.10.21`; Linux x86_64; PyTorch `2.14.1+cu130`; CUDA runtime `13.0`; NumPy `1.26.4`; SciPy `1.15.3`; SoundFile `0.14.0`; PyYAML `6.0.3`.
- CUDA khả dụng theo kiểm tra phiên, nhưng LOCAL-11 smoke chỉ dùng CPU và không tạo CUDA context.

## 3. Tệp LOCAL-11 tạo mới hoặc chỉnh sửa

### Mã và cấu hình

- `data/moe_feature_store.py` — writer, validator, shard NPY an toàn và reader mmap.
- `data/moe_training_dataset.py` — Dataset split rõ ràng và collate gate.
- `training/target_transform.py` — đổi nhãn raw/normalized và kiểm tra khoảng score lồi.
- `training/moe_trainer.py` — loss gốc, optimizer gate-only, diversity và train/validation step.
- `training/sampler.py` — thứ tự mẫu có trạng thái để resume.
- `training/checkpoint_manager.py` — checkpoint gate-only nguyên tử và nạp strict.
- `training/moe_evaluator.py` — metric validation, dự đoán test và CSV.
- `training/config.py`, `training/__init__.py` — cấu hình an toàn và package.
- `configs/moe_cloud.example.yaml` — cấu hình đường dẫn tương đối cho Colab/Kaggle.
- `scripts/build_moe_feature_store.py`, `scripts/validate_moe_feature_store.py`.
- `scripts/train_moe_gating.py`, `scripts/evaluate_moe_gating.py`, `scripts/infer_moe_gating.py`.
- `scripts/smoke_test_moe_training.py` — smoke có giới hạn, chỉ chấp nhận artifact LOCAL-10.1 được xác minh.

### Kiểm thử và tài liệu

- `tests/test_moe_feature_store.py`
- `tests/test_moe_training_pipeline.py`
- `tests/test_moe_checkpoint_resume.py`
- `tests/test_moe_validation_inference.py`
- `tests/test_moe_training_config.py`
- `tests/test_moe_cloud_entrypoints.py`
- `docs/LOCAL-11/training-architecture.md`
- `docs/LOCAL-11/cloud-execution-plan.md`
- `docs/LOCAL-11/implementation-report.md`
- `docs/superpowers/plans/2026-11-02-local11-portable-pipeline.md`

Không sửa adapter, kiến trúc expert, checkpoint expert, `loss/loss_function.py`, `Train_MOE.py`, `Inference_MOE.py`, Unified Dataset/Collate hay UniLM source clone.

## 4. Kiến trúc huấn luyện

Pipeline dùng `GatingNetwork` LOCAL-10, nhận `[B,4]` score và `[B,7936]` feature. Không có expert encoder trong trainer. Optimizer được kiểm tra chỉ chứa tham số gate; score/feature đã detach trước khi dùng. Phép tổng bốn hạng và Softmax của LOCAL-10 không đổi.

Optimizer mặc định AdamW, learning rate `1e-5`, weight decay `1e-4`. Scheduler là ReduceLROnPlateau `mode=min`, factor `0.5`, patience `5`, được bước một lần sau validation mỗi epoch. Cấu hình mẫu giữ batch train 16, validation 1, tối đa 50 epoch và patience dừng sớm 50. Smoke địa phương không dùng các giá trị cloud này làm một lượt train thật.

## 5. Feature store và dữ liệu thật

Store split-specific dùng manifest JSON và shard NPY; không dùng pickle. Shapes/dtypes được xác thực là `scores float32 [N,4]`, `features float32 [N,7936]`; nhãn raw `[N]` giữ scale `0–10`. ID phải có prefix đúng split và không trùng; shard, feature width, expert order, checkpoint provenance, preprocessing identity, finite values, dtype và hash được kiểm tra. Ghi shard nguyên tử và hỗ trợ tiếp tục shard chưa hoàn tất theo nội dung trùng khớp.

Các artifact LOCAL-10.1 được tìm thấy tại `/tmp/moescore_local101/`. Importer xác thực manifest SHA-256 `da37ac6af4ff578fa2dbde1a583208f26a5f905a76e32e185e5666101e78404d`, artifact hash, bốn báo cáo `PASS`, thứ tự ID, provenance checkpoint path/size/mtime và ánh xạ nguồn CSV/WAV/caption/nhãn. Import tạo store tại `/tmp/moescore_local11_feature_store/` với identity `4a05861e216e043a7b2942bbd1f7df5b0dcd0d7df554119e3238ae4db9597bfa`:

```text
sample count = 4
scores       = float32 [4,4]
features     = float32 [4,7936]
shards       = 2
```

Provenance này xác minh bằng báo cáo runtime, hash numeric artifact và metadata checkpoint path/size/mtime; không phải SHA-256 nội dung checkpoint. Trong quá trình import và smoke, checkpoint expert không được nạp lại.

Sample IDs theo đúng thứ tự: `train:04667.wav`, `train:03824.wav`, `train:01374.wav`, `train:06092.wav`. Nhãn raw lần lượt `4.25`, `5.25`, `9.0`, `5.75`. Smoke không đọc lại hoặc xử lý lại âm thanh; các đầu ra là artifact thật của LOCAL-10.1.

## 6. Chuẩn hóa nhãn và hạng loss

Nhãn gốc được đổi duy nhất ở ranh giới trainer bằng `(raw-5)/5`; batch phải khai báo `labels_scale="raw_0_to_10"`. Inverse cho dự đoán là `normalized*5+5`, không clip. Feature store, expert score và gating không nhận nhãn đã đổi làm đầu vào.

Gate trả `[B,1]`, nhưng loss hồi quy/Contrastive dùng vector `[B]`. Trainer chỉ reshape prediction ở ngay ranh giới loss để tránh Contrastive broadcast `[B,B,1]` với ma trận nhãn `[B,B]` thành `[B,B,B]`. B=1/2/4, scalar loss và ma trận cặp được kiểm thử.

Các công thức Clipped MSE, Contrastive và entropy gốc không bị sửa. Kiểm thử xác nhận miền Clipped MSE trong `tau` cho loss 0 và miền ngoài ngưỡng cho loss khác 0. Cấu hình CBL hiện `false`; khi bật mà thiếu `num_class`, trainer báo lỗi. Khi có class metadata, nhánh ClippedMSE gốc vẫn broadcast trọng số `[B,1]` với loss `[B]`; pipeline từ chối cấu hình đó để không chấp nhận kết quả sai hình dạng/ngữ nghĩa. Không sửa loss gốc.

**Sai khác hệ số cần giải quyết:** `configs/config.yaml` hiện có `[1.0, +0.5]`; yêu cầu LOCAL-11 ghi `[1.0, -0.5]`. `configs/moe_cloud.example.yaml` theo giá trị `-0.5` mà yêu cầu nêu, còn trainer luôn dùng đúng config truyền vào. Không tuyên bố hai cấu hình giống nhau. Trước CLOUD-03 cần chốt hệ số chuẩn.

## 7. Kiểm tra khả năng biểu diễn score

Với weights Softmax không âm có tổng 1, dự đoán không thể ra ngoài khoảng min/max score bốn expert trên cùng mẫu. Bốn mẫu thật có:

| Sample ID | Nhãn raw | Target chuẩn hóa | Min score expert | Max score expert | Khoảng cách ngoài khoảng |
|---|---:|---:|---:|---:|---:|
| `train:04667.wav` | 4.25 | -0.15 | 0.558580 | 0.818856 | 0.708580 |
| `train:03824.wav` | 5.25 | 0.05 | 0.274492 | 0.627415 | 0.224492 |
| `train:01374.wav` | 9.00 | 0.80 | 0.604385 | 0.692670 | 0.107330 |
| `train:06092.wav` | 5.75 | 0.15 | 0.585162 | 0.760897 | 0.435162 |

Trên đúng bốn mẫu này: `4/4` target ngoài khoảng, `3` dưới min, `1` trên max; khoảng cách trung bình `0.3688909`, lớn nhất `0.7085800`, MSE lower bound trung bình `0.1883419`. Đây là diagnostics của subset nhỏ, không khái quát hóa toàn train và không phải đánh giá hiệu năng. Không hiệu chỉnh score hoặc thay gating.

## 8. Smoke huấn luyện trên đầu ra thật

Lệnh đã chạy:

```bash
conda run -n nlp --no-capture-output python3 \
  scripts/smoke_test_moe_training.py \
  --feature-store /tmp/moescore_local11_feature_store \
  --artifact-dir /tmp/moescore_local101 \
  --dataset-root /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset \
  --device cpu --batch-size 2 --max-steps 3 --seed 2027 \
  --report-json /tmp/moescore_local11_smoke.json
```

Kết quả:

- Nguồn: `REAL_VERIFIED_OUTPUTS`; 4 mẫu train; 2 optimizer steps thực tế vì B=2.
- Dự đoán `[2,1]`, target `[2]`; Clipped MSE, Contrastive, regression, entropy, total loss và gradient hữu hạn.
- Bước 1: clipped `0.37576869`, contrastive `0.08955628`, regression `0.33099055`, diversity `0.03692824`, total `0.33837619`, gradient norm `2.74572158`.
- Bước 2: clipped `0.12637694`, contrastive `0.13449216`, regression `0.05913086`, diversity `0.03281796`, total `0.06569445`, gradient norm `0.78514659`.
- 6 tensor tham số gate thay đổi; score/features/nhãn đầu vào giữ nguyên.
- Checkpoint gate-only tạm được lưu và nạp lại strict; gate, optimizer, scheduler, step count và weights eval khớp. Khả năng tiếp tục chính xác còn được so sánh ba bước với và không resume trong unit test.
- Expert model construction `false`; expert checkpoint deserialization `false`; CUDA `false`.
- Peak RSS tiến trình khoảng `950,681,600` byte (`0.89 GiB`), bao gồm runtime PyTorch. Đây không phải mức RAM của expert và không dùng để ước lượng memory GPU.

Smoke là kiểm tra thực thi loss/gradient/update, không phải huấn luyện cạnh tranh, convergence hay đánh giá chất lượng.

## 9. Checkpoint, resume, validation và inference

Checkpoint gate-only gồm state dict, optimizer/scheduler, epoch, global step, cấu hình/fingerprint, train/validation store identity, expert provenance, sampler order/cursor, seed và RNG. Ghi atomic; nạp `weights_only=True`, strict, reject legacy 7168 input. Unit test xác nhận optimizer/scheduler/global step, mismatch provenance/config/store bị từ chối và resume khớp loss/parameter với chạy liên tục.

Trainer kiểm tra split `train`, không nhận gradient expert và không đưa caption/model ID/label vào gate. Evaluator yêu cầu `validation`, eval/no-grad, không gọi optimizer, tính SRCC/MSE/MAE ở thang 0–10 và normalized loss khi cấu hình. SRCC không xác định được trả `null` có nhãn trạng thái. Inference yêu cầu split `test`, không yêu cầu nhãn và ghi CSV theo upstream `wav_file_name,pred_score`. Schema nộp 2027 chưa được xác minh.

Các entrypoint `run_training`, `evaluate_checkpoint`, `infer_checkpoint` đã được gọi bằng test phần mềm với store giả lập train/validation/test nhỏ; test test split không có nhãn. Đây không phải dữ liệu cuộc thi và không phải full validation/test.

## 10. Kiểm tra store, test và hồi quy

- `scripts/build_moe_feature_store.py`: import 4 output thật; `PASS`, không dựng model/không nạp checkpoint trong importer.
- `scripts/validate_moe_feature_store.py --split train --require-labels`: `PASS`, 4 mẫu, 2 shard, đúng `[4,4]`/`[4,7936]`.
- Smoke JSON: `/tmp/moescore_local11_smoke.json`.
- Import JSON: `/tmp/moescore_local11_feature_store_import.json`.
- Store validation JSON: `/tmp/moescore_local11_store_validation.json`.
- Test LOCAL-11 mới: **33** test.
- Bộ `test_moe*.py`: **71** test đạt.
- Python `py_compile` cho các module, script và test LOCAL-11: **PASS**.
- Full suite cuối: **261 test đạt**, 0 lỗi, 0 bỏ qua; `Ran 261 tests`, `OK`. Log: `/tmp/moescore_local11_full_regression.log`.

Lệnh full suite:

```bash
conda run -n nlp --no-capture-output python3 \
  -m unittest discover -s tests -p 'test_*.py' -v
```

## 11. Đánh giá bộ nhớ

Smoke chỉ dùng CPU; peak RSS được đo trực tiếp như trên. CUDA có sẵn nhưng không được khởi tạo cho smoke; không đo allocation/reservation VRAM. Store feature FP32 ước lượng khoảng 227 MiB train + 91 MiB validation + 91 MiB test, chưa kể score/nhãn/metadata. Reader mmap và loader xử lý theo shard/batch. Không chạy full extraction để benchmark bộ nhớ.

## 12. Khả năng mang lên Colab/Kaggle

YAML dùng đường dẫn tương đối, `device=auto`, seed và batch cấu hình được. Mã huấn luyện không import factory expert, không cột vào đường dẫn workstation. Checkpoint tiếp tục an toàn theo store/provenance identity. `docs/LOCAL-11/cloud-execution-plan.md` mô tả Colab, Kaggle, mount, lưu artifact và thứ tự CLOUD-01..04.

Chưa xác minh runtime cloud. CLOUD-02 cần extractor full split có shard/resume đã kiểm thử; importer hiện chỉ chứng minh tương thích với định dạng artifact LOCAL-10.1. Rule 7 về cache expert offline cần được làm rõ nếu chưa có phê duyệt. Hệ số loss cần quyết định trước train đầy đủ.

## 13. Giới hạn và mức sẵn sàng

- Không có full XACLE training, full validation/SRCC, full test inference hay cloud run.
- Không nạp expert trong smoke; chỉ dùng cache thật đã được LOCAL-10.1 kiểm chứng.
- Kết quả score feasibility chỉ có bốn mẫu và nêu một giới hạn biểu diễn đáng lưu ý.
- Cấu hình nguồn hiện mâu thuẫn về dấu Contrastive; không được bỏ qua trước khi huấn luyện cloud.
- CBL bị chặn an toàn do broadcast trong mã loss gốc; cấu hình hiện dùng `false`.
- Header CSV upstream đã có, nhưng chưa xác nhận submission schema chính thức 2027.
- Chưa có extractor shard tổng quát cho full train/validation/test.

**Sẵn sàng cho CLOUD-01 ở quy mô nhỏ**, dùng artifact/checkpoint đã xác minh và sau khi người chạy chọn rõ hệ số loss. **Chưa sẵn sàng cho CLOUD-02/03/04 quy mô đầy đủ** cho tới khi có extractor split tổng quát, thống nhất hệ số Contrastive, xác nhận Rule 7 và đối chiếu định dạng nộp.
