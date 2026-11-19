# MOE27-Core Full7: triển khai Core.1–Core.5

## Kiến trúc

Profile `full_7` có thứ tự `expert1` đến `expert7`, các độ rộng lần lượt
`2049, 4097, 3840, 1024, 11267, 4609, 3073`, tổng **29959**. E5 và E6
được dựng từ đầu ra đã lưu của Full9 và Whisper bằng adapter/head sẵn có. Nguồn
Full9 `m2d_lora` chỉ được đổi tên thành Full7 `expert7` tại bước lắp ghép.
Builder không gọi backbone. Gate dùng đúng `Linear(29959,512) → LayerNorm →
ReLU → Dropout(0.1) → Linear(512,7) → Softmax`; dự đoán là tổng có trọng số
của bảy score. Chỉ các tham số gate thuộc optimizer.

## Store và provenance

`MoeFeatureStoreWriter` dùng schema v3 riêng cho `full_7`; v1 `top1_4` và v2
`full_9` giữ nguyên. Manifest v3 ghi profile, thứ tự và độ rộng, score width 7,
dtype float32, nhãn raw 0–10 cho train/validation, không nhãn cho test, nguồn
Full9/Whisper và sample manifest, provenance checkpoint E1–E7, hợp đồng score,
hash hợp đồng lắp ghép, hash manifest và hash mỗi shard. Mỗi shard dùng ghi
atomic và gắn hash hợp đồng nguồn để resume không tái sử dụng shard từ đầu vào
khác. Đọc store sẽ kiểm lại manifest, shard và tensor.

Builder tái dùng `SetACachedDataset`, `SetBCachedDataset`, `load_set_a_checkpoint`
và `load_set_b_checkpoint`. Các lớp này kiểm tra source manifest, thứ tự ID và
tên WAV, SHA-256 WAV, caption/Whisper dataset identity, LAION extraction identity,
Whisper checkpoint và preprocessing identity, kiến trúc/normalization và nguồn
checkpoint E5/E6. CLI kiểm tra WAV trùng giữa mọi Full9 split hiện có trước khi
ghi artifact hoàn tất. E5/E6 checkpoint được đọc bằng loader hiện có với
`weights_only=False`: chỉ dùng checkpoint **nội bộ đã tin cậy**.
Builder kiểm tra mọi Full9 split hiện có, kể cả khi CLI chỉ yêu cầu build một
split. Full7 manifest giữ SHA-256 WAV/caption theo từng hàng và trạng thái
in-sample/out-of-sample của E5/E6; train kiểm tra WAV trùng với validation,
test inference kiểm tra WAV trùng với train/validation từ checkpoint. Checkpoint
v3 và run manifest ghi SHA-256 mã nguồn liên quan; run manifest ghi cờ Git dirty.

## Score và huấn luyện

Metadata score ghi nguồn, ngữ nghĩa, output scale, conversion `identity`,
fingerprint và trạng thái audit cho từng expert. E1/E2 giữ similarity gốc;
E3 dùng regression đã được huấn luyện trên `(raw-5)/5` (xem
`moescore_expert3_dataset.py`); E4 giữ tanh; E5/E6 giữ normalized residual;
E7 giữ chuyển đổi đã có trong M2D-LoRA. Không thêm scaler, clamp hay calibrator.
Contract thiếu, mâu thuẫn hoặc `UNVERIFIED` bị từ chối. Với cache thật, builder
đối chiếu score/feature contract, checkpoint provenance và SHA-256 của mã adapter,
source model, E3 training-target code với cache Full9. Artifact tổng hợp phải
khai báo `SYNTHETIC_VERIFIED`. Audit chỉ dùng train và ghi count, min/max,
mean/std, quantile, non-finite, tỷ lệ ngoài [-1,1] và sai số proxy với nhãn.
Giá trị ngoài [-1,1] là tín hiệu cần xem xét, không tự chứng minh sai scale.
Huấn luyện dữ liệu thật cần file review riêng, gắn với hash store, score contract
và audit, trước bước optimizer đầu tiên.

File review JSON có schema dưới đây; điền các hash từ output
`audit_full7_scores.py`, ghi người rà và bằng chứng cho từng expert rồi thêm
`data.score_review_file` vào bản sao cấu hình Full7:

```json
{
  "schema_version": 1,
  "train_store_identity_sha256": "<STORE_ID_FROM_AUDIT>",
  "score_contract_sha256": "<CONTRACT_HASH_FROM_AUDIT>",
  "score_audit_sha256": "<AUDIT_HASH_FROM_AUDIT>",
  "reviewer": "<REVIEWER_NAME>",
  "evidence": "<TRAIN_ONLY_SCORE_AND_CHECKPOINT_REVIEW_REFERENCE>",
  "experts": {
    "expert1": "REVIEWED_FOR_GATE_PROXY", "expert2": "REVIEWED_FOR_GATE_PROXY",
    "expert3": "REVIEWED_FOR_GATE_PROXY", "expert4": "REVIEWED_FOR_GATE_PROXY",
    "expert5": "REVIEWED_FOR_GATE_PROXY", "expert6": "REVIEWED_FOR_GATE_PROXY",
    "expert7": "REVIEWED_FOR_GATE_PROXY"
  }
}
```

Gate train dùng AdamW với learning rate `1e-5`, weight decay `1e-4`, batch 8,
tối đa 50 epoch, seed 2027; scheduler `ReduceLROnPlateau(min, factor=0.5,
patience=5)` theo validation regression loss. Loss giữ nguyên ClippedMSE +
0.5 Contrastive + 0.2 Diversity; `cbl=false`. Sampler có trạng thái và
`drop_last=True`; báo `train_rows_used` theo epoch. Chỉ official train tạo
gradient; validation báo SRCC/LCC/KTAU/MSE/MAE trên raw 0–10. Best checkpoint
chỉ thay khi SRCC tăng nghiêm ngặt. Schema gate v3 ghi kiến trúc, thứ tự,
provenance, fingerprint cấu hình/store, optimizer/scheduler/sampler/RNG và chính
sách chọn model. Inference so hợp đồng nguồn không phụ thuộc split, cho phép
sample hash của test khác train/validation.

## Lệnh CLI

Sao chép `configs/moe_full7.example.yaml`, sửa `data.feature_store_root` và
`output_dir` theo môi trường. `--source-audio-root` chứa thư mục mỗi split
(`train`, `validation`, `test`); WAV nằm trực tiếp trong đó hoặc dưới `wav/`.
Các đường dẫn dưới đây là placeholder:

```bash
python scripts/build_full7_feature_store.py \
  --full9-root <FULL9_STORE_ROOT> --whisper-root <WHISPER_CACHE_ROOT> \
  --e5-checkpoint <TRUSTED_E5_PT> --e6-checkpoint <TRUSTED_E6_PT> \
  --source-audio-root <OFFICIAL_WAV_ROOT_WITH_SPLIT_SUBDIRS> \
  --output-root <FULL7_STORE_ROOT> \
  --split train --split validation --split test --report-json <BUILD_REPORT_JSON>

python scripts/audit_full7_scores.py \
  --store-root <FULL7_STORE_ROOT> --output-json <SCORE_AUDIT_JSON>

python scripts/train_moe_gating.py --config <FULL7_CONFIG_YAML>
python scripts/train_moe_gating.py --config <FULL7_CONFIG_YAML> \
  --resume <RUN_DIR>/latest_gate.pt
python scripts/evaluate_moe_gating.py --config <FULL7_CONFIG_YAML> \
  --checkpoint <RUN_DIR>/best_gate.pt
python scripts/infer_moe_gating.py --config <FULL7_CONFIG_YAML> \
  --checkpoint <RUN_DIR>/best_gate.pt --output <SUBMISSION_CSV>
```

Full7 CSV chỉ gồm `wave_file_name,pred_score` và giữ thứ tự, làm tròn hai chữ
số như writer cũ. Writer cũ mặc định dùng `wav_file_name`; cần đối chiếu header
chính thức trước khi nộp vì hai cách viết xuất hiện trong mã/tài liệu nguồn.

## Giới hạn và Core.6

Đường chạy tổng hợp không xác nhận số học của checkpoint pretrained thật.
Chưa có bằng chứng về Full9/Whisper cache tương thích cho đủ ba split, E5/E6
checkpoint thật, alignment thật, score-scale thực nghiệm E3, huấn luyện gate
official hoặc metric validation thật. Core.6 cần các artifact đó, kiểm tra
SHA-256 và provenance từng nguồn, audit score train, xác minh official split,
rồi mới train/evaluate trên dữ liệu thật. Scheduler theo validation loss và
chọn best/early stop theo validation SRCC là chính sách V1 **tạm thời**; cần
chấp thuận của ban tổ chức. Tài nguyên pretrained bên ngoài cũng cần xác minh
phê duyệt và công bố dữ liệu huấn luyện trước khi nộp.
