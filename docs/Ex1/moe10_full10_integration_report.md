# MOE-10 — Báo cáo tích hợp M2D-LoRA CORAL / Full10

Ngày cập nhật: 2026-11-20  
Nhánh: `feature/moe-10-m2d-lora-coral`  
Commit gốc: `64312c6adb524d85ac71a5b762e526e39b675572`

## 1. Trạng thái ban đầu và tài nguyên

Repository ở commit trên; các file đã track sạch trước khi bắt đầu. Notebook E31 do người dùng cung cấp tại `notebooks/Copy_of_e31_lora_colab.ipynb` là file untracked từ trước và được giữ nguyên. Không commit, push, upload checkpoint hoặc chạy job Kaggle tính phí.

Đã dùng `conda` environment `nlp`: Python 3.10.21, PyTorch `2.14.1+cu130`. `torch.cuda.is_available()` là `False`, số GPU là 0. Dataset được xác nhận tại:

`/home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset`

Preflight Full10 đạt `PASS`: train 7.500, validation 3.000, test 3.000; đủ checkpoint cho 10 expert; các dependency có phiên bản phù hợp (`msclap==1.3.4`); model text `roberta-base`, `bert-base-uncased`, `google-bert/bert-base-uncased` và `gpt2` có trong cache offline. Báo cáo preflight: `/tmp/moe10_full10_local/preflight/preflight.json`.

## 2. Xác minh checkpoint Expert 10

Checkpoint người dùng chỉ định là `weights/m2d-lora-Nam.pt` (14.447.361 byte, SHA-256 `d6edbec6b724be48a18f618c98a2e25d044f06108e8715d86b58c59e97e6cbdb`). Payload có đúng `head`, `lora`, `cfg`; toàn bộ cấu hình khớp đặc tả: `variant=audio_text`, seed 7, 5 epoch, patience 2, batch 16/eval batch 64, `lr_head=0.001`, `lr_lora=0.0005`, weight decay 0.01, rank 8/alpha 16, 4 block audio/2 layer text, workers 2, `run_logo=False`, `smoke=False`. Có 24 tensor float32 hữu hạn: 16 audio và 8 text. Head CORAL có đúng sáu tensor, với các lớp `3073→1024→256→10`.

Đã đối chiếu cách chèn LoRA và thứ tự serialization với notebook E31: block audio 8–11, mỗi block `attn.qkv` rồi `attn.proj`; tiếp layer BERT 10–11, mỗi layer `attention.self.query` rồi `value`; serialization theo traversal `model.parameters()` chỉ gồm tham số trainable. Không dùng loader E34; chỉ dùng các hàm LoRA/encode/feature tương thích sau khi xác minh ý nghĩa. Checkpoint được nạp bằng `torch.load(..., map_location="cpu", weights_only=True)`, head nạp `strict=True`; adapter kiểm tra key, config, shape, dtype, giá trị hữu hạn, thứ tự tham số, sau đó freeze và eval.

Backbone bắt buộc `weights/pretrained/checkpoint-30.pth` tồn tại, kích thước 1.694.677.655 byte, SHA-256 `238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1`. SHA trùng với giá trị đã ghim trong cấu hình Expert 3. PortableM2D khởi tạo thành công và báo `<All keys matched successfully>`; sample rate là 16 kHz. Không thay bằng checkpoint khác.

## 3. Adapter và hợp đồng đầu ra

`M2DLoRACoralAdapter` nhận `expert3_wavs [B,1,160000]` float32 cùng `captions_raw`. Adapter tái sử dụng đường preprocessing Expert 3, PortableM2D và text encoder đã có. Audio/text embedding được chuẩn hóa như E31; vector tương tác giữ đúng thứ tự `[audio, text, audio*text, abs(audio-text), sum(audio*text)]`, tổng 3073 chiều. Head tạo 10 logit; score là `(sigmoid(logits).sum(-1)-5)/5`, không dùng cosine để dự đoán.

Đã dựng adapter từ checkpoint thật và chạy forward CPU. Embedding audio/text đều `[B,768]`; đầu ra hữu hạn, float32, cùng device; shape score `[1,1]`, feature `[1,3073]`; tất cả tham số frozen và model ở eval. Không có fixture output số của notebook E31 để đo sai số số học trực tiếp; thứ tự tensor và công thức được đối chiếu từ notebook, còn forward checkpoint thật được kiểm tra độc lập.

## 4. Registry và cấu hình

Registry giữ nguyên `top1_4` và `full_9`, thêm `m2d_lora_coral` theo factory lazy và profile `full_10` theo thứ tự cố định:

`expert1, expert2, expert3, expert4, top2_m2d, top2_msclap, top2_laion, top2_mga, m2d_lora, m2d_lora_coral`.

Contract được kiểm tra: Full9 có 9 expert, width 27.399; Full10 có 10 expert, width 30.472, trong đó Expert 10 có 3.073 chiều. `configs/moe_full10.example.yaml` là cấu hình portable: checkpoint Expert 10 trỏ tương đối tới `weights/m2d-lora-Nam.pt`; người dùng có thể thay bằng đường dẫn tuyệt đối trong local override. Schema, thứ tự, checkpoint resolution và resource validation chấp nhận đúng 10 expert cho `full_10` mà không đổi contract Full9.

## 5. Extraction, cache và feature store

Extraction đăng ký nguồn code, checkpoint, preprocessing `_expert3_audio`, text model, dependency, feature width và cache provenance của Expert 10. Cache giữ identity theo expert, bao gồm checkpoint/source/preprocessing/dependency/device/sample identity; thêm profile không làm đổi identity Expert 1–9. Assembly Full10 có identity mới và schema-v2 manifest chứa profile, thứ tự, kích thước, checkpoint provenance, preprocessing, cache IDs, assembly/source manifest identity, sample order và hash shard.

Cache Full9 được tham chiếu ở báo cáo trước (`/tmp/moe09_4_smoke`) không còn trên máy tại thời điểm chạy. Vì vậy smoke lần này tái trích xuất đủ 10 expert, không tái sử dụng shard Full9: **0/9 cache Full9 tái sử dụng; 10 cache expert mới, 30 shard cho ba split**. Các test identity xác nhận cache ID của expert cũ không đổi khi thêm Expert 10, cache sai provenance/hash bị từ chối và Full9/Full10 có assembly identity khác. Không relabel cache và không sửa artifact Full9.

Smoke CPU thực tế hoàn tất với 4 mẫu mỗi split. Feature store:

`/tmp/moe10_full10_local/smoke/feature_cache/assemblies/full_10/787bd928e5e217e373b94768c688d7cc9ff0cc9dfa89e207e8ecb62559401ade`

Mỗi split có `scores [4,10]` và `features [4,30472]`, float32, hữu hạn; sample ordering được xác thực. Train/validation có nhãn; test không có nhãn. Cache Expert 10 có ID `fabc1d9d259877a9e4201b462a3c8b4257e5de064e298aea65cda6e3046f315d`.

## 6. Gate và checkpoint

GatingNetwork giữ nguyên kiến trúc MOESCORE: `Linear(30472,512) → LayerNorm → ReLU → Dropout(0.1) → Linear(512,10) → Softmax`. Fusion và chuẩn hóa target không đổi. Checkpoint schema-v2 ghi profile/order/dimensions/width, feature-store identity, checkpoint provenance, optimizer/scheduler, epoch/step và RNG state. Full10 từ chối checkpoint thiếu/sai resume state; Full9 schema cũ vẫn tải được theo profile của nó và không được load như Full10.

Trên feature store smoke thật, gate được train thử 1 epoch (2 optimizer steps), chỉ cập nhật tham số gate. Checkpoint được lưu tại `/tmp/moe10_full10_local/runs/smoke_gate/{best_gate.pt,latest_gate.pt}`. Validation inference đạt `PASS` với 4 mẫu; test inference đạt `PASS` với 4 mẫu, không cần nhãn. Chỉ số validation nhỏ này là smoke, không phải benchmark: SRCC 0,800; LCC 0,957; KTAU 0,667; MSE 2,835 trên thang 0–10, n=4.

## 7. Kaggle hai GPU

`scripts/kaggle_full9_dual_gpu.py` hiện nhận profile `full_9` hoặc `full_10`, phân công expert theo hai queue tuần tự `cuda:0`/`cuda:1`, cho hai queue chạy đồng thời, nhận batch size theo expert, resume shard hợp lệ và assemble theo contract profile. `scripts/kaggle_full9_artifacts.py` xác thực/package Full9 lẫn Full10. Pipeline config/text-cache và notebook mới `notebooks/MOESCORE_Full10_Kaggle.ipynb` dùng script chung, không tạo scheduler khác; notebook đặt Expert 10 batch size mặc định 1.

Notebook và scheduler đã được compile/unit-test; **chưa chạy trên Kaggle hoặc hai GPU thật** vì môi trường hiện có 0 GPU. Notebook trỏ `REPO_REF` tới nhánh feature này; để clone từ Kaggle, source branch phải được gắn sẵn hoặc được publish sau theo quyết định riêng. Không có remote job nào được khởi chạy.

## 8. Kiểm thử và trạng thái thực thi

| Giai đoạn | Kết quả |
|---|---|
| Unit/integration `tests/` | `381 passed, 1 skipped, 278 subtests passed` |
| Full9 regression/cache/config/store/checkpoint/Kaggle | Qua trong bộ `tests/`; `top1_4`/`full_9` giữ nguyên contract |
| Checkpoint Expert 10 thật | Nạp thật, PortableM2D strict restore và forward CPU thành công |
| Full10 extraction smoke | Hoàn tất 4 mẫu train/validation/test trên CPU, 10 expert |
| Gate smoke/checkpoint | 1 epoch, 2 bước; save/load và strict provenance thành công |
| Validation smoke | 4 mẫu; không dùng làm so sánh benchmark |
| Test inference smoke | 4 mẫu, không nhãn; thành công |
| Full extraction / full gate training / matched Full9-vs-Full10 benchmark | Chưa chạy |
| Kaggle hai GPU thật | Chưa chạy |

Lệnh chạy toàn bộ test lồng nhau (thêm parent vào `sys.path`) kết thúc `418 passed, 11 failed, 1 skipped, 282 subtests`. 11 lỗi nằm trong test/harness Expert 2/3/4 không thuộc các file thay đổi: thiếu alias import như `Train_Expert2`, `models`, `inference_expert3`, package `expert4_repro`, và đường dẫn notebook Expert 4 cũ `expert4_finetune/colab/Finetune_MOESCORE_Expert4.ipynb` không tồn tại trong checkout. Bộ `tests/` của MOESCORE — bao gồm hồi quy các profile liên quan — pass đầy đủ.

## 9. So sánh benchmark Full9 / Full10

Chưa có feature store/cache Full9 phù hợp và checkpoint gate Full9 được train/evaluate theo cùng split, cấu hình và tiêu chí chọn model. Vì thế không đưa số benchmark hoặc delta giả định:

| Model | SRCC | LCC | KTAU | MSE |
|---|---:|---:|---:|---:|
| Full9 | Chưa đo | Chưa đo | Chưa đo | Chưa đo |
| Full10 | Chưa đo trên validation đầy đủ | Chưa đo | Chưa đo | Chưa đo |
| Delta | Chưa tính | Chưa tính | Chưa tính | Chưa tính |

Số Full10 n=4 ở mục trên chỉ là smoke, không dùng làm hàng benchmark hoặc so sánh với Full9.

## 10. File đã tạo/sửa

**Tạo:**

- `model/expert_adapters/m2d_lora_coral_adapter.py`
- `model/expert_adapters/m2d_lora_coral_runtime.py`
- `configs/moe_full10.example.yaml`
- `notebooks/MOESCORE_Full10_Kaggle.ipynb`
- `tests/test_m2d_lora_coral_adapter.py`
- `tests/test_moe10_full10.py`
- `docs/Ex1/moe10_full10_integration_report.md`
- Các ghi chú implementation trong `.superpowers/sdd/2026-11-20-moe10-full10/`, `docs/superpowers/specs/` và `docs/superpowers/plans/`.

**Sửa:** `configs/moe_config.py`, `data/moe_feature_store.py`, `model/expert_adapters/registry.py`, `scripts/extract_moe_feature_store.py`, `scripts/kaggle_full9_artifacts.py`, `scripts/kaggle_full9_dual_gpu.py`, `scripts/kaggle_moe_pipeline.py`, `scripts/prepare_moe_text_cache.py`, `training/checkpoint_manager.py`, `tests/test_kaggle_full9_dual_gpu.py`, `tests/test_moe09_3_cache_identity.py`.

## 11. Lệnh tái lập

Trong Kaggle notebook, đặt Inputs (dataset, weights, HF cache, UniLM), `EXPERT_PROFILE="full_10"`, hai GPU, và chạy stage smoke trước. Lệnh dual-GPU tương đương:

```bash
python scripts/kaggle_full9_dual_gpu.py \
  --profile full_10 \
  --config /kaggle/working/moescore/configs/moe_expert_config.yaml \
  --dataset-root /kaggle/input/<xacle-root> \
  --output-root /kaggle/working/moescore/extraction \
  --worker-devices-json /kaggle/working/moescore/configs/worker_devices.json \
  --batch-sizes 1,1,1,1,1,1,1,1,1,1 --shard-size 128 --smoke
```

Bỏ `--smoke` để trích xuất đầy đủ; chạy lại cùng lệnh để xác thực và resume shard/store đúng identity. Gating YAML phải trỏ `data.feature_store_root` tới assembly path trong report extraction.

```bash
python scripts/train_moe_gating.py \
  --config /kaggle/working/moescore/configs/moe_gating_config.yaml \
  --device cuda:0 --report-json /kaggle/working/moescore/logs/training.json

# Resume từ checkpoint mới nhất, giữ nguyên feature-store identity
python scripts/train_moe_gating.py \
  --config /kaggle/working/moescore/configs/moe_gating_config.yaml \
  --resume /kaggle/working/moescore/runs/latest_gate.pt --device cuda:0

python scripts/evaluate_moe_gating.py \
  --config /kaggle/working/moescore/configs/moe_gating_config.yaml \
  --checkpoint /kaggle/working/moescore/runs/best_gate.pt --device cuda:0 \
  --report-json /kaggle/working/moescore/runs/validation_metrics.json

python scripts/infer_moe_gating.py \
  --config /kaggle/working/moescore/configs/moe_gating_config.yaml \
  --checkpoint /kaggle/working/moescore/runs/best_gate.pt --device cuda:0 \
  --output /kaggle/working/moescore/runs/submission.csv
```

## 12. Hạn chế và kết luận

Môi trường `nlp` cục bộ có thư viện Numba trong site-packages không ghi được cache mặc định; bounded smoke dùng `NUMBA_CACHE_DIR` tạm và HF cache root ghi được trong `/tmp`, symlink tới Hub cache cục bộ. Đây là cấu hình thực thi cục bộ, không sửa preprocessing hay phép tính expert. Smoke đủ để xác nhận pipeline trên dữ liệu thật và checkpoint thật, nhưng không thay thế chạy Kaggle 2 GPU.

Full10 **đã tích hợp xuyên suốt và sẵn sàng bắt đầu full gate training sau khi hoàn tất full feature extraction trên Kaggle**. Adapter thật, smoke extraction, checkpoint gate, validation inference và label-free test inference đều đã được kiểm tra. Chưa thể gọi là benchmark hoàn thành: chưa chạy extraction đầy đủ, huấn luyện gate đầy đủ, đánh giá validation đủ 3.000 mẫu hoặc so sánh Full9/Full10 có kiểm soát.
