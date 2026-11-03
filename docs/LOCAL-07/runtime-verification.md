# LOCAL-07.1-FIX — Báo cáo xác minh runtime Expert 4

## 1. Kết luận

**Trạng thái: PASS — INCLUDING REAL VALIDATION INFERENCE.**

Đã dựng kiến trúc Expert 4 gốc bằng BEATs và LAION-CLAP pretrained thật, nạp fine-tuned checkpoint qua loader gốc, xác nhận toàn bộ 103 tensor được lưu, rồi so standalone với Expert4Adapter trên synthetic và hai WAV validation. Score và fused features trùng khớp chính xác; chênh lệch lớn nhất bằng 0.

Không bỏ qua fingerprint, đổi checkpoint, sửa kiến trúc, precision, phép tính score hoặc attention mask.

## 2. Nguyên nhân lần chạy trước bị chặn

Checkpoint yêu cầu UniLM commit 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd và source SHA fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487. Lần trước dùng bundled source trong expert4_finetune/unilm; source này có identity khác nên loader gốc dừng trước dựng model.

Lần này dùng clone độc lập:

    /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original

Clone có identity chính xác; không sửa bundled source hoặc clone.

## 3. Môi trường

| Thành phần | Giá trị |
|---|---|
| Conda / Python | nlp / 3.10.21 |
| PyTorch / CUDA runtime | 2.14.1+cu130 / 13.0 |
| GPU | NVIDIA GeForce GTX 1650 |
| Transformers / LAION-CLAP | 5.17.0 / 1.1.7 |
| Thiết bị / precision | cuda:0 / FP32 |
| Offline variables | HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; HF_DATASETS_OFFLINE=1 |
| Seed / tolerance | 2027 / atol=1e-6, rtol=1e-5 |

Không dùng autocast, FP16/BF16, quantization, offloading, tải mạng hoặc training.

## 4. Tài nguyên

| Artifact | Đường dẫn | Kích thước |
|---|---|---:|
| BEATs pretrained | weights/BEATs_iter3_plus_AS2M.pt | 361,499,833 bytes |
| LAION-CLAP pretrained | weights/laion-clap.pt | 1,863,587,645 bytes |
| Expert 4 best | weights/finetuned_weights/expert4_best.pt | 155,455,153 bytes |
| Training config | expert4_finetune/expert4_colab.yaml | 732 bytes |
| UniLM source | /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original/beats | Có |

SHA fine-tuned checkpoint: 1d80c0c8ace8a6f76f2b632cb0becedc6f0016e4b880138f8b03b81f1db841be.

SHA các checkpoint/source không đổi sau verifier; size và mtime cũng giữ nguyên.

## 5. UniLM identity

Verifier gọi đúng expert4_finetune.trainer._source_identity() (trainer.py:140–160), sau đó so sánh identity đầy đủ với metadata fine-tuned checkpoint.

| Identity | Checkpoint yêu cầu | Clone quan sát | Kết quả |
|---|---|---|---|
| Git commit | 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd | 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd | Khớp |
| Source SHA-256 | fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487 | fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487 | Khớp |

SHA từng file:
- BEATs.py: 27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb
- backbone.py: 31c0378379a7e0f1d1069f9da444fb86890fe1ea078959a2dcd39640cdcadbaa
- modules.py: edeb6b6cd6a784da749f932c3e0783c0bce556fc768d0d23a4d53d4b819eb424

## 6. Cache và import BEATs

RobertaConfig/RobertaTokenizer của roberta-base hoạt động offline; tokenizer RobertaTokenizer, vocab 50,265, hidden size 768. Expert 4 tạo đúng RoBERTa architecture rồi nạp text branch/projection từ LAION-CLAP.

UniLM revision này dùng import tuyệt đối BEATs → backbone → modules. Bridge trong model/expert_adapters/expert4_adapter.py đã nạp các file vào private package, tạm alias ba tên top-level và kiểm tra origins. Factory thật xác nhận cả ba alias trỏ đúng clone; sau factory aliases và đường dẫn beats/ được khôi phục. Model tiếp tục chạy sau khi context đóng.

Test xác nhận import tuyệt đối/tương đối, hai import-order Expert2/Expert4, phát hiện origin xung đột, giữ module lạ, và cleanup trong exception path. Không sửa BEATs source.

LAION-CLAP tự thêm site-packages/laion_clap vào sys.path trong import; đây là side effect của dependency, không phải BEATs path. Coexistence factory thật với MGA-CLAP chưa được chạy.

## 7. Dựng model và pretrained restoration

Chuỗi thật: Expert4Adapter.from_finetuned → expert4_finetune.inference.load_finetuned → Expert4.from_pretrained (inference.py:50–59; model.py:8–24).

| Thành phần | Quan sát |
|---|---|
| Class | expert4_finetune.model.Expert4 |
| BEATs | Dựng thành công, 12 layers, original loader strict=True |
| BEATs state | 250 checkpoint/model keys; missing 0, unexpected 0, shape mismatch 0 |
| CLAP | Original text/projection loader thành công offline |
| SeqCoAttn / RegressionHead | Dựng thành công |

Sau fine-tuning overlay, 48 key BEATs được kiểm tra theo artifact fine-tuned; 202 key pretrained không bị fine-tune khớp chính xác base. CLAP có 36 key fine-tuned và 167 key pretrained không bị fine-tune khớp chính xác.

Checkpoint CLAP có thêm đúng buffer text_branch.embeddings.position_ids, shape [1,514], int64, không phải parameter. Loader gốc dùng strict=False và lọc missing text parameters; verifier chỉ cho phép buffer này, key lạ khác bị từ chối.

## 8. Fine-tuned partial checkpoint

Schema thực tế khớp writer gốc (trainer.py:323–324): model_state_dict, config, fingerprints, epoch, best_srcc. Epoch 5; best_srcc 0.5629903400800055.

| Kiểm tra | Kết quả |
|---|---:|
| Expected trainable keys / checkpoint keys | 103 / 103 |
| Missing / unexpected | 0 / 0 |
| Shape mismatch | 0 |
| Non-tensor / non-finite | 0 / 0 |
| Tensor đối chiếu sau loader gốc | 103 |
| Exact value matches | 103 |

Loader partial gốc dùng strict=False nhưng bắt chính xác tập parameter trainable trước khi load (trainer.py:120–129). Adapter factory và verifier kiểm tra lại giá trị sau nạp. Không có pretrained-only fallback. BEATs, CLAP và UniLM fingerprints khớp.

Checkpoint có train/validation CSV fingerprints; source loader inference không xác minh hai CSV fingerprints. Báo cáo không tuyên bố xác minh training CSV provenance.

## 9. Runtime output và dimensions

Đo được audio-query branch 512, text-query branch 512, fused/head width 1024, adapter.feature_dim 1024. Fused tensor runtime là [B,1024].

SeqCoAttn gốc nối pooled audio-query attends-to-text trước rồi pooled text-query attends-to-audio sau (seq_coattn.py:14–31). Standalone trả score [B] sau tanh gốc; adapter chỉ thêm chiều thành [B,1]. Không có tanh lần hai, clamp, rescale hay normalize bổ sung.

## 10. Synthetic parity và masks

Synthetic waveforms FP32 [B,1,160000], seed 2027:

| Trường hợp | Lengths | Standalone score | Score diff max | Feature shape / diff max |
|---|---|---:|---:|---|
| B=1 đầy đủ | [160000] | 0.0152914133 | 0.0 | [1,1024] / 0.0 |
| B=1 ngắn | [80000] | -0.1217914820 | 0.0 | [1,1024] / 0.0 |
| B=2 trộn | [80000,160000] | [-0.1002327949,-0.0908028632] | 0.0 | [2,1024] / 0.0 |

Mọi score/feature đều allclose với atol=1e-6, rtol=1e-5; max/mean absolute và max relative differences đều 0.0. Score shape standalone [B], adapter [B,1]. Output float32, cuda:0, finite.

Probe trên model gốc xác nhận wav_lens đi nguyên vẹn vào AudioEncoder. Với B=2 mixed:
- Waveform padding counts: [80000,0].
- BEATs mask reduction 1: shape [2,998], padding counts [498,0].
- Reduction 2 / final audio mask: shape [2,496], padding counts [246,0], valid counts [250,496].
- Encoder masks giữ nguyên khi truyền vào SeqCoAttn; MHA key_padding_masks khớp nguồn tương ứng.
- Mask là bool, padding nằm cuối sequence.

Hai mẫu validation thật có length đầy đủ; kiểm thử mixed length được thực hiện bằng synthetic B=2.

## 11. Native validation

Dùng Expert4Dataset và collate_expert4 gốc từ expert4_finetune/dataset.py. Cả hai WAV mono, 16 kHz, 160,000 frames; sau preprocessing valid lengths [160000,160000], batch [2,1,160000] float32.

| WAV | Standalone score | Adapter score | Score diff | Feature diff max |
|---|---:|---:|---:|---:|
| 07500.wav | 0.7870628834 | 0.7870628834 | 0.0 | 0.0 |
| 07501.wav | 0.1560858041 | 0.1560858041 | 0.0 | 0.0 |

Feature shape [2,1024]. Dataset gốc được gọi, nhưng hai mẫu cùng 16 kHz nên không kích hoạt resampling. Không tính SRCC hoặc suy luận chất lượng dự đoán từ hai mẫu.

## 12. Single-pass và frozen/eval

Mỗi lần forward được instrument riêng: Expert4 1; AudioEncoder 1; TextEncoder 1; SeqCoAttn 1; mỗi cross-attention 1; pool 2; head 1.

Tổng 220,212,605 parameters; trainable expert parameters 0; adapter-only trainable 0. Tất cả module vẫn eval sau adapter.train() và adapter.train(False). Lặp lại B=1 cho score/features max diff 0.0.

## 13. GPU memory

| Số đo | Giá trị |
|---|---:|
| VRAM tổng | 3,895,132,160 bytes |
| Free trước model | 3,734,241,280 bytes |
| Peak allocated | 1,067,779,584 bytes |
| Free sau verification | 2,448,687,104 bytes |
| Batch lớn nhất | B=2 |

Không OOM; chỉ dựng một Expert 4 instance cho cả hai đường inference.

## 14. Kiểm thử hồi quy

Chạy trong Conda nlp:
- test_expert4_adapter.py: 25/25 PASS.
- test_expert4_runtime_verification.py: 4/4 PASS.
- Toàn bộ tests/test_*.py: 136 tests, 136 PASS, 0 FAIL, 0 SKIP (97.99 giây).
- py_compile scripts/verify_expert4_runtime.py: PASS.

Có các cảnh báo deprecation từ torch weight_norm, timm.models.layers, torch.meshgrid; không có test failure.

## 15. Lệnh runtime

    conda run -n nlp --no-capture-output python3 scripts/verify_expert4_runtime.py \
      --beats-checkpoint weights/BEATs_iter3_plus_AS2M.pt \
      --clap-checkpoint weights/laion-clap.pt \
      --finetuned-checkpoint weights/finetuned_weights/expert4_best.pt \
      --unilm-source-dir /home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original \
      --training-config expert4_finetune/expert4_colab.yaml \
      --device auto --offline \
      --validation-csv /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/meta_data/validation_average.csv \
      --audio-root /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/wav/validation \
      --num-real-samples 2 --seed 2027 \
      --report-json /tmp/moescore_expert4_fix_runtime.json

Verifier exit code 0; final JSON status PASS — INCLUDING REAL VALIDATION INFERENCE.

## 16. Diễn tiến lỗi và sửa

Runtime ban đầu chỉ so toàn bộ BEATs state sau khi fine-tuned overlay đã nạp, vì thế các tensor trainable hợp lệ bị so nhầm với base. CLAP comparison cũng cần phân biệt buffer position_ids và key mismatch thật. Cuối cùng, lần chạy real validation đầu tiên dùng probe đã đóng sau synthetic phase. Verifier đã được sửa để so base keys không fine-tune, kiểm tra 103 fine-tuned values riêng, cho phép đúng position_ids buffer tương thích, và tạo probe riêng cho real subset.

Các sửa đổi chỉ thuộc bridge và verifier; không đổi model math hoặc checkpoint. Các test mới xác nhận quy tắc overlay và nguồn import.

## 17. Files đã tạo/cập nhật

- model/expert_adapters/expert4_adapter.py — BEATs private import bridge cho absolute imports, origin checks và cleanup.
- tests/test_expert4_adapter.py — import order/origin/cleanup tests.
- scripts/verify_expert4_runtime.py — hashes nguồn, tensor-overlay comparisons, factory origins, mask/parity và real-validation probe.
- tests/test_expert4_runtime_verification.py — bốn test cho logic so sánh checkpoint.
- docs/LOCAL-07/runtime-verification.md — báo cáo này.

Không sửa expert4_finetune/, expert khác, checkpoint, config model, gating/loss hoặc runtime verifier Expert 1–3. Không cài dependencies, tải mạng, train, commit hay push.

## 18. Giới hạn và readiness

- Real restoration, numerical parity và hai-mẫu native validation đều PASS.
- Full validation metrics/SRCC chưa chạy.
- Hai WAV thật cùng độ dài đầy đủ; mixed lengths được kiểm tra synthetic.
- Factory coexistence thật giữa Expert4 và MGA/Expert2 chưa được chạy; không tuyên bố đã chứng minh.
- LAION-CLAP import thêm site-packages/laion_clap vào sys.path; BEATs alias và BEATs source path được dọn đúng.
- Khởi tạo đồng thời bốn expert, VRAM joint load và end-to-end MoE training chưa kiểm thử.

Expert 4 đã sẵn sàng cho bước tích hợp factory nhiều expert/Unified MoE ở cấp adapter. Cần kiểm tra riêng MGA/Expert4 factory coexistence và VRAM khi dựng chung trước khi xác nhận toàn hệ thống.
