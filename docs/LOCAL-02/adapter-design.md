# LOCAL-02.5 — Thiết kế Expert Adapter hợp nhất và kế hoạch tích hợp MOESCORE

> Phạm vi: thiết kế tĩnh dựa trên working tree và bốn audit LOCAL-02. Không tạo adapter, không sửa production code, không nạp checkpoint.

### Mục lục

1. Tóm tắt điều hành
2. Đánh giá hệ thống hiện tại
3. Kiến trúc đề xuất
4. Giao diện Expert Adapter
5–8. Bốn Expert Adapter
9. Unified Dataset và batch contract
10. MOEModel refactor
11. Dynamic gating input width
12. Freeze/eval/gradient policy
13. Config và checkpoint management
14. Training và inference integration
15. Local/Colab compatibility
16. Test và verification strategy
17. Implementation roadmap
18. Decision register
19. Rủi ro và giả định
20. Source references

## 1. Tóm tắt điều hành

Đề xuất giữ nguyên bốn implementation fine-tuned và thêm bốn adapter mỏng dưới model/expert_adapters. Adapter dựng model gốc, restore bằng loader riêng, ánh xạ input chuyên biệt và trả cặp (score, features). Một UnifiedMOEDataset tạo bốn audio representations từ cùng row; không dùng chung waveform/token IDs khi chưa chứng minh parity. MOEModel dùng features theo thứ tự E1→E2→E3→E4 để tính gate per-sample và weighted score.

Source hiện tại:
- MOEModel import legacy classes từ Basic_comp.py, chỉ nạp pretrained state bằng generic loader, hardcode gate width 7168. (model/MOE_model.py L3–7,L28–89)
- Root dataset không tạo CLAP waveform riêng và có lỗi MGA khi source rate đúng 32 kHz. (data/dataset.py L34–87,L89–135)
- Width gate mới là 3072+2*D_CLAP+5*D_M2D; hai dimension động chưa runtime-verified. (LOCAL-02.4)
- Train_MOE gọi model.train() mỗi epoch; cần giữ expert descendants ở eval. (model/MOE_model.py L55–58; Train_MOE.py L108–110)
- Loss source hiện là clipped MSE + contrastive + 0.2 entropy term; paper nêu clipped MSE + contrastive, không nêu entropy. Không thay objective khi chưa có quyết định nghiên cứu riêng. (loss/loss_function.py L6–103; Train_MOE.py L21–26,L127–132; MOESCORE.pdf §2.3)
- Không thấy official XACLE 2027 rules trong repo; README/PDF mô tả 2026. Compliance 2027 cần tài liệu rules chính thức. (README.md; MOESCORE.pdf)

Thiết kế đủ để bắt đầu implementation khi có resources, nhưng checkpoint restore/parity chưa thể xác nhận. Không có fallback sang pretrained-only.

## 2. Đánh giá hệ thống hiện tại

| Vùng | Hiện trạng | Hệ quả |
|---|---|---|
| Model selection | MOE_model dùng Basic_comp experts | Refactor sang adapter factory; Basic_comp giữ legacy |
| Restore | Generic loader đọc checkpoint rồi load vào legacy model | Adapter gọi loader full/partial đúng expert |
| Gate | Linear width 7168 → 512 → 4 + softmax | Width tính từ feature_dim thực tế |
| Scores | Rank không đồng nhất; multiply trực tiếp | Chuẩn mọi score [B,1], aggregate theo expert axis |
| Dataset | Một waveform chính và MGA branch, thiếu 48 kHz CLAP branch | Bốn preprocessing branches, identity dùng chung |
| MGA root preprocessing | Resampler cấu hình 16→32; gọi sai với input rates khác; rate=32 không gán biến | Không tái dùng, source-specific resampling |
| Config | utils.load_config dùng json.load nhưng configs/config.yaml là YAML; Train_MOE cần audio_encoder trong khi file có audio_wrapper; model constructor đọc cfg attributes nhưng loader trả dict | Chuẩn hóa config parser/schema |
| Labels | Dataset trả scores_raw, trainer đọc real_scores | Thống nhất batch label contract |
| Expert mode | experts eval tại constructor; epoch gọi parent train() | Giữ experts eval sau mỗi parent train |
| Paths | config có absolute paths của máy khác | Resolve relative/config roots, fail-fast |

Các mismatch này được ghi nhận để đưa vào roadmap, không thay đổi expert architecture.

## 3. Kiến trúc đề xuất

~~~text
model/
  expert_adapters/
    __init__.py
    base.py
    expert1_adapter.py
    expert2_adapter.py
    expert3_adapter.py
    expert4_adapter.py
  Basic_comp.py                 # giữ legacy/reference
  MOE_model.py                  # refactor để dùng adapters
data/
  expert_preprocessing.py       # bốn nhánh source-compatible
  unified_moe_dataset.py        # row identity, tensors, collate
configs/
  moe_local.yaml                # đề xuất, chưa tạo
  moe_colab.yaml                # đề xuất, chưa tạo
tests/
  test_adapter_contracts.py
  test_unified_dataset.py
  test_moe_gradients.py
expert*_finetune/               # giữ nguyên source fine-tune
~~~

Trách nhiệm: config resolver xác thực path/resources; adapter factory dựng và restore model; adapter forward map input/output; dataset tạo per-expert representations; MOEModel quản lý order/freeze/gate; Train_MOE xử lý optimizer/loss/metrics. Data không import models; adapters không import Train_MOE; entrypoint assemble config→dataset/model→training. Không tạo generic plugin framework hoặc copy implementations vào Basic_comp.

Construction: parse/validate config → xác minh tất cả required resources → tạo mỗi expert đúng một lần → base restore → fine-tuned restore → validate identity → đọc feature_dim → device/eval/freeze → tạo gate theo tổng dims. Forward: mỗi adapter trả [B,1] + [B,D_i] → kiểm B/order/device/dtype/finite → concat features [B,D_gate] → weights [B,4] → stack score [B,4] → weighted sum [B,1].

Basic_comp.py không xóa hoặc sửa. MOE_model.py được chuyển sang adapter imports ở phase riêng; generic loader không dùng cho partial E3/E4.

### Approaches considered

1. **Chép/copy bốn model vào Basic_comp:** không chọn vì dễ lệch implementation, checkpoint key paths và preprocessing giữa fine-tuning và MOE.
2. **Thin adapters quanh source fine-tuned:** khuyến nghị. Giữ model/checkpoint loaders gốc; adapter chỉ map inputs, expose features/score và validate shapes. Đây là đường ít thay đổi nhất để bảo toàn parity.
3. **Precompute score/features cho gate:** chỉ là phương án tối ưu memory sau này. Có thể giữ sample-dependent gating nếu mỗi sample giữ feature riêng, nhưng cần cache fingerprints, parity tests và xác nhận rules; chưa chọn trước khi đo VRAM.

## 4. Giao diện Expert Adapter

Đề xuất base class nhỏ kế thừa nn.Module, không có model parameters mới, với property feature_dim và forward(batch) trả tuple score/features. Nếu abstract base gây thêm boilerplate, protocol/convention cũng đủ; bắt buộc feature_dim là int dương sau factory construction.

- score Tensor [B,1], prediction gốc, không rescale/clamp.
- features Tensor [B,D_i], representation gốc, không đổi thứ tự/normalization.
- Cùng B/order/device tương thích; width, rank hoặc finite-value sai thì lỗi rõ.
- Input validation ở từng adapter vì keys/audio shapes khác nhau; output validation ở helper chung hoặc MOE collection boundary.
- Sample identity/order được đảm bảo ở dataset/collate, không thể khôi phục trong adapter nếu đã sai.
- Adapter không tự move tensors; entrypoint/device helper làm một lần.
- Factory classmethod from_config/resources chịu trách nhiệm restore; __init__ nhận model đã dựng và metadata để tránh double initialization.
- Tất cả paths kiểm tra trước model initialization. Adapter không ready khi chỉ pretrained weights load.
- Exceptions nêu expert, file, checkpoint/config mismatch; không suppress.

Shapes là STATIC_VERIFIED trong LOCAL-02.4; base class/factory là PROPOSED.

## 5. Adapter Expert 1 — LAION-CLAP

**Model và restore:** dùng expert1_finetune.Train_Expert1.Expert1; giữ enable_fusion=False. Factory nhận CLAP pretrained checkpoint, fine-tuned best raw full state_dict, config và device. Dựng Expert1 để CLAP_Module.load_ckpt nạp base; sau đó load fine-tuned state strict=True, xác nhận keys/shapes, eval/freeze. Artifact chưa local nên strict restore chưa runtime verified.

**Input:** expert1_wavs [B,1,480000] → clap_wavs; caption_raw list giữ whitespace → caption_tokens (giá trị là strings). Giữ 48 kHz và deterministic validation crop giữa cho audio dài; audio ngắn repeat rồi zero-pad. Tokenizer vẫn ở LAION API.

**Single pass:** inner CLAPAudioTextEncoder.forward đã trả audio/text embeddings. Adapter gọi đúng method này một lần, tính dot product giống Expert1.forward và concat [audio,text]. Không gọi Expert1.forward rồi encode thêm lần nữa. Reshape score [B] sang [B,1], feature [B,2*D_CLAP]. Không thêm normalize, temperature, regression head hay calibration.

**Dimension/error/parity:** lấy D_CLAP từ loaded model config nếu expose embed_dim, rồi assert widths thực tế. Nếu không có metadata đáng tin, yêu cầu config đã được xác nhận; không đoán dimension/lazy-init gate. So standalone/adapter scores trên cùng input/eval, compare feature ordering/width, dùng hook/call count xác nhận single encoder invocation. LAION normalization/range và D numeric vẫn UNVERIFIED. Import training module không được thực thi main.

## 6. Adapter Expert 2 — MGA-CLAP

**Restore:** dùng expert2_model.Expert2, create_expert2, load_mga_state và ASE gốc. Config cần MGA source dir, architecture YAML đúng training run, base checkpoint, fine-tuned raw full state, freeze_policy đã dùng (config hiện chọn partial), device. Existing factory dựng ASE, load pretrained checkpoint["model"] strict qua position_ids guard, dựng Expert2; adapter sau đó strict-load full Expert2 state. Sau restore đóng băng toàn bộ parameters, không giữ partial fine-tuning policy để gating.

**Namespace:** create_expert2 prepends MGA source vào sys.path và import top-level models. Vendor dùng top-level models/tools; BEATs cũng dùng global module BEATs và kiểm source. Khuyến nghị process sạch: reject các module đã import từ source khác; gọi existing factory một lần; assert module __file__ dưới MGA root; giữ bindings/path ổn định. Không xóa sys.modules modules sau khi model đã giữ references, không copy vendor code. Nếu namespace collision thì fail rõ/restart; nếu không đủ thì thiết kế isolated loader riêng trước code.

**Input/output:** expert2_wavs [B,1,320000] → mga_wavs; caption đã strip như standalone → caption_tokens raw strings. Giữ 32 kHz, first 10 seconds/zero-pad; HTSAT, BERT tokenizer và MSC ở trong ASE. Expert2 đã trả cosine [B,1] và concat [audio_MSC,text_MSC] [B,2048] theo embed_size=1024; width lấy config, assert output. Test origin, base/full strict restoration, legacy position_ids, policy, score/features parity. Raw best không có base fingerprint theo source, cần trusted run manifest.

## 7. Adapter Expert 3 — M2D-CLAP

Factory dùng MoescoreExpert3, PortableM2D và specialized load_best_checkpoint. Config: M2D checkpoint path rõ ràng, expected SHA-256, fine-tuned best partial checkpoint, model config, concat_assignment released, writable cache/layout root, local text resources và device. Dựng đúng base/family → constructor suy D và tạo head 5D → load_best_checkpoint kiểm format/kind/base hash/config/key/shape → eval/freeze. Không dùng generic strict=False.

Checked-in config từng trỏ tới expert3_finetune/weights/checkpoint-30.pth trong khi inventory tìm thấy weights/checkpoint-30.pth ở root. Cần explicit override/path; không auto-search hoặc thay checkpoint. Layout/cache writable phải cấu hình ngoài source tree.

Map expert3_wavs [B,1,160000] thành wavs; captions strip theo standalone thành raw strings. Giữ mono 16 kHz, first 10s/pad; không tokenize/truncate ngoài model. Output score [B,1], cross_features [B,5D] nguyên vẹn. feature_dim đọc từ model.cross_feature_dim sau constructor. Fine-tuning config chọn released, block đầu feature là text rồi audio, kế là diff, product, repeated cosine; không reorder.

Fail-fast nếu base path/hash/config/assignment/keys/shape/resource cache sai. Tests: loader, identity mismatch, D/5D, ordering, standalone parity và offline cache. Checkpoint/runtime chưa sẵn sàng.

## 8. Adapter Expert 4 — BEATs + CLAP + SeqCoAttn

Dùng expert4_finetune.model.Expert4 và expert4_finetune.inference.load_finetuned(checkpoint,resources,device), không dựng model lần hai. Loader xác minh BEATs/CLAP SHA, UniLM identity, dựng Expert4.from_pretrained, nạp trainable-only keys; strict=False có chủ đích để frozen base giữ pretrained initialization. Missing/mismatch là lỗi; không fallback pretrained-only.

Resources: BEATs checkpoint, CLAP checkpoint, UniLM/BEATs source directory, best_expert4.pt. Map expert4_wavs [B,1,160000] → wavs; expert4_wav_lens [B] int64 samples trước padding → wav_lens; caption theo standalone whitespace → captions. Giữ scipy decode/PCM scale, mono, resample_poly 16k, first-10s crop, valid lengths và padding. Mask True là padding ở key-padding boundary. Không đổi behavior hiện hành: query padding có thể tham gia adaptive max pool.

Output gốc score [B], fused [B,1024]; adapter chỉ thêm chiều score và trả fused nguyên. feature_dim=2*seq_coattn.dim=1024. BEATs global import cần process sạch và origin check. Tests: fingerprint/key failures, eval/freeze, score reshape invariance, fused parity, lengths/masks và batch 1. Runtime chưa được xác minh.

## 9. Unified Dataset và batch contract

UnifiedMOEDataset đọc một manifest row rồi tạo bốn audio branches riêng. Module expert_preprocessing.py là đề xuất; không tái dùng root data/dataset.py vì code hiện tại không giữ preprocessing đúng cho mọi expert. Giữ đúng decoder, PCM scaling, mono, resampler, crop/pad và caption whitespace của standalone; parity-test các helper trước khi dùng chung.

| Field | Producer → consumer | Shape/dtype | Semantics |
|---|---|---|---|
| sample_ids | parser/collate → logging/model | list[str], B phần tử | Stable CSV row identity, không chỉ filename |
| captions_raw | parser → E1/debug | list[str] | Caption nguyên văn; không tokenize chung |
| E2/E3/E4 captions | adapter boundary | list[str] | Exact whitespace strip như standalone |
| expert1_wavs | E1 branch → E1 | [B,1,480000], float32 | 48 kHz mono, 10s; eval center crop; short repeat+zero-pad |
| expert2_wavs | E2 branch → E2 | [B,1,320000], float32 | scipy PCM conversion, 32 kHz mono, first 10s/zero-pad |
| expert3_wavs | E3 branch → E3 | [B,1,160000], float32 | soundfile float32, 16 kHz mono, first 10s/zero-pad |
| expert4_wavs | E4 branch/collate → E4 | [B,1,160000], float32 | 16 kHz; crop/pad policy riêng E4 |
| expert4_wav_lens | E4 collate → E4 | int64 [B] | Valid samples sau resample/crop, trước batch padding |
| scores_raw | parser → metrics | float32 [B] | Scale 0–10 |
| scores | parser/collate → objective | float32 [B] | (scores_raw-5)/5 |
| num_class | parser → current loss API | integer [B] | Giữ nếu configured loss cần; không tự tạo nhãn mới |

E1 giữ whitespace nguyên văn; E2/3/4 giữ policy strip trong standalone datasets. Tất cả fine-tuned forwards nhận raw strings; caption_tokens ở E1/E2 không phải IDs. Tokenization và frontend ở bên trong từng expert. Chỉ E4 cần lengths/masks.

### Identity và labels

Sample ID ổn định nên gồm split + relative path + source row identifier. Nếu không có row ID, thêm raw caption và duplicate ordinal vào deterministic digest; tránh absolute paths/Python hash(). Không dedupe theo filename, vì có thể có nhiều caption cùng WAV. Collate giữ cùng permutation cho mọi branch/caption/score/ID và assert sample IDs aligned.

CSV validation map rõ alias filename/text/score, reject ambiguous/missing files, NaN/Inf và score ngoài 0–10; không clip. Giữ scores_raw và normalized scores với tên riêng. Root Train_MOE hiện đọc real_scores trong khi collate trả scores_raw; cần một contract thống nhất ở implementation phase. E2 standalone từ chối duplicate filenames; unified parser nên giữ row-level semantics và test multi-caption cases.

### Preprocessing graph

~~~text
CSV row → sample_id, raw caption, score, audio path
  ├─ E1: torchaudio → mono → 48k → validation center crop / short repeat+pad
  ├─ E2: scipy PCM → mono → 32k resample_poly → first crop/zero-pad
  ├─ E3: soundfile float32 → mono → 16k resample_poly → first crop/zero-pad
  └─ E4: scipy PCM → mono → 16k resample_poly → first crop → valid length → collate pad
caption: raw for E1; exact standalone strip for E2/E3/E4 → model-internal tokenizers
score_raw → preserve → normalized target
~~~

HTSAT mel, M2D log-mel/patching, BEATs and tokenization remain inside experts. Initially do not share audio decodes/resamples; optimize only after parity. Four float32 waveforms total 1.12M samples/item, about 35.8 MB per batch of 8 before worker/pinned/GPU copies, excluding model weights and activations.

## 10. MOEModel refactor

Compose adapters in fixed order E1,E2,E3,E4; new path must not instantiate Basic_comp experts. Keep current gating network unless separately approved: Linear(D_gate,512), LayerNorm(512), ReLU, Dropout(0.1), Linear(512,4), Softmax across expert axis. Gate input is concatenated features, not expert scores.

~~~python
outputs = [expert(batch) for expert in experts]  # score [B,1], feature [B,D_i]
features = torch.cat([item.features for item in outputs], dim=1)
weights = gate(features)                         # [B,4]
scores = torch.cat([item.score for item in outputs], dim=1)  # [B,4]
prediction = (weights * scores).sum(dim=1, keepdim=True)    # [B,1]
~~~

Validate ranks before stack to prevent [B] times [B,1] broadcasting into [B,B]. Return (prediction, weights) for current trainer convention; diagnostics must not alter prediction. Expert order must also define gate columns, scores, config metadata and checkpoints.

Per-row softmax over input-dependent features is architecturally sample-dependent. It does not prove learned weights vary meaningfully. After training inspect row sums, per-expert distributions/variance/correlation and gate collapse on held-out samples. Do not replace MoE with averaging, voting, static weights or weight averaging.

## 11. Dynamic gating input width

- E1: D1=2*D_CLAP, from loaded LAION config and checked against embedding outputs.
- E2: D2=2*embed_size=2048 for active config; assert ASE outputs.
- E3: D3=model.cross_feature_dim=5*embedding_dim; available after construction without forward.
- E4: D4=2*seq_coattn.dim=1024 from two 512-wide pooled branches.
- D_gate=sum(adapter.feature_dim)=3072+2*D_CLAP+5*D_M2D.

Restore experts before creating ordinary gate Linear. Require each feature_dim positive, match config/model metadata and match first real batch; combined width equals gate.in_features. On mismatch report expert, expected/actual, model config and checkpoint identity. Do not use LazyLinear. D_CLAP/D_M2D numeric and total remain UNVERIFIED until actual resources construct.

Legacy width 7168 is hardcoded with no source derivation. A guessed legacy sum does not establish compatibility with current features. Set gate width only from adapter metadata.

## 12. Freeze/eval/gradient policy

After restoration, before optimizer:
1. Set every parameter in all adapters requires_grad=False.
2. Call adapter.eval() recursively.
3. Build optimizer from gating_network.parameters() only.
4. Run expert forward inside torch.no_grad(); call gate outside that context.
5. Keep experts eval after every parent train(mode).

Recommend MOEModel.train(mode) calls parent behavior, then reapplies eval() to experts while gate follows mode. A single explicit setter is alternative but less robust because current Train_MOE calls model.train() every epoch. Expert2 has custom train() for fine-tuning; after parent recursion, eval all descendants again.

Test parameter freezing, evaluation/dropout state, absence of expert grads, nonzero gate gradients on a nondegenerate loss, and unchanged expert parameters after optimizer step. requires_grad, eval and no_grad solve different problems. Do not replace no_grad with inference_mode without verifying gate backward through expert-generated inputs.

## 13. Config and checkpoint management

### Portable config proposal

Use validated YAML and local/Colab path overrides; no host-specific absolute paths in code. Resolve relative paths from config location and allow explicit MOESCORE_ROOT, MOESCORE_DATA_ROOT and MOESCORE_RESOURCE_ROOT overrides. Implement environment expansion explicitly; do not assume YAML supports it. Current json.load versus YAML config mismatch must be fixed in future config phase.

~~~yaml
schema_version: 1
runtime:
  device: auto
  offline: true
  cache_dir: resources/cache
data:
  train_csv: data/meta/train.csv
  validation_csv: data/meta/validation.csv
  audio_root: data/wav
experts:
  expert1:
    pretrained_checkpoint: resources/laion-clap.pt
    finetuned_checkpoint: resources/expert1_best.pt
    enable_fusion: false
  expert2:
    source_dir: expert2_finetune/MGA-CLAP
    architecture_yaml: expert2_finetune/MGA-CLAP/settings/inference_example.yaml
    training_config: expert2_finetune/expert2_finetune.yaml
    pretrained_checkpoint: resources/mga-clap.pt
    finetuned_checkpoint: resources/expert2_best.pt
    freeze_policy: partial
  expert3:
    source_dir: expert3_finetune/XACLE_2027_baselinemodel
    base_checkpoint: resources/checkpoint-30.pth
    expected_base_sha256: REQUIRED
    finetuned_checkpoint: resources/expert3_best_model.pt
    concat_assignment: released
    model_config: expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json
  expert4:
    beats_checkpoint: resources/BEATs_iter3_plus_AS2M.pt
    clap_checkpoint: resources/laion-clap.pt
    unilm_source_dir: expert4_finetune/unilm
    finetuned_checkpoint: resources/best_expert4.pt
gate:
  expert_order: [expert1, expert2, expert3, expert4]
  hidden_dim: 512
  dropout: 0.1
~~~

Sketch chưa tạo. Validate readable paths, expected identities/configs before model construction. Missing resource/hash mismatch fails; no download, auto-search or substitute. E1/E2 raw fine-tuned states lack built-in base fingerprints, so require trusted run manifest or document weaker identity assurance.

### Restoration ownership

| Expert | Sequence | Existing behavior to retain |
|---|---|---|
| E1 | CLAP pretrained → Expert1 fusion-off → full fine-tuned raw state | strict full keys/shapes |
| E2 | MGA source/config → ASE base checkpoint["model"] strict + position_ids guard → Expert2 → full fine-tuned state strict | create_expert2/load_mga_state |
| E3 | M2D base/hash → MoescoreExpert3 dynamic D/released → best partial checkpoint | load_best_checkpoint metadata/key/shape validations |
| E4 | BEATs+CLAP+UniLM identity → Expert4.from_pretrained → trainable-only best | inference.load_finetuned/load_trainable_state_dict |

Construct once. Do not mark ready until fine-tuned restore passes. Where supported, restore on CPU then move to chosen device. Never load E2 wrapper state into ASE; never use generic unchecked partial loads. weights_only=False artifacts must be trusted.

### Gate checkpoint metadata

Best inference artifact binds format version, gate state only, gate architecture, expert order, expert checkpoint/base identities, preprocessing/config fingerprint, per-expert/total dimensions, target transform, best epoch and validation metrics. It does not embed expert weights. Reject different expert order/resources/dims at inference.

Resume artifact additionally stores optimizer/scheduler, scaler/RNG if used, epoch/global step and early-stop state. Keep best and resume schemas separate. This is proposal, not current code.




## 14. Training và inference integration

### Training path

Future Train_MOE sequence: parse validated config → build aligned UnifiedMOEDataset and deterministic loaders → create/restore all adapters → derive gate width/create gate → freeze/eval experts → optimizer over gate only → call model.train() with expert mode guard → move batch tensors while preserving strings → forward → existing configured loss → backward/step gate → validation eval/no_grad → inverse target transform for raw-scale metrics → select best gate by existing validation SRCC unless approved otherwise.

Current source uses batch_size/config values from config, AdamW over gate parameters only, ReduceLROnPlateau, score objective from get_loss_function, then adds diversity_loss(weights)*0.2; validation reports raw score via pred*5+5 and saves best_model.pt on SRCC. Resume bundle includes model/optimizer/scheduler/best metric/patience/epoch, loaded strict=False. (Train_MOE.py L41–105,L108–132,L159–219)

Preserve current optimizer/lr/scheduler/metric/loss by default; fix only necessary interface/config incompatibilities. But the checked-in config is YAML while utils.load_config uses json.load, Train_MOE reads audio_encoder.sample_rate absent from current config, MOEModel expects attributes from a dict loader, and train loop reads real_scores absent from root collate. These prevent treating current script/config as a validated baseline. Root data changes need explicit implementation plan.

Research paper says clipped MSE plus margin contrastive with coefficients beta/gamma; current loss factory combines these, while trainer adds entropy penalty that pushes higher gate entropy. Treat removal/retuning of entropy as separate approval/research decision, not a mechanical integration change. No official 2027 compliance claim without current rules.

### Training checkpoint behavior

Inference-best artifact should store gate-only state plus expert identities/dimensions/order/config fingerprint and selection metric. Resume artifact adds optimizer/scheduler and RNG state. Bind current normalized label convention and preprocessing config. Reject loading gate weights alongside different adapter feature order or fine-tuned checkpoint IDs. Current root resume uses strict=False and can hide missing keys; future gate loader should validate exact gate keys/shapes and identity metadata.

### Inference

Flow: load config/resources → restore four fine-tuned experts and validate identities → derive dimensions → build gate → validate/restore gate best checkpoint → set all modules eval and experts frozen → unified deterministic preprocessing → preserve sample IDs/order → four adapter forwards → per-sample weights → [B,1] normalized prediction.

Adapters return normalized prediction space without transform. For current validation metrics convert to 0–10 exactly once as raw = normalized*5+5; do not scale both inside adapter and metrics. What official 2027 submission expects is UNVERIFIED pending rules. Do not use target labels at inference; do not group/reorder rows based on model_id. Batch size 1 must preserve [1,1] final score. Return weights as diagnostics only; weights do not affect score beyond the same aggregation.

## 15. Local and Google Colab compatibility

- Launch from repository root with package imports rooted there; no per-module sys.path hacks beyond existing source loaders. Keep source checkout intact.
- Keep separate local/Colab YAML overrides for data/resource roots; resolve paths through explicit environment variables and assert all paths exist.
- Store checkpoints/datasets on local storage or mounted Drive; do not bake /root/autodl paths into source.
- Pre-populate exact tokenizer/HF cache, set offline/local-only behavior where supported; fail if resources unavailable rather than auto-fetch from Hub.
- Run lightweight import/config/dataset-stub tests on CPU. Device selection checks CUDA availability; expert restoration tests target the configured device.
- Verify MGA models/tools and BEATs module origins in a fresh Colab runtime. Do not remove/rebind modules already in sys.modules.
- Pin compatible dependencies for torchaudio/scipy/transformers/vendor packages in a future lock/environment record; no dependency changes in this design task.

Pretrained inventory reports individual files of roughly 1.8 GB CLAP, 1.6 GB MGA, 1.6 GB M2D and 345 MB BEATs; loaded model memory is not equal to file size and Expert1/E4 may hold separate CLAP-derived modules. Batch-8 raw waveform inputs alone are approximately 35.8 MB before activations/copies. Keeping four models on GPU likely dominates memory; measure peak allocated/reserved VRAM after restore and forward before selecting batch size.

**Alternatives, not selected yet:**
- All experts resident on GPU: simplest/few transfers, highest VRAM; choose only if measured fit.
- Sequential evaluation with one model on GPU at a time: lowers model-weight peak if other models live on CPU, but device transfers/latency and floating-point differences require parity testing.
- Offline feature/score caching: may reduce training VRAM substantially because experts are frozen and deterministic in eval; cache key must bind sample ID, all model/checkpoint hashes, preprocessing/config and expert order. It preserves input-dependent gate if every sample retains its own features/scores and no averaging is introduced; it does change operational pipeline and needs rules review and train/val/test separation. Do not cache test labels or use validation/test for gate training.
No optimization is chosen before exact output parity and official rules review.

## 16. Test and verification strategy

No test requiring checkpoint loading/inference is run in LOCAL-02.5.

| Level | Resources / device | Assertions | Failure means |
|---|---|---|---|
| 1. Import/config | No large checkpoints; CPU | Adapter imports, no train side effects/cycles, clean-process module origins, parse/schema, path missing errors, expert order, no auto-download | Package/config bug or collision |
| 2. Dataset/collate | Synthetic small WAV/CSV; CPU | Sample IDs, multiple captions, exact target, 48/32/16 kHz shapes, mono/crop/pad, E4 valid lengths, captions/order, B=1/B>1 | Preprocessing/identity contract mismatch |
| 3. Adapter interface | Stub/injected lightweight models; CPU | Input key mapping, tuple structure, score [B,1], feature [B,D], rank-preserving score reshape, width/order/finite validation | Adapter contract bug; stub is not checkpoint proof |
| 4. Checkpoint restore | Real pretrained + fine-tuned files; CPU load where supported, target GPU as needed | Correct base identities; E1/E2 strict; E3 specialized partial loader; E4 fingerprints/exact trainable keys; shapes/config; eval/frozen | Missing/wrong/mismatched artifact |
| 5. Preprocessing/parity | Actual restored experts; usually GPU/CPU heavy | Same source samples, deterministic eval pipeline, score/feature allclose within prespecified tolerances; E3 released order; E4 lengths/masks | Input/model parity failure |
| 6. MoE/gradients | Restored experts, small batch; GPU preferred | Four outputs, cat width, weights [B,4], row sums≈1, scores [B,4], final [B,1], no [B,B], expert eval/no grad/unchanged, gate gradients nonzero | Gate dimension, broadcasting, mode or optimizer bug |
| 7. End-to-end smoke | Checkpoints + small train/val subset; GPU | Construct, loader, forward, loss/backward/step, validation, best/resume save/load, order and expert immutability | Pipeline integration failure |

Tolerance values must be chosen before parity runs and account for matching precision/device. Run B=1 and B>1. Gate dependence check after training should measure across multiple examples and detect near-constant weights; it is not a requirement that random untrained weights already be meaningful.

## 17. Phased implementation roadmap

Each phase is a separate review point. Checkpoint artifacts are not present locally, so phases can have stub/import tests but must not be called fully accepted until real restoration/parity tests pass.

| Phase | Objective/files | Depends on | Acceptance/tests, risk and gate |
|---|---|---|---|
| 1. Package/config foundation | Create adapter package/base, config schema/resolver, import/config tests; modify only config utility if approved | This design + confirm source roots | Clean import, resource error, module-origin tests; no model init/download. Fix JSON/YAML/schema mismatch. Review before adapters. |
| 2. Expert1 adapter | Create expert1_adapter.py + contract tests | Phase 1; LAION base available | Single-pass extraction, no main side effect, strict full-state restore, score/feature parity. **First phase requiring fine-tuned checkpoint** for full acceptance. |
| 3. Expert2 adapter | Create expert2_adapter.py + namespace tests | Phase 1; MGA source/YAML/base | Module origin, ASE strict load/position_ids guard, full Expert2 strict restore, config-derived width/parity. Require fine-tuned artifact for final acceptance. |
| 4. Expert3 adapter | Create expert3_adapter.py + partial-loader tests | Phase 1; exact base path/hash and local text cache | Existing best loader, released assignment, D/5D, dynamic width, no downloads, parity. Block if source path/hash unresolved. |
| 5. Expert4 adapter | Create expert4_adapter.py + mask/fingerprint tests | Phase 1; BEATs/CLAP/UniLM and best bundle | Reuse load_finetuned, source identity, trainable-key checks, [B,1]/[B,1024], mask/length parity. |
| 6. Unified Dataset | Create data/expert_preprocessing.py and unified_moe_dataset.py + tests | Input audits; adapter key map agreed | Four exact branches, same sample IDs/order, multiple captions, raw/norm labels, B=1/B>1. No data cache mutation. |
| 7. MOEModel refactor | Modify model/MOE_model.py; retain Basic_comp.py | Adapters report feature_dim; dataset contract | No legacy construction, gate width sum, 4 fixed order, safe weighted output, batch checks, experts mode guard. |
| 8. Training/config integration | Modify Train_MOE.py, config utility, configs; tests | Refactored MOE + decision on entropy/paper discrepancy | Optimizer gate-only, labels agree, best/resume identity metadata, raw metrics conversion. Do not alter loss/selection without approval. |
| 9. Artifact parity/integration | Add integration tests/docs/manifest checks only | All four fine-tuned checkpoints obtained | Strict/partial load, standalone score/feature parity, gradients/frozen invariants, E2/E4 namespace, gate B=1/B>1. No next phase until pass. |
| 10. Colab reproduction | Add Colab run instructions/config override and execution record | Phase 9 pass; official rules/resources confirmed | Fresh runtime, offline cache, same fingerprints/predictions, measured memory/batch. No new model behavior. |

Acceptance condition between phases: report and tests reviewed; no phase silently changes preprocessing, expert head, score, feature order, loss, or split logic. Phases 2–5 may be developed without artifacts with stubs, but phase-specific checkpoint acceptance is gated by the actual files.



## 18. Decision register

| ID | Subject | Options | Recommendation | Reason / impact | Verification needed | Status |
|---|---|---|---|---|---|---|
| D01 | Common adapter type | Abstract nn.Module base; Protocol/convention | Base nhỏ với feature_dim và shape checks | Bốn adapters cần cùng model registration/output invariant, không cần plugin framework | Stub contract tests | PROPOSED |
| D02 | E1 score+feature | Call encoder twice; expose one-pass embeddings | Adapter calls inner encoder once and recomputes exact original dot | Tránh double inference, giữ score mathematics | Hook count + standalone score parity | PROPOSED |
| D03 | MGA namespace | Dynamic sys.path as-is; alias-loader; isolated process/origin checks | Fresh process + origin assertions; fail on foreign module | Vendor imports global models/tools; avoid copy/vendor rewrite | Test clean import orders in same interpreter | PROPOSED; runtime unresolved |
| D04 | M2D base path | Auto-search alternatives; explicit override | Explicit configured path and expected hash; fail if absent | Avoid silently substituting a different checkpoint; checked-in relative path discrepancy | Confirm authoritative local/Colab artifact path/hash | BLOCKED until resource manifest |
| D05 | M2D feature order | Semantic reorder; preserve training order | Preserve concat_assignment released | Fine-tuned head is slot-sensitive | Best checkpoint model_config + parity | PROPOSED |
| D06 | BEATs import identity | Clear/rebind global module; fresh process and source check | Fresh runtime/origin validation, never delete loaded module | Current loader rejects a different BEATs origin | Integration import test | PROPOSED; collision runtime unresolved |
| D07 | Unified data implementation | Reuse four separate loaders in synchronized streams; new row-level branches | One manifest row + four source-equivalent branches/collate | Single sample identity and four different preprocessing needs | Waveform/caption parity to each native dataset | PROPOSED |
| D08 | Feature dimensions | Hardcode 7168; lazy gate; adapter metadata | Construct experts, read feature_dim, validate, then build Linear | E3/M2D and E1 dimensions dynamic; avoids silent mismatch | Restore actual models and compare batch output widths | PROPOSED; numeric dims unresolved |
| D09 | Gate checkpoint metadata | Store only gate tensor; bind expert identities/config/order | Gate-only best plus identity/dim/config manifest; separate resume | Prevent use with different experts/order and keep inference file small | Save/load tests with changed-ID rejection | PROPOSED |
| D10 | Expert eval mode | Freeze params only; explicit setter; MOE.train override | MOE train override reapplies eval after parent recursion | Root loop calls train() each epoch; E2 has custom mode logic | Assert all descendants eval during gate train | PROPOSED |
| D11 | Device placement | All resident on GPU; CPU swapping; feature cache | First measure resident GPU path; defer optimization | Any alternative may affect latency, precision, identity or rules | Peak VRAM and prediction parity | UNRESOLVED |
| D12 | Preprocessing parity | Share waveform/tokenization; preserve branches | Separate branch outputs and raw captions, exact native transforms | Sample rates, decoders, crop/pad and text stripping differ | Per-expert comparison on identical rows | PROPOSED |
| D13 | Local/Colab paths | Absolute paths in code; separate machine configs; rooted overrides | Relative config paths plus explicit root overrides | Portable and auditable resource mapping | Fresh local and Colab config resolution | PROPOSED |
| D14 | Entropy term vs paper | Keep current objective; remove/retune entropy | Preserve current source objective until a research decision approves change | Removing 0.2 term changes training algorithm; paper doesn't document it | Compare chosen spec with project owner/paper implementation | UNRESOLVED |
| D15 | Challenge compliance | Assume paper rules apply; official 2027 rule check | Do not claim compliance until official 2027 rules are located | Checked-in paper/README are for 2026, no rules source found | Obtain current official rules and verify caching/data restrictions | BLOCKED for compliance verification |
| D16 | E1/E2 base identity | Trust path alone; record manifest/hash | Require trusted external run manifest where available; report absence otherwise | Raw full-state checkpoints do not encode base fingerprint | Compare original fine-tune run resource identity | UNRESOLVED |

No decision in this table is implemented here. Items marked unresolved/blocked are explicit inputs to implementation review, not hidden defaults.

## 19. Rủi ro, blockers và giả định

| Item | Classification | Consequence / handling |
|---|---|---|
| Fine-tuned checkpoint files external/unavailable | BLOCKED for runtime | Design can proceed; strict restore and parity phases wait |
| D_CLAP and D_M2D numeric widths | UNVERIFIED | Discover from restored models; don't use legacy dimension |
| Expert1 LAION normalization/actual score range | UNVERIFIED | Keep original dot; measure embeddings/score on fixed samples |
| Expert1/E2 original base identity manifest | UNVERIFIED | Require trusted provenance or document weaker assurance |
| MGA models/tools and BEATs top-level namespaces | STATIC_VERIFIED risk; runtime UNVERIFIED | Fresh process, validate __file__, fail on collision |
| M2D configured path vs inventory path | STATIC_VERIFIED discrepancy in LOCAL-02.2 | Explicit path override; no auto-search |
| Expert4 key-padding mask allows padded query pooling | STATIC_VERIFIED behavior, parity impact UNVERIFIED | Preserve behavior and compare standalone/adapter |
| Gate hardcoded width and score broadcasting | STATIC_VERIFIED current issue | Dynamic width + rank validation before train |
| Parent train mode can reactivate expert train mode | STATIC_VERIFIED risk | Override/setter and runtime state test |
| Current config parser/schema and data label key mismatch | STATIC_VERIFIED current incompatibility | Resolve before first integrated training smoke test |
| Official 2027 rules unavailable in repo | BLOCKED compliance claim | Check official rules before feature caching/submission workflow |
| Paper objective vs current extra entropy regularizer | STATIC_VERIFIED discrepancy | Keep source default until approved decision |
| Cached features could misalign or leak splits | PROPOSED operational risk | Cache key binds sample/model/preprocess; isolate train/validation/test |
| Resource sizes do not predict peak memory | DERIVED risk | Measure actual resident/activation memory; no batch-size promise |
| PyTorch checkpoint deserialization | STATIC_VERIFIED some loaders use weights_only=False | Only load trusted artifacts; no automatic downloads |

Assumptions labeled PROPOSED: unified row represents one audio-caption-score example; deterministic eval preprocessing is used while experts are frozen; all experts see the same row/order; feature concatenation is the gating input. These are design choices consistent with current source/audits, not runtime-confirmed behavior.

## 20. Source references

### Prior static audits

- docs/LOCAL-02/expert-architecture-audit.md — active fine-tuned classes, component paths, dimensions and freeze policies.
- docs/LOCAL-02/checkpoint-compatibility.md — full/partial schemas, resource inventory, restore order, M2D path discrepancy and fingerprint checks.
- docs/LOCAL-02/input-contracts.md — four preprocessing branches, captions, lengths/masks, label normalization and root dataset mismatch.
- docs/LOCAL-02/output-contracts.md — score/features, 7168 assessment, dynamic gate width, score rank and frozen-mode risks.

### Current root pipeline

- model/MOE_model.py:L3-L7,L11-L24,L28-L89 — legacy imports/loader; gate shape; hardcoded 7168; freeze, feature concat and weighted score.
- model/Basic_comp.py:L13-L128,L202-L257,L261-L336 — legacy expert and fusion output implementations.
- data/dataset.py:L34-L87,L89-L135,L138-L183 — root dataset keys, audio/resampler path, labels, collate and inference shape.
- Train_MOE.py:L21-L26,L29-L105,L108-L132,L159-L219,L231-L245 — entropy term, data/model/optimizer, mode changes, metrics, save/resume/entrypoint.
- loss/loss_function.py:L6-L103 — contrastive, clipped MSE, configured combination.
- utils.py:L1-L14,L27-L43 — JSON-only config loading and recursive device movement.
- configs/config.yaml:L1-L65 — absolute roots, sample rates, loss values and schema.
- README.md:L27,L40-L75,L153-L166, sections Method/Loss/Train/Inference — repository's 2026 description and documented objective.
- MOESCORE.pdf: §2.1–2.3, pp.1–2 — feature-aware four-expert MoE, experts frozen during gate training, and paper loss formulation. This paper is not the official 2027 rules.

### Fine-tuned experts/resources

- expert1_finetune/Train_Expert1.py:L98-L144,L195-L221,L247-L300 — CLAP init/embedding forward, output score, training/restore/save.
- expert2_finetune/expert2_model.py:L12-L23,L26-L99 — MGA wrapper, freeze modes, key compatibility, dynamic import and base factory.
- expert2_finetune/Train_Expert2.py:L103-L115,L127-L175,L180-L218 — fine-tuned full best state and resume distinction.
- expert2_finetune/expert2_finetune.yaml:L1-L20; expert2_finetune/MGA-CLAP/settings/inference_example.yaml:L1-L36 — active freeze/architecture configuration.
- expert2_finetune/MGA-CLAP/models/ase_model.py:L62-L144 — MSC/codebook/projection dimensions.
- expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py:L95-L166,L181-L205,L210-L266 — class, dimensions/order, partial freeze, forward output.
- expert3_finetune/XACLE_2027_baselinemodel/utils/expert3_checkpoint.py:L13-L93,L130-L156 — best and resume partial loaders/checks.
- expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py:L32-L65,L386-L414 — path/config resolution, construction and resume.
- expert3_finetune/XACLE_2027_baselinemodel/configs/expert3_moescore.json:L15-L16,L40 — expected base hash/path and released assignment.
- expert4_finetune/model.py:L8-L24; expert4_finetune/inference.py:L50-L59 — model factory and inference restore.
- expert4_finetune/audio_encoder.py:L21-L48,L50-L76 — BEATs source identity, dimensions, masks.
- expert4_finetune/text_encoder.py:L39-L60,L111-L124 — text projection and token mask.
- expert4_finetune/seq_coattn.py:L6-L31 — attention, pooling and regression shapes.
- expert4_finetune/trainer.py:L120-L160,L224-L265,L320-L340 — partial state, identities and resume/best.
- expert4_finetune/dataset.py:L23-L26,L73-L115 — standalone sample/length/collation.
- expert4_finetune/config.py:L7-L49 — resource and training config validation.

### Evidence status and final design status

- Current source properties above are STATIC_VERIFIED unless identified as discrepancies requiring runtime.
- Equations such as D_gate and batch waveform memory are DERIVED from source/config.
- Adapter modules, config schema, dataset branches, mode override, checkpoint metadata and phases are PROPOSED, not implemented.
- Numerical dimensions, real checkpoint restoration, source identity hashes at runtime, prediction parity, gate behavior after training and official 2027 compliance remain UNVERIFIED/BLOCKED as specified.
- Only docs/LOCAL-02/adapter-design.md is modified/created by this task; no production code/config/checkpoint is changed.
- Final status: **PASS — DESIGN COMPLETE**. This confirms design/report completion only, not implementation readiness, checkpoint compatibility, prediction parity, competition compliance or successful MoE training.



