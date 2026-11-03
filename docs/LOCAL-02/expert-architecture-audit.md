# LOCAL-02.1 — Kiểm toán kiến trúc các Expert

> Phạm vi: phân tích tĩnh working tree hiện tại. Không tải checkpoint, khởi tạo model, chạy inference/train hay kiểm tra tương thích checkpoint. Dòng được ghi theo dạng `đường/dẫn.py:Lx-Ly`.

## 1. Tóm tắt

| Expert | Class được fine-tune | Kiến trúc chính | Kết quả hiện có | Mức adapter dự kiến |
|---|---|---|---|---|
| 1 — LAION-CLAP | `expert1_finetune.Train_Expert1.Expert1` | LAION-CLAP, cosine giữa embedding audio/text | `forward(batch) -> score[B]`; không trả features | Vừa |
| 2 — MGA-CLAP | `expert2_finetune.expert2_model.Expert2` bọc `ASE` | HTSAT + BERT, projection, Modality-Shared Codebook (MSC), cosine | `(score[B,1], features[B,2048])` | Ít |
| 3 — M2D-CLAP | `...models.moescore_expert3.MoescoreExpert3` | PortableM2D, audio/text projection, alignment head | `(score[B,1], cross_features[B,5D])`; `D` lấy từ checkpoint/model | Vừa đến đáng kể |
| 4 — BEATs + CLAP | `expert4_finetune.model.Expert4` | BEATs + CLAP RoBERTa + hai chiều cross-attention + head | `(score[B], fused[B,1024])` | Ít đến vừa |

Các class trên được xác nhận qua call site trong entry point, không chỉ dựa theo tên. Khác biệt giao diện quan trọng: Expert 1/4 trả score rank 1; Expert 2/3 trả `[B,1]`. MOE legacy nhân score với trọng số `[B,1]`, nên adapter cần chuẩn hóa score thành cùng một shape. `MOEModel` còn hard-code `gate_input_dim=7168`; tổng feature width của các model fine-tune chưa thể xác nhận tĩnh do kích thước embedding CLAP/M2D phụ thuộc checkpoint. Tương thích gate và checkpoint thuộc các bước kiểm toán sau.

## 2. Expert 1 — LAION-CLAP

### Class, thành phần và luồng tính toán

**Class thực tế:** `Expert1` trong `expert1_finetune/Train_Expert1.py:L125-L137`; `main()` khởi tạo đúng class này tại `L215-L221`. Đây là wrapper fine-tuning; backbone là `laion_clap.CLAP_Module(enable_fusion=False)` được nạp checkpoint trong `CLAPAudioTextEncoder.__init__` (`L98-L115`). Không có head regression bổ sung.

```text
batch["clap_wavs"], batch["caption_tokens"]
  └─ CLAPAudioTextEncoder
      ├─ LAION-CLAP audio encoder → audio_embedding[B,D]
      └─ LAION-CLAP text encoder  → text_embedding[B,D]
  └─ sum(audio_embedding * text_embedding, dim=-1) → score[B]
```

`CLAPAudioTextEncoder.forward` gọi `get_audio_embedding_from_data(..., use_tensor=True)` và `get_text_embedding(..., use_tensor=True)`, trả hai embedding (`L117-L122`). `Expert1.forward` lấy tích theo phần tử, cộng chiều embedding và chỉ trả score (`L132-L137`). Feature ghép chưa được trả qua API của expert hiện tại, dù hai embedding được tạo bên trong encoder.

### Kích thước và fine-tuning

Nguồn này không khai báo số chiều embedding; chỉ chứng minh hai tensor phải có shape tương thích để nhân theo phần tử. Gọi chiều chung là `D_CLAP`, phụ thuộc cấu hình/checkpoint và cần xác minh runtime ở bước sau. Score là `[B]`. Không có fusion module ngoài cosine similarity.

Encoder đóng băng toàn bộ tham số trước, sau đó mở tên chứa `projection` hoặc `mlp` và audio branch `layers.2`, `layers.3`, `head`, `norm` (`Train_Expert1.py:L108-L115`). Optimizer chỉ nhận tham số `requires_grad` (`L216-L221`).

### So với legacy và tái sử dụng

Legacy `Basic_comp.Expert1` cũng dùng LAION-CLAP, chọn cùng nhóm trainable và tính cosine; legacy forward trả `(score[B,1], cat(audio,text)[B,2D])` (`model/Basic_comp.py:L202-L257`). Bản mới cố định `enable_fusion=False`, còn legacy lấy từ config (`L208-L214`); vì vậy tương đương về đường tính cosine khi cấu hình legacy tắt fusion, nhưng không thể khẳng định tương đương với mọi cấu hình. Module nesting mới tuyên bố giữ tên tương thích legacy (`Train_Expert1.py:L98-L100`), nhưng checkpoint không được kiểm tra ở task này.

**Phân loại:** tương thích khi dùng adapter; kiến trúc phần backbone/score tương đương có điều kiện, interface khác. Cần map keys audio/text và expose embeddings để tạo features; ép score về `[B,1]`. Nỗ lực: **vừa**.

### Evidence

- **Finding:** class huấn luyện và wrapper backbone. **Source:** `expert1_finetune/Train_Expert1.py:L98-L137`. **Symbol:** `CLAPAudioTextEncoder`, `Expert1`. **Evidence:** `Expert1.forward` gọi encoder rồi trả tổng cosine. **Conclusion:** đây mới là class fine-tune hiện hành.
- **Finding:** class được optimizer huấn luyện. **Source:** `expert1_finetune/Train_Expert1.py:L215-L221`, `L240-L255`. **Symbol:** `main`. **Evidence:** instantiate `Expert1`, optimizer nhận các tham số mở gradient, loop gọi `expert(batch)`. **Conclusion:** xác nhận call path training.
- **Finding:** legacy trả thêm features. **Source:** `model/Basic_comp.py:L244-L257`. **Symbol:** `Basic_comp.Expert1.forward`. **Conclusion:** interface legacy khác bản fine-tune.

## 3. Expert 2 — MGA-CLAP

### Class, thành phần và luồng tính toán

**Class fine-tune:** `Expert2` trong `expert2_finetune/expert2_model.py:L26-L71`. Training `Train_Expert2.train` đọc architecture YAML, gọi `create_expert2(..., freeze_policy=config.get(...))`, sau đó tối ưu tham số mở gradient (`Train_Expert2.py:L143-L151`). `create_expert2` khởi tạo `ASE`, nạp pretrained state rồi bọc trong `Expert2` (`expert2_model.py:L86-L99`). YAML đang chọn `freeze_policy: partial` (`expert2_finetune/expert2_finetune.yaml:L6-L10`).

```text
mga_wavs + caption_tokens
  └─ Expert2 → MGAEncoder → ASE
      ├─ HTSAT audio encoder → frame features → frame_proj
      ├─ BERT text encoder → word features → word_proj
      ├─ shared codebook [4096, 1024] + MSC aggregation
      └─ L2 normalize audio/text vectors
  ├─ cosine similarity → score[B,1]
  └─ concat(audio,text) → features[B,2048]
```

`ASE` chứa `AudioEncoder`, `TextEncoder`, `frame_proj`, `word_proj`, temperature và codebook; `embed_size=1024`, codebook có 4096 entry (`MGA-CLAP/models/ase_model.py:L95-L128`). Cấu hình kiến trúc trỏ audio sang transformer HTSAT và text sang `bert-base-uncased` (`MGA-CLAP/settings/inference_example.yaml:L17-L26`). `ASE.encode_audio/text` cho frame/word embeddings đã project (`ase_model.py:L136-L144`). MSC tính attention giữa token/frame và codebook, max-pool theo chuỗi rồi weighted-sum codebook (`ase_model.py:L62-L93`). Wrapper fine-tune gọi MSC cho cả hai modality và L2-normalize trước cosine (`expert2_model.py:L17-L23`).

### Kích thước và fine-tuning

Embedding sau projection và mỗi vector MSC là 1024 chiều từ `embed_size`; text/audio vectors đều `[B,1024]`. Do đó concat là `[B,2048]`, score là `[B,1]` (`ASE:L103-L128`; `expert2_model.py:L68-L71`). Theo cấu hình active, HTSAT encoder trả `audio_width=768`, BERT base có `text_width=768`; cả hai được project từ 768 sang 1024 (`MGA-CLAP/models/audio_encoder.py:L35-L56`; `models/text_encoder.py:L38-L60`; `ASE:L103-L122`).

Policy `partial` đóng băng tham số khác, mở hai block HTSAT cuối (`layers.3.blocks.0/.1`), audio norm, hai layer BERT cuối (10/11), `frame_proj`, `word_proj`, và codebook (`expert2_model.py:L35-L48`). `Expert2.train()` giữ encoder ở eval rồi đưa đúng các block/layers đã mở sang train (`L50-L65`). Policy `full` cũng được hỗ trợ và mở toàn bộ transformer layers/norm, BERT, projections và codebook; đây không phải lựa chọn YAML hiện tại.

### So với legacy và tái sử dụng

Legacy `Basic_comp.Expert2` dùng cùng `ASE`/MSC/cosine và concat features; nhưng nạp ASE qua wrapper riêng, gọi `audio.mean(1)` và gọi `load_state_dict(..., strict=False)` (`model/Basic_comp.py:L132-L161,L261-L272`). Current wrapper nhận `mga_wavs` và tự `squeeze(1)` (`expert2_model.py:L17-L23,L68-L71`), với policy fine-tuning tường minh. Kiến trúc chính **tương đương về tính toán embedding/cosine**, nhưng lớp trainable và interface đã được tổ chức lại; checkpoint tương thích chưa được khẳng định.

**Phân loại:** tương thích với adapter, nỗ lực **ít**: map đúng `mga_wavs`/tokenized captions và chuẩn hóa output. Nguồn MGA dùng layout `MGA-CLAP` và `create_expert2` thêm source vào `sys.path` (`expert2_model.py:L86-L91`); tích hợp cần tránh xung đột module tên `models` nếu các expert được import cùng tiến trình.

### Evidence

- **Finding:** active class. **Source:** `expert2_finetune/Train_Expert2.py:L143-L151`; `expert2_finetune/expert2_model.py:L86-L99`. **Symbol:** `train`, `create_expert2`. **Conclusion:** training nạp ASE rồi dùng `expert2_model.Expert2`.
- **Finding:** components và 1024 dimension. **Source:** `expert2_finetune/MGA-CLAP/models/ase_model.py:L95-L144`; `settings/inference_example.yaml:L1-L26`. **Conclusion:** projections vào shared 1024-D codebook space.
- **Finding:** MSC and cosine outputs. **Source:** `expert2_model.py:L17-L23,L68-L71`; `ase_model.py:L62-L93`. **Conclusion:** returns score and 2048-D concat feature.
- **Finding:** actual freeze policy. **Source:** `expert2_finetune/expert2_finetune.yaml:L6-L10`; `expert2_model.py:L35-L65`. **Conclusion:** config selects partial; full is supported alternative only.

## 4. Expert 3 — M2D-CLAP

### Class, thành phần và luồng tính toán

**Class fine-tune:** `MoescoreExpert3` trong `expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py:L95-L165`. `train()` khởi tạo đúng class này bằng checkpoint/config đã resolve rồi đưa lên device (`train_expert3.py:L370-L401`). `PortableM2D` là backbone/runtime vendor được tạo trong class này, không phải final Expert wrapper (`moescore_expert3.py:L123-L130`).

```text
wavs [B,T] hoặc [B,1,T] + raw caption strings
  └─ PortableM2D
      ├─ audio ViT/M2D → audio_proj → clip vector [B,D]
      └─ CLAP text encoder → (text_proj nếu checkpoint có) → [B,D]
  └─ L2 normalize
  └─ concat(audio, text, |audio-text|, audio*text, cosine broadcast)
      → cross_features[B,5D]
  └─ Linear(5D,512)-ReLU-Dropout-Linear(512,128)-ReLU-Dropout-Linear(128,1)
      → score[B,1]
```

`PortableM2D` loads ViT backbone/checkpoint and calculates internal feature width from positional embedding and flat/non-flat stack factor (`vendor/portable_m2d.py:L339-L369`). It exposes `encode_clap_audio` and `encode_clap_text` (`L449-L470`). Current Expert3 invokes those methods, validates `[B,D]`, normalizes, checks audio/text dimensions equal, then creates cross features and prediction (`moescore_expert3.py:L210-L266`). Text projection may be absent from checkpoint; then D is inferred from text encoder config (`L75-L92,L132-L147`).

### Kích thước và fine-tuning

**Không hard-code D.** `embedding_dim = audio_dim`, where `audio_dim` is derived from `backbone.audio_proj`'s last linear output (or normalized output shape), and `text_dim` comes from `text_proj` or text encoder configuration. Equality is checked at initialization. Fusion width is exactly `5D`, because five D-wide vectors are concatenated; head input is assigned as `cross_feature_dim=5*embedding_dim` (`moescore_expert3.py:L62-L92,L132-L156,L255-L266`). The final score is `[B,1]`. The actual numeric D depends on the local checkpoint/projector and remains **runtime-unverified** here.

`_configure_partial_finetuning()` first freezes all M2D parameters, then opens the last two audio backbone blocks, last two text transformer layers, `audio_proj`, and `text_proj` if present (`moescore_expert3.py:L181-L205`). The alignment head is outside `self.m2d`, so it stays trainable by default. Training optimizer uses trainable model parameters; no later freeze override appears in the traced construction (`train_expert3.py:L386-L401`).

### So với legacy và tái sử dụng

Legacy `Basic_comp.Expert3` uses `in_dim=3072` (`model/Basic_comp.py:L276-L291`) and names a method `encode_clap_audio_multiscale` (`L191-L198`). The current vendor class exposes `encode_clap_audio`, not that method (`vendor/portable_m2d.py:L449-L470`). Legacy head size therefore cannot be assumed to fit current `5D`; current code intentionally derives it dynamically. Legacy also returns cross features in `(audio,text)` assignment while current released configuration obtains `(text,audio)` from `encode_modalities` then binds that pair as `(audio_feats,text_feats)` by documented released behavior (`moescore_expert3.py:L242-L260`). Dot product is symmetric, but ordered feature slots differ; for `concat_assignment="semantic"`, the assignment changes. These semantics and the checkpoint's D require verification.

**Phân loại:** kiến trúc khác legacy; current class reusable without rewriting its architecture, but adapter effort **vừa đến đáng kể** due raw caption requirement, batch key mapping, dynamic feature width, and the gate's fixed expected width. Must reconcile whether MoE needs the expert's returned `cross_features` or an alternate feature vector.

### Evidence

- **Finding:** model actually instantiated. **Source:** `train_expert3.py:L370-L401`; `moescore_expert3.py:L95-L130`. **Conclusion:** `MoescoreExpert3`, not legacy `Basic_comp.Expert3`, is current trainer model.
- **Finding:** dynamic widths and head. **Source:** `moescore_expert3.py:L62-L92,L132-L156`. **Conclusion:** output feature dimension is `5D`, D is checkpoint-derived and checked.
- **Finding:** forward/dataflow. **Source:** `moescore_expert3.py:L210-L266`. **Conclusion:** normalized clip embeddings, five-part interaction features, scalar head, score plus features.
- **Finding:** partial freeze. **Source:** `moescore_expert3.py:L181-L205`; trainer optimizer `train_expert3.py:L400-L401`. **Conclusion:** final two blocks/layers plus projections and head train.
- **Finding:** vendor API. **Source:** `vendor/portable_m2d.py:L339-L369,L449-L470`. **Conclusion:** flat feature dimension depends on backbone and `flat_features`; API name differs from legacy call.

## 5. Expert 4 — Sequential Cross-Attention

### Class, thành phần và luồng tính toán

**Class fine-tune:** `expert4_finetune.model.Expert4`, constructed by `Expert4.from_pretrained()` which creates `AudioEncoder` and `TextEncoder.load_pretrained()` (`model.py:L8-L24`). Trainer calls this factory (`trainer.py:L229-L245`).

```text
wavs + lengths → BEATs (12 layers)
  → collect transformer outputs → learned softmax layer aggregation
  → LayerNorm/Linear/GELU/LayerNorm → audio sequence [B,Ta,512]
captions → CLAP RoBERTa (768 per token)
  → pretrained CLAP token projection (768→512), remove first token
  → text sequence [B,Tt,512]
audio attends to text (8 heads); text attends to audio (8 heads)
  → adaptive max pool each sequence → concat [B,1024]
  → Linear(1024,2048)-ReLU-Dropout-Linear(2048,1)-Tanh
  → score[B], fused[B,1024]
```

`SeqCoAttn` has two independent `MultiheadAttention(512,8)` modules, masks the keys in each direction, max-pools the attended sequences and concatenates two 512-D pooled vectors (`seq_coattn.py:L6-L22`). `RegressionHead` produces scalar tanh score (`L25-L31`).

### Kích thước và fine-tuning

Text branch explicitly requires RoBERTa hidden size 768 and 12 layers; projection is validated as 768→512 (`text_encoder.py:L39-L60`). It removes token index 0 and corresponding mask before returning `[B,Tt,512]` (`L111-L124`). Audio projection maps BEATs `cfg.encoder_embed_dim` to 512; sequence width is checked as 512 (`audio_encoder.py:L39-L48,L68-L75`). `SeqCoAttn` gives fused width 2×512=1024; head is explicitly 1024→2048→1 (`seq_coattn.py:L25-L31`).

BEATs requires exactly twelve layers; first ten transformer layers are frozen, last two and all non-layer backbone parameters remain trainable. Learned aggregation weights and audio projection are trainable (`audio_encoder.py:L39-L48`). Text freezes RoBERTa then opens last two layers and projection (`text_encoder.py:L43-L53`). SeqCoAttn and head default trainable; trainer optimizer includes every `requires_grad` parameter (`trainer.py:L229-L245`).

### So với legacy và tái sử dụng

Legacy `Basic_comp.Expert4` wraps `TemporalAlignModel`. It uses the same two attention directions, 8 heads, max pooling and feature concat, so that fusion path is **equivalent in core computation** (`model/Basic_comp.py:L13-L42,L86-L128`; compare `expert4_finetune/seq_coattn.py:L6-L31`). The legacy module declares residual projections, norms, dropout and fusion norm, but the residual/norm/fusion operations are commented out in `forward`; they are not active path (`Basic_comp.py:L95-L127`).

The outer implementation differs materially: current text tokens are projected to 512 before attention (`text_encoder.py:L111-L124`), while legacy `get_text_sequence()` returns unprojected last hidden state (`model/wrapper.py:L137-L158`) and `SeqCoAttn` expects the configured 512 dimension (`Basic_comp.py:L19,L86-L93`). Legacy audio path and checkpoint source also differ. Current head explicitly uses `Tanh` and 1024→2048→1; legacy `Projection` has configurable hidden size and only applies tanh if enabled (`Basic_comp.py:L46-L83`).

**Phân loại:** attention/fusion equivalent; full model **compatible with adaptation**, not proven equivalent. Reuse current class and map current batch keys; normalize score `[B]` to `[B,1]`. Effort **ít đến vừa**. BEATs dynamically loads vendored source via `sys.path` and module name `BEATs` (`audio_encoder.py:L23-L38`), a module identity collision risk if other copies are imported in the same process.

### Evidence

- **Finding:** trained model composition/forward. **Source:** `expert4_finetune/model.py:L8-L24`; `trainer.py:L229-L245`. **Conclusion:** current training instantiates `Expert4` with both encoders, cross-attention and regression head.
- **Finding:** BEATs feature width/freeze. **Source:** `audio_encoder.py:L39-L48,L68-L75`. **Conclusion:** 12 layers, final two open, output width 512.
- **Finding:** RoBERTa projection/freeze. **Source:** `text_encoder.py:L39-L60,L111-L124`. **Conclusion:** last 2/12 layers and projection train; token features 512-D.
- **Finding:** attention and regression dimensions. **Source:** `seq_coattn.py:L6-L31`. **Conclusion:** two 8-head directions; fusion width 1024; scalar tanh output.

## 6. Ma trận so sánh kiến trúc

| Thuộc tính | Expert 1 hiện tại | Expert 2 hiện tại | Expert 3 hiện tại | Expert 4 hiện tại |
|---|---|---|---|---|
| Main class | `Train_Expert1.Expert1` | `expert2_model.Expert2` + `ASE` | `MoescoreExpert3` | `expert4_finetune.model.Expert4` |
| Backbone | LAION-CLAP, fusion off | HTSAT + BERT-base | PortableM2D ViT + text encoder từ checkpoint | BEATs + CLAP RoBERTa |
| Project/fusion | cosine audio/text | frame/word projection + MSC | audio/text projector + 5-way interaction concat | learned layer projection + bidirectional attention + max-pool |
| Prediction | cosine | cosine | learned MLP | learned MLP + Tanh |
| Forward result | score `[B]` | score `[B,1]`, features `[B,2048]` | score `[B,1]`, features `[B,5D]` | score `[B]`, features `[B,1024]` |
| D/fusion width | `D_CLAP` checkpoint dependent | D=1024; fused 2048 | D checkpoint-derived; fused 5D | modality width 512; fused 1024 |
| Fine-tuning | CLAP projections/MLP/audio layers 2–3/head/norm | partial policy from YAML | last 2 audio/text blocks, projectors, head | BEATs last 2 + non-layer params, RoBERTa last 2 + text projection, fusion/head |
| Status vs legacy | Similar computation; feature interface differs | Similar core architecture; policy/wrapper differs | Architecturally different; legacy hardcoded head/API | Fusion equivalent; encoders/head differ |

## 7. Legacy vs fine-tuned implementation

| Expert | Legacy `Basic_comp.py` | Fine-tuned source | Kết luận |
|---|---|---|---|
| 1 | `CLAP_audio_text_encoder` + `Expert1`; config-selectable fusion; cosine + `(score, cat embeddings)` | `CLAPAudioTextEncoder` + `Expert1`; fusion disabled; score only | **Tương đương có điều kiện** ở backbone/cosine khi legacy fusion tắt. Cần adapter để expose feature và reshape score. |
| 2 | `MGA_audio_text_encoder` wraps ASE; MSC/cosine + concat | `MGAEncoder` + `Expert2` wraps ASE; MSC/cosine + concat | **Tương thích với adaptation**; core architecture similar, trainable policy explicit, input/output wrapper differs. Weight compatibility unverified. |
| 3 | `PortableM2D` wrapper, fixed `in_dim=3072`, calls multiscale method | `MoescoreExpert3`, checkpoint-derived `D`, `5D` head input and current `encode_clap_audio` API | **Khác kiến trúc/interface**; legacy dimensions/API cannot be assumed to fit. |
| 4 | `TemporalAlignModel` with BEATs/CLAP wrappers, configurable projection | standalone `Expert4`, validated BEATs/RoBERTa, 512-D token projections, 1024→2048→1 Tanh head | Attention core **tương đương**; full model **khác thành phần/head và tương thích qua adapter**. |

Legacy `MOEModel` freezes all four experts, calls each expecting `(score,features)`, concatenates all feature tensors and sets gate input to 7168 (`model/MOE_model.py:L28-L89`). Fine-tuned Expert 1 returns only score; Expert 4 returns score `[B]`; Expert 3 feature width is dynamic. Thus the legacy MoE interface and fixed gate dimension cannot be carried forward by name similarity alone. Exact aggregate width remains unverified until checkpoint/config values are inspected.

## 8. Yêu cầu cho Unified Expert Adapters

Mỗi adapter cần định nghĩa thống nhất:

1. Input keys, audio sample rate/layout/length, tokenized IDs versus raw text, mask convention, and device transfer.
2. Frozen inference behavior; fine-tuning policies documented above are separate and should not carry over implicitly.
3. `score` as `[B,1]` for weighted score combination; avoid broadcasting `[B]` against `[B,1]`.
4. A documented feature tensor `[B,F]`; expose current embeddings/features without inventing architecture. Expert 1 needs an embedding-level path; Experts 2–4 already return features.
5. Feature dimension `F` and gate input dimension checked at adapter/model construction rather than silently assumed. Expert 3 requires reading actual checkpoint-derived D.
6. Preserve raw model output semantics and training-time preprocessing. Adapter-only input mapping is needed for Expert 1 (`clap_wavs`, `caption_tokens`), Expert 2 (`mga_wavs`, `caption_tokens`), Expert 3 raw-string captions, and Expert 4 (`wavs`, `wav_lens`, `captions`).
7. Isolate Expert 2 `models.*` and Expert 4 dynamically imported `BEATs` modules to prevent Python module namespace collisions.

These are integration requirements, not adapter implementation instructions.

## 9. Risks and unverified assumptions

- Numeric `D_CLAP` for Expert 1 and numeric M2D `D` for Expert 3 are not statically established by the reviewed trainer source. Do not assign a numeric width without checkpoint/runtime inspection.
- Expert 3 code supports a missing `text_proj`; final D then comes from the loaded text encoder. Which branch a particular artifact takes is checkpoint-dependent.
- Legacy Expert 3 calls `encode_clap_audio_multiscale`; the inspected PortableM2D API declares `encode_clap_audio` and `encode_clap_text`. Whether another runtime module was intended by the legacy import is unverified.
- Current Expert 3 default `concat_assignment="released"` binds returned text/audio embeddings in the documented released order into local audio/text names. Its score is symmetric, but feature slot ordering can affect a downstream gate.
- Expert 2 training sets `sys.path` to its MGA source and imports top-level package `models`; module collisions with other `models` packages are possible. Source import path resolution for a unified application should be designed in LOCAL-02.2.
- Expert 4 dynamically imports top-level `BEATs`, mutating `sys.path`, and rejects a different pre-imported `BEATs` module. This is a concrete same-process integration risk.
- No numeric activation shapes, checkpoint compatibility, normalization equivalence, or inference parity were runtime-tested. No weights were opened.
- Legacy gate width 7168 may not equal the sum of current expert feature dimensions. Exact mismatch depends on checkpoint-derived dimensions; must be measured before wiring.

## 10. Đề xuất câu hỏi cho LOCAL-02.2

- Which exact fine-tuned artifact and base-backbone artifact are paired for each expert? Record formats, state-dict prefixes, metadata, and hashes.
- Does each checkpoint include full model state, only trainable parameters, or optimizer state? Expert 4 trainer explicitly saves trainable-only model state (`trainer.py:L120-L128,L323-L330`); Expert 2/3 formats must be checked independently.
- Can strict loading reconstruct all frozen weights from declared pretrained checkpoints, and what loading order is required?
- What are the checkpoint-derived numeric dimensions for Expert 1 CLAP and Expert 3 audio/text projection? Does current feature concatenation sum to a supported gate input dimension?
- Do saved weights match the fine-tuning class/module names or legacy `Basic_comp` names? Check keys and tensor shapes without silently using `strict=False`.
- Are model outputs numerically consistent with the fine-tuning code after adapter input mapping, including audio resampling, mono conversion, caption tokenization, and Expert 3 concat ordering?
- How will Expert 2 top-level `models` and Expert 4 top-level `BEATs` import names be namespaced in a process that contains all four experts?

## 11. Tham chiếu nguồn

- Expert 1 trainer/model: `expert1_finetune/Train_Expert1.py:L98-L137,L194-L221,L240-L255`.
- Expert 2 trainer/config/wrapper: `expert2_finetune/Train_Expert2.py:L127-L155`; `expert2_finetune/expert2_finetune.yaml:L1-L20`; `expert2_finetune/expert2_model.py:L12-L99`.
- Expert 2 ASE and encoders: `expert2_finetune/MGA-CLAP/models/ase_model.py:L62-L144`; `expert2_finetune/MGA-CLAP/models/audio_encoder.py:L12-L71`; `expert2_finetune/MGA-CLAP/models/text_encoder.py:L38-L75`; architecture settings `expert2_finetune/MGA-CLAP/settings/inference_example.yaml:L1-L36`.
- Expert 3 trainer/wrapper: `expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py:L370-L401,L509-L535`; `.../models/moescore_expert3.py:L62-L92,L95-L165,L181-L266`.
- Expert 3 vendor runtime: `expert3_finetune/XACLE_2027_baselinemodel/models/m2d/vendor/portable_m2d.py:L24-L31,L339-L369,L390-L470`.
- Expert 4 model/fusion/encoders/trainer: `expert4_finetune/model.py:L8-L24`; `expert4_finetune/seq_coattn.py:L6-L31`; `expert4_finetune/audio_encoder.py:L10-L17,L20-L75`; `expert4_finetune/text_encoder.py:L39-L60,L111-L124`; `expert4_finetune/trainer.py:L101-L125,L229-L245,L323-L330`.
- Legacy MoE and experts: `model/Basic_comp.py:L13-L42,L46-L128,L132-L198,L202-L340`; `model/wrapper.py:L7-L87,L90-L185`; `model/MOE_model.py:L11-L89`.

## Kết luận

**PASS cho kiểm toán tĩnh LOCAL-02.1; một số chiều số học và mọi tương thích checkpoint vẫn chưa xác minh.** Bốn class fine-tune và call path được truy vết. Những điểm runtime/checkpoint được đánh dấu rõ để xử lý ở LOCAL-02.2. Không có production code nào được sửa.
