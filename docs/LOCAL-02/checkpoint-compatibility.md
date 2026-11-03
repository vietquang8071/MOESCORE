# LOCAL-02.2 — Kiểm toán checkpoint và khả năng khôi phục

> Phạm vi: audit tĩnh source/config và metadata tên/kích thước file. Không mở/deserialise checkpoint, không tải model, không inference/train. `STATIC_VERIFIED` nghĩa là xác nhận từ mã lưu/nạp; mọi artifact fine-tuned hiện vẫn `UNVERIFIED`.

## 1. Tóm tắt

| Expert | Fine-tuned checkpoint | Nội dung được lưu | Tái dựng cần gì trước khi nạp fine-tuned state? | Trạng thái |
|---|---|---|---|---|
| 1 — LAION-CLAP | `weights/expert1_best.pt` (mặc định) | Raw full `Expert1.state_dict()`; không có resume checkpoint riêng | Khởi tạo đúng `Expert1` và nạp pretrained `weights/laion-clap.pt` trước | Schema static verified; artifact chưa có |
| 2 — MGA-CLAP | `weights/expert2_best.pt` | Raw full `Expert2.state_dict()`; `weights/expert2_run/last.pt` là resume bundle | Dựng ASE theo YAML, nạp `weights/mga-clap.pt['model']`, bọc `Expert2` với đúng freeze policy, rồi nạp best raw state | Schema static verified; artifact chưa có |
| 3 — M2D-CLAP | `runs/expert3_moescore/<run>/best_model.pt` | Versioned partial trainable parameters + metadata; resume có optimizer/RNG | Dựng `MoescoreExpert3` từ đúng M2D base, config và concat policy, rồi gọi `load_best_checkpoint()` | Schema/loader static verified; artifact chưa có |
| 4 — BEATs + CLAP | `<output_dir>/best_expert4.pt` | Structured bundle; `model_state_dict` chỉ chứa tham số trainable; `last_expert4.pt` thêm resume state | Nạp BEATs + CLAP từ đúng nguồn và UniLM source; dựng `Expert4`, xác minh fingerprints, gọi `load_trainable_state_dict()` | Schema/loader static verified; artifact chưa có |

Trong repository hiện có bốn pretrained file ở `weights/`, tổng kích thước hiển thị 5.3 GB. Không tìm thấy file fine-tuned `.pt/.pth/.ckpt/.safetensors/.bin` trong working tree. Do chưa mở fine-tuned artifacts, chưa có kết quả `ARTIFACT_VERIFIED` hay `RUNTIME_VERIFIED`.

## 2. Expert 1 — LAION-CLAP

### Khởi tạo và schema

Trainer tạo `expert1_finetune.Train_Expert1.Expert1(checkpoint, device)`; constructor chứa `CLAPAudioTextEncoder`, khởi tạo `laion_clap.CLAP_Module(enable_fusion=False, device=...)`, rồi gọi thư viện `load_ckpt(checkpoint)` (`Train_Expert1.py:L98-L137`). Best save dùng `expert.state_dict()` và detach/copy mọi state entry lên CPU rồi lưu atomically qua file `.tmp` (`L292-L301`). Đây là **raw state dict của toàn bộ model**, không phải dictionary có `model_state_dict`; state dict bao gồm tham parameters trainable và frozen, cùng persistent buffers nếu có. Không lưu epoch, SRCC, config, optimizer hay resume state trong payload này. Path mặc định là `weights/expert1_best.pt` (`L154-L156`). Không có resume checkpoint implementation trong trainer.

Module nesting do source xác định: `Expert1.model` → `CLAPAudioTextEncoder.model` → `CLAP_Module`. Vì vậy dự kiến có key prefix `model.model.`; các suffix/key cụ thể vẫn phụ thuộc LAION-CLAP version/model internals. Class docstring nói giữ module nesting và tên trainable tương thích `Basic_comp` (`L98-L100,L125-L130`), nhưng key set của artifact không được kiểm tra.

### Thứ tự khôi phục

1. Khởi tạo `Expert1(weights/laion-clap.pt, device)`. API hiện tại luôn cần pretrained CLAP file vì `CLAPAudioTextEncoder.__init__` gọi `load_ckpt()`.
2. Nạp raw fine-tuned state lên model đó với `load_state_dict(state, strict=True)`; có thể dùng `model.MOE_model.load_pretrain_weight`, vốn gọi `torch.load(path)` rồi `model.load_state_dict(...)` (PyTorch mặc định `strict=True`). Loader cũ không chỉ định `map_location`/`weights_only`, nên thiết bị và độ tin cậy file phải được xử lý ở adapter tương lai.
3. Kiểm tra không thiếu/thừa key, shape khớp, output hữu hạn và so sánh standalone.
4. Chuyển `.eval()`, sau đó freeze tham số cho MoE.

Vì best file chứa full state, pretrained base về lý thuyết sẽ bị ghi đè bởi các giá trị đầy đủ khi strict-load. Tuy nhiên, constructor hiện tại bắt buộc nạp pretrained file trước; API không cung cấp khởi tạo kiến trúc không checkpoint. File base vì vậy là dependency bắt buộc của restoration path hiện có.

**Schema:** type=`fine-tuned best`; format=`raw state_dict`; top-level keys là parameter/buffer names, chưa xác nhận artifact; coverage=`full model`; metadata=`UNVERIFIED/không được save theo code`; pretrained=`LAION-CLAP`; config cần để tái tạo training không nằm trong checkpoint. `STATIC_VERIFIED`; artifact/runtime `UNVERIFIED`.

**Legacy compatibility:** Có khả năng nạp vào legacy `Basic_comp.Expert1` nếu cùng LAION-CLAP version, `enable_fusion=False` và cùng module nesting; cả hai class có nesting tương đương theo source và trainer chủ động ghi chú compatibility. Nhưng legacy config có thể bật fusion (`model/Basic_comp.py:L208-L214`), và không có key/shape comparison artifact. Phân loại: **likely compatible but unverified**. Không xem tên key prefix là bằng chứng checkpoint thực tế khớp.

### Evidence

- **Finding:** pretrained loading. **Source:** `expert1_finetune/Train_Expert1.py:L101-L106`. **Symbol:** `CLAPAudioTextEncoder.__init__`. **Evidence:** tạo module CLAP rồi gọi `load_ckpt`. **Conclusion:** base checkpoint cần trước khi nạp state fine-tuned. **Status:** `STATIC_VERIFIED`; schema nội bộ của `load_ckpt` `UNVERIFIED`.
- **Finding:** full best save. **Source:** `expert1_finetune/Train_Expert1.py:L292-L301`. **Symbol:** best-SRCC block. **Evidence:** lưu toàn bộ `expert.state_dict()` sau CPU detach. **Conclusion:** raw full state không có optimizer/config metadata. **Status:** `STATIC_VERIFIED`.
- **Finding:** legacy generic loader. **Source:** `model/MOE_model.py:L5-L7,L33-L58`. **Symbol:** `load_pretrain_weight`, `MOEModel.__init__`. **Conclusion:** raw full state path only, `strict=True` mặc định; không phải loader cho mọi expert format. **Status:** `STATIC_VERIFIED`.

## 3. Expert 2 — MGA-CLAP

### Hai checkpoint khác nhau và thứ tự khôi phục

Pretrained ASE checkpoint là `weights/mga-clap.pt`, payload theo source có top-level `model`, được nạp vào object `ASE`. `create_expert2()` thêm source `MGA-CLAP` vào `sys.path`, import `models.ase_model.ASE`, copy architecture config rồi đặt `audio_encoder_args.pretrained=False`, `text_encoder_args.init_from_config=True`, `training.spec_augmentation=False`; sau đó dựng ASE, `torch.load(..., map_location='cpu', weights_only=False)`, lấy `checkpoint['model']` và gọi `load_mga_state()` (`expert2_model.py:L74-L99`). `load_mga_state` chỉ bỏ legacy `text_encoder.text_encoder.embeddings.position_ids` khi runtime ASE không khai báo key đó và tensor đúng dãy vị trí chuẩn; cuối cùng gọi `ase.load_state_dict(state, strict=True)` (`L74-L83`).

Fine-tuned model khác với ASE base: `Expert2` bọc `ASE` qua `MGAEncoder`, với state key nesting `model.model.*` (`expert2_model.py:L12-L33`). Best file do trainer ghi raw CPU copy của **toàn bộ** `model.state_dict()` vào path `config['output']` (`Train_Expert2.py:L202-L207`), mặc định `weights/expert2_best.pt` (`expert2_finetune.yaml:L8-L10`). `evaluate_checkpoint` dựng lại ASE+Expert2 rồi nạp raw state bằng `model.load_state_dict(state, strict=True)` (`Train_Expert2.py:L103-L115`). Đây là loader fine-tuned tốt nhất hiện có; không nạp fine-tuned Expert2 state thẳng vào ASE.

Thứ tự bắt buộc theo API hiện tại:

1. Đọc `expert2_finetune.yaml`, đặc biệt `mga_source`, `mga_config`, `mga_checkpoint`, `freeze_policy`.
2. Dựng ASE bằng đúng architecture YAML (`embed_size=1024`, `codebook` 4096×1024, HTSAT, BERT base; `MGA-CLAP/settings/inference_example.yaml:L1-L26`). Nạp pretrained ASE `checkpoint['model']` strict sau compatibility rule `position_ids`.
3. Bọc bằng `Expert2(..., freeze_policy='partial')` — đây là giá trị config hiện tại. Freeze policy ảnh hưởng tập `requires_grad`/mode, không đổi key tensor shapes; raw best state không lưu policy hay config, nên phải lấy từ đúng run config.
4. Nạp `expert2_best.pt` raw vào **Expert2** bằng `strict=True`, không vào ASE.
5. Xác nhận keys/shapes, output; chuyển `.eval()` và đóng băng khi adapter dựng xong.

`partial` mở block HTSAT cuối, norm, BERT layers 10/11, projections và codebook; `full` cũng được code hỗ trợ nhưng không phải YAML đang chọn (`expert2_model.py:L26-L65`; YAML `L10`). Toàn bộ state được save nên frozen pretrained values và trainable fine-tuned values cùng nằm trong best file; cần base trước theo constructor nhưng final state sẽ nạp lại toàn bộ.

### Best và resume

`weights/expert2_best.pt`: raw full `Expert2.state_dict()`, không có wrapper metadata. `weights/expert2_run/last.pt`: structured resume dict có `epoch`, `model`, `optimizer`, `scheduler`, `best_srcc`, `stale`, `config`, `metrics`, `torch_rng`, `numpy_rng`, `python_rng`, và `cuda_rng` nếu chạy CUDA (`Train_Expert2.py:L210-L218`). Resume nạp model strict, optimizer/scheduler và RNG (`L167-L175`). Không dùng `last.pt` làm inference/adapter artifact; dùng best file.

`torch.load(weights_only=False)` được dùng cho pretrained ASE, fine-tuned best evaluation và resume (`expert2_model.py:L97`; `Train_Expert2.py:L113,L168`). Với nguồn chỉ tin cậy; best fine-tuned là tensor-only raw state theo save code nên loader tương lai có thể ưu tiên `weights_only=True` trên PyTorch hỗ trợ.

**Schema:** pretrained ASE=`dict` có key `model` chứa ASE state; best=`raw full state_dict`; resume=`structured dict` như trên. Key prefixes fine-tuned dự kiến `model.model.*`. Artifact chưa có nên keys/shape `UNVERIFIED`.

### Evidence

- **Finding:** ASE restoration and legacy buffer handling. **Source:** `expert2_finetune/expert2_model.py:L74-L99`. **Conclusion:** pretrained payload goes to ASE strict; then wrapper is constructed. **Status:** `STATIC_VERIFIED`.
- **Finding:** best state format and strict load. **Source:** `expert2_finetune/Train_Expert2.py:L103-L115,L202-L207`. **Conclusion:** raw full Expert2 state is intended. **Status:** `STATIC_VERIFIED`.
- **Finding:** resume schema. **Source:** `expert2_finetune/Train_Expert2.py:L167-L175,L210-L218`. **Conclusion:** distinct resume bundle with optimizer/scheduler/RNG. **Status:** `STATIC_VERIFIED`.
- **Finding:** actual freeze policy/config. **Source:** `expert2_finetune/expert2_finetune.yaml:L6-L11`; `expert2_finetune/expert2_model.py:L26-L65`. **Conclusion:** `partial` selected; policy is not serialized in raw best state. **Status:** `STATIC_VERIFIED`.

## 4. Expert 3 — M2D-CLAP

### Pretrained restoration and partial best checkpoint

`MoescoreExpert3` needs the M2D base path and optionally expected SHA-256, cache/layout path, activation checkpointing and `concat_assignment`. Constructor checks file existence, computes SHA-256, compares configured expected hash if provided, prepares the directory layout by symlinking to `checkpoint-30.pth`, creates `PortableM2D(flat_features=True)`, initializes the lazy CLAP text encoder, derives D and head shape, then freezes M2D and selects trainable parameters (`moescore_expert3.py:L25-L59,L98-L165`). `PortableM2D.get_backbone` loads the pretrained file on CPU with `weights_only=False`, unwraps optional top-level `model`, extracts/reformats backbone keys (`models/m2d/vendor/portable_m2d.py:L266-L275`). Text encoder initialization may call Hugging Face `from_pretrained` depending on parsed M2D checkpoint family (`L545-L601`); ensure its tokenizer/model resource is already cached or provisioned locally. This audit made no Hub requests.

Fine-tuned `best_model.pt` is an **atomic, versioned partial checkpoint**, not a full `state_dict`. `_make_payload` produces top-level keys: `format_version`, `checkpoint_kind`, `model_trainable`, `model_config`, `base_checkpoint_identity`, `epoch`, `best_srcc`, `metrics`, `config` (`utils/expert3_checkpoint.py:L40-L51`). `FORMAT_VERSION=1` (`L13`); `checkpoint_kind="best"`. `model_trainable` includes each named parameter where `requires_grad` is true, cloned to CPU (`L16-L21`), including alignment head, open final audio/text blocks and projectors; frozen M2D weights are absent. Buffers are absent from `model_trainable` because the collector iterates `named_parameters()`.

`model_config` records checkpoint-derived `embedding_dim`, `cross_feature_dim=5D`, `concat_assignment`, activation checkpointing, opened layer indices, and whether `text_proj` exists (`moescore_expert3.py:L146-L165`). Thus it guards the dynamic head/feature architecture; it does not assume a fixed D. `base_checkpoint_identity` records path, size, SHA, and M2D model directory (`L167-L172`); loader compares the SHA only, while expected config is compared as a full dict (`expert3_checkpoint.py:L70-L75`).

### Existing loader and resume distinction

`load_best_checkpoint(model,path)` safely loads to CPU using `weights_only=True` (fallback to default `torch.load` only when the argument is unsupported), verifies `checkpoint_kind="best"`, format version, base SHA, exact `model_config`, and exact set of trainable parameter names. It copies each tensor to current parameter device/dtype under `no_grad` (`expert3_checkpoint.py:L60-L93`). There is no explicit preflight tensor-shape comparison; exact shape mismatches and any broadcasting behavior are left to `Tensor.copy_` (`L82-L85`) and require artifact-level testing. Do not replace this with generic `strict=False`: the frozen base must already be restored and exact trainable keys must be checked.

`last_checkpoint.pt` is `checkpoint_kind="resume"`, with same partial fields plus `optimizer`, `scheduler`, `scaler`, `stale_epochs`, and RNG state (`L130-L142`). `load_resume_checkpoint` validates kind, restores trainable parameters through the same checks, then optimizer/scheduler/scaler/RNG (`L145-L156`). Training calls it only for `--resume` (`train_expert3.py:L406-L414`). `save_best_checkpoint` and `save_resume_checkpoint` are called separately; best artifact is `run_dir/best_model.pt`, resume artifact `run_dir/last_checkpoint.pt` (`L483-L493`). Inference constructs the pretrained model first, calls `load_best_checkpoint`, then `.eval()` (`inference_expert3.py:L49-L56`).

### Config path and risks

The checked-in config includes expected SHA `238521...c8a1`, `concat_assignment="released"`, and `m2d_checkpoint="../weights/checkpoint-30.pth"` (`configs/expert3_moescore.json:L15-L16,L40`). `resolve_path` resolves relative paths against the Expert 3 source root (`train_expert3.py:L32,L45-L48`), so the configured value points to `expert3_finetune/weights/checkpoint-30.pth`. That path was not present in the inspected tree; the file found is `weights/checkpoint-30.pth` at repository root. Trainer allows a `--m2d-checkpoint` override (`train_expert3.py:L509-L538`); actual intended path must be confirmed. Do not compare the configured expected hash without hashing the local artifact; this task did not compute file hashes.

`concat_assignment` is included in `model_config`; restoration with the other value is rejected. M2D best uses partial parameter copy; pretrained base checkpoint is indispensable. Source-level risk: `inference_expert3.py` imports modules using top-level package `MOESCORE...` (`L12-L15`), which may not resolve when launched from inside the repository root unless it is exposed as that package; this was not executed.

**Schema:** best/resume are structured partial payloads; pretrained M2D checkpoint accepts optional `model` wrapper or raw state and is deserialized with `weights_only=False`; text encoder weights may be separate or sourced via pretrained identifier depending on M2D model family. Fine-tuned partial schema static verified; artifact and numeric D unverified.

### Evidence

- **Finding:** base identity/config capture. **Source:** `moescore_expert3.py:L108-L125,L146-L172`; `train_expert3.py:L386-L392`. **Conclusion:** constructor receives base, config, expected hash and records dynamic geometry. **Status:** `STATIC_VERIFIED`.
- **Finding:** best schema and format version. **Source:** `utils/expert3_checkpoint.py:L13-L57`. **Conclusion:** version 1, best tag, trainable-only weights plus metadata, atomic save. **Status:** `STATIC_VERIFIED`.
- **Finding:** exact validation and restoration. **Source:** `utils/expert3_checkpoint.py:L60-L93`. **Conclusion:** `weights_only`, format/kind, SHA/config/key equality, then tensor copy. **Status:** `STATIC_VERIFIED`.
- **Finding:** resume bundle. **Source:** `utils/expert3_checkpoint.py:L130-L156`; `train_expert3.py:L406-L414,L483-L493`. **Conclusion:** separate optimizer/scheduler/scaler/RNG resume path. **Status:** `STATIC_VERIFIED`.

## 5. Expert 4 — BEATs + CLAP + SeqCoAttn

### Pretrained initialization

`Expert4.from_pretrained(beats_checkpoint, clap_checkpoint, unilm_source_dir)` builds `AudioEncoder` and `TextEncoder.load_pretrained()` (`expert4_finetune/model.py:L8-L24`). BEATs encoder checks checkpoint and source files, dynamically imports BEATs from the given UniLM source directory, loads checkpoint with `map_location='cpu', weights_only=True`, reconstructs `BEATsConfig(checkpoint['cfg'])`, and strictly restores `checkpoint['model']` (`audio_encoder.py:L21-L38`). The UniLM source directory is required in addition to the BEATs weight file; trainer fingerprints its code identity.

CLAP text loader expects a checkpoint containing top-level `state_dict`, `model`, or directly a state mapping. It loads with `weights_only=False, mmap=True` (fallback without mmap for certain errors), validates CLAP projection shapes and RoBERTa embedding shapes, constructs a local RoBERTa-base model, checks all text-branch parameter names exist, removes `module.` prefixes, then calls `clap.model.load_state_dict(..., strict=False)`. It explicitly fails on missing `text_branch.*` or `text_projection.*` keys; extra keys and omissions outside those used text modules are ignored intentionally because it extracts only text branch, projection and tokenizer (`text_encoder.py:L8-L29,L62-L109`). This is not full CLAP restoration; Expert4 has no CLAP audio branch.

### Fine-tuned checkpoint, fingerprints and loader

Trainer fingerprints SHA-256 of BEATs checkpoint, CLAP checkpoint, train/validation CSVs and an identity object for UniLM source (`trainer.py:L132-L160,L224-L230`). Best save is structured: `model_state_dict` from `trainable_state_dict(model)`, `config`, `fingerprints`, `epoch`, `best_srcc` (`L320-L324`). The state dict collector includes only trainable named parameters (`L120-L122`): BEATs trainable parameters/last blocks, aggregation weights, audio projection, RoBERTa final layers, text projection, SeqCoAttn and regression head; it excludes frozen parameters and buffers. Resume `last_expert4.pt` adds `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `global_step`, `stale_epochs`, `seed`, `rng_state` and same config/fingerprints (`L329-L334`). Intended inference artifact is `best_expert4.pt`; `last_expert4.pt` is for resume. Neither stores a `checkpoint_kind` discriminator.

Inference `load_finetuned()` loads structured payload, verifies BEATs and CLAP hashes and UniLM source identity, reconstructs `Expert4` with those base resources, calls `load_trainable_state_dict`, then moves to device and `.eval()` (`inference.py:L50-L59`). `load_trainable_state_dict` compares exact expected trainable parameter-name set, then intentionally calls `load_state_dict(...,strict=False)` so omitted frozen base weights remain initialized. Exact key check prevents missing/extra fine-tuned trainable params; matching-key shape mismatch still raises from PyTorch loader. It does not explicitly inspect returned missing/unexpected key lists, compare `state['config']`, or reject resume-vs-best kind in inference (`trainer.py:L125-L129`; `inference.py:L50-L59`). Resume path separately validates training config and complete fingerprint dictionary before loading model/optimizer/scheduler/scaler/RNG (`trainer.py:L255-L265`).

Both base weight files and source tree must match fine-tuning run. `resources` are null in `expert4_colab.yaml` and must be set (`expert4_colab.yaml:L2-L12`); notebook selects `OUTPUT_DIR/best_expert4.pt` unless overridden (`Finetune_MOESCORE_Expert4.ipynb`, cells represented at lines 107-112, 221-223). Config itself is part of best payload but inference restoration accepts resources separately.

**Security:** BEATs tensor checkpoint uses safer `weights_only=True`. CLAP checkpoint uses `weights_only=False` by design and source comments say its supplied file has a NumPy scalar; only use the trusted intended file. Fine-tuned best/resume bundle loads through `load_checkpoint(...weights_only=False)` because resume includes Python/NumPy RNG state; loader explicitly documents trusted experiment-directory restriction (`trainer.py:L45-L48`). Keep these files trusted.

**Schema:** fine-tuned best and resume structured dicts; `model_state_dict` is trainable-only. Best is chosen for MoE. Resume only for same training run. Artifact unverified.

### Evidence

- **Finding:** pretrained restoration. **Source:** `expert4_finetune/model.py:L16-L18`; `audio_encoder.py:L21-L38`; `text_encoder.py:L62-L109`. **Conclusion:** both pretrained checkpoints and UniLM source are needed before fine-tuned weights. **Status:** `STATIC_VERIFIED`.
- **Finding:** partial state schema and safe non-strict controls. **Source:** `trainer.py:L120-L129,L320-L334`. **Conclusion:** save trainable params only; exact names checked before deliberate `strict=False`. **Status:** `STATIC_VERIFIED`.
- **Finding:** base identity validation/inference restoration. **Source:** `inference.py:L50-L59`; `trainer.py:L132-L160,L224-L225`. **Conclusion:** pretrained hashes and source identity checked, then partial parameters restored. **Status:** `STATIC_VERIFIED`.

## 6. Ma trận checkpoint

| Expert | Base checkpoint format/load | Fine-tuned best format/coverage | Resume format | Existing loader | Integrity validation | Recommended source |
|---|---|---|---|---|---|---|
| 1 | CLAP library `load_ckpt`; internal format unverified | Raw state dict, full model | Không có | Generic `load_pretrain_weight` (default strict) | PyTorch key/shape check only; no identity hash | `expert1_best.pt` |
| 2 | `dict['model']` → ASE strict after optional validated position_ids removal | Raw state dict, full Expert2 | Structured model+optimizer/scheduler/config/metrics/RNG | `create_expert2` then `load_state_dict(strict=True)` | Strict key/shape at each state load; no artifact fingerprint | `expert2_best.pt` |
| 3 | PortableM2D accepts model-wrapped/raw; `weights_only=False`; optional expected SHA | Version 1 structured; trainable parameters only | Same partial fields + optimizer/scheduler/scaler/RNG | `MoescoreExpert3` then `load_best_checkpoint` | Version, kind, base SHA, exact model_config, exact key set; copy enforces shapes | `<run>/best_model.pt` |
| 4 | BEATs `dict[cfg,model]` strict and `weights_only=True`; CLAP text subset, guarded `strict=False` | Structured bundle; trainable parameters only | Same partial state + optimizer/scheduler/scaler/RNG/run metadata | `Expert4.from_pretrained` then `load_finetuned` / `load_trainable_state_dict` | Base SHA/source identity; exact trainable keys; shapes enforced on load | `best_expert4.pt` |

**Tất cả trạng thái artifact:** `UNVERIFIED` do fine-tuned checkpoints chưa có trong local tree. Các save/load schemas ở bảng là `STATIC_VERIFIED` từ source.

## 7. Kiểm kê pretrained resources

File listing và kích thước dưới đây lấy từ `weights/`; không file nào được deserialize hoặc hash trong audit.

| Expert | Resource | Vai trò | Local availability | Cần cho restoration hiện tại |
|---|---|---|---|---|
| 1 | `weights/laion-clap.pt` (1.8G) | LAION-CLAP pretrained | Có | Có, `CLAP_Module.load_ckpt` bắt buộc trong constructor |
| 1 | `weights/expert1_best.pt` | Expert1 fine-tuned best | Không thấy | Có |
| 2 | `weights/mga-clap.pt` (1.6G) | Pretrained ASE payload `['model']` | Có | Có |
| 2 | `weights/expert2_best.pt` | Expert2 fine-tuned best | Không thấy | Có |
| 2 | `weights/expert2_run/last.pt` | Expert2 resume | Không thấy | Không cho inference; chỉ nếu resume |
| 3 | `weights/checkpoint-30.pth` (1.6G) | M2D-CLAP pretrained backbone | Có | Có; config path hiện lệch như nêu trên |
| 3 | `runs/.../best_model.pt` | Expert3 partial fine-tuned best | Không thấy | Có |
| 3 | `runs/.../last_checkpoint.pt` | Expert3 resume | Không thấy | Chỉ nếu resume |
| 4 | `weights/BEATs_iter3_plus_AS2M.pt` (345M) | BEATs pretrained | Có | Có |
| 4 | `weights/laion-clap.pt` (1.8G) | CLAP/RoBERTa pretrained text branch/projection | Có | Có |
| 4 | `expert4_finetune/unilm/beats/{BEATs.py,backbone.py,modules.py}` | BEATs implementation source | Có trong repo | Có, source identity phải trùng |
| 4 | `<output_dir>/best_expert4.pt` | Expert4 fine-tuned best | Không thấy trong tree | Có |
| 4 | `<output_dir>/last_expert4.pt` | Expert4 resume | Không thấy trong tree | Chỉ nếu resume |

Expert4 `resources` paths chưa điền trong YAML mặc định; cần dùng đường dẫn thực tế của base files, UniLM source và output dir. Không tự tải fine-tuned files.

## 8. Đánh giá tương thích với legacy MOESCORE

`model.MOE_model.load_pretrain_weight` nạp `torch.load(path)` như raw object rồi truyền thẳng vào `model.load_state_dict(...)`; không unwrap nested `model_state_dict`, không map location, mặc định strict=True (`model/MOE_model.py:L5-L7`). `MOEModel.__init__` gọi loader này cho từng legacy class (`L33-L58`).

| Expert | Legacy loader assessment | Lý do |
|---|---|---|
| 1 | **Likely compatible but unverified** | Best fine-tune raw full state; source ghi rõ giữ nesting tương tự Basic_comp. Chỉ khớp khi base architecture/fusion/version giống và key/shape thực tế strict-load được. |
| 2 | **Likely compatible but unverified** | Best raw full Expert2 và cả hai wrapper nesting ASE dưới `model.model`; pretrained cần nạp trước. Freeze policy không phải parameter key. Cần so sánh source/config và strict-load artifact thật. |
| 3 | **Incompatible với generic loader** | Best là structured partial dict; legacy loader không unwrap, legacy class/head/API khác. Phải dựng đúng MoescoreExpert3+M2D rồi dùng `load_best_checkpoint`. |
| 4 | **Incompatible với generic loader** | Best structured và chỉ trainable params; legacy `Expert4/TemporalAlignModel` không phải `expert4_finetune.Expert4`. Phải dựng pretrained current model, xác minh fingerprints, rồi specialized partial loader. |

Không có compatibility nào `ARTIFACT_VERIFIED`. Với Expert1/2, cùng prefix chỉ là dấu hiệu source-level tốt; checkpoint actual và tensor shape vẫn phải strict-test.

## 9. Restoration contracts cho adapter tương lai

Đây là thứ tự yêu cầu khái niệm; chưa tạo executable adapter.

```text
Expert1: CLAP pretrained → Expert1(enable_fusion=False) → raw full best state strict
Expert2: MGA source + YAML + ASE base['model'] strict → Expert2(freeze_policy='partial') → raw full best strict
Expert3: config + M2D base/hash + PortableM2D → MoescoreExpert3(dynamic D/concat) → load_best_checkpoint
Expert4: BEATs source/checkpoint strict + CLAP text checkpoint → Expert4.from_pretrained
         → check fingerprints → load_trainable_state_dict(partial exact keys)
```

Sau khi restoration hoàn tất: validate architecture/config/resource identity; chuyển device; gọi `.eval()`; đặt toàn bộ expert params `requires_grad=False`; chạy smoke forward với input đã preprocess đúng standalone; kiểm tra finite outputs, shape và parity với standalone. Với Expert4, loader hiện trả `.eval()` qua `load_finetuned`; Expert1/2/3 adapter cần thực hiện rõ bước này.

- Expert 1 phải giữ `enable_fusion=False`, `CLAP_Module` nesting và pretrained CLAP file hiện dùng. Strict-load raw best; chỉ cho phép các key/shape đúng.
- Expert 2 phải tách rõ ASE base `['model']` và Expert2 raw full checkpoint. YAML/source/freeze policy phải đúng run; không đẩy raw wrapper state vào ASE.
- Expert 3 phải dùng partial loader và đúng SHA/config; không thay bằng `strict=False`. Bắt lỗi thiếu key, model_config khác, base SHA khác, tensor shape khác. D phải được tạo từ base checkpoint trước.
- Expert 4 cần hai pretrained files, UniLM source identity, `Expert4.from_pretrained`, exact trainable-key loader. Chọn best; chỉ cho phép partial missing weights là frozen params đã được khôi phục từ các base resources tương ứng.

## 10. Runtime verification plan sau khi nhận fine-tuned checkpoints

Không chạy các bước này trong LOCAL-02.2.

1. **Inventory và trust:** xác minh path, kích thước, nguồn chuyển giao và hash từ manifest đáng tin cậy trước load; giữ bản gốc bất biến.
2. **Schema scan an toàn:** tải trên CPU với weights-only nếu schema cho phép; kiểm tra top-level keys/types, tổng parameter keys, prefixes và shapes. Chỉ cho phép `weights_only=False` với trusted inputs khi schema có NumPy/Python RNG objects; không dùng nguồn không đáng tin.
3. **Base identity:** so sánh pretrained SHA/config; Expert3 expected SHA + model_config + concat; Expert4 fingerprints BEATs/CLAP/UniLM; Expert2 xác nhận architecture yaml, freeze policy và MGA checkpoint identity nếu có manifest.
4. **Strict/partial load:** Expert1/2 strict-load raw full state; Expert3 dùng `load_best_checkpoint`; Expert4 dùng `load_finetuned`/`load_trainable_state_dict`. Xác nhận mọi expected fine-tuned parameter đã được restore và không có unexpected keys. Shape mismatch phải lỗi.
5. **Output parity:** cùng input samples/preprocessing như standalone; forward finite; so sánh score (E1 `[B]`, E2/E3 `[B,1]`, E4 `[B]`) và feature values với standalone reference trong tolerance định trước. Check Expert3 `D`, `5D`, `concat_assignment`; xác nhận output MoE shape.
6. **Freeze verification:** sau nạp, gọi eval, đóng băng tất cả parameters; ghi snapshot hash/clone parameter; chạy forward/gating-only backward và xác nhận không tham số expert nào nhận grad hoặc đổi giá trị.
7. **Resource/offline behavior:** xác nhận tokenizer/text encoder M2D đã có cache/local files; không cho phép tải ngầm Hub trong lần test. Xác nhận model không yêu cầu checkpoint không nằm trong inventory.

Các phép load, forward và gradient cần fine-tuned checkpoints; identity tests cần exact base resources. Chưa có actual fine-tuned artifact để xác nhận.

## 11. Rủi ro và blocker còn lại

- Fine-tuned checkpoints chưa tải về: không thể xác nhận contents, metadata, tensor keys, dimensions hay prediction parity.
- Expert3 checked-in config trỏ tới `expert3_finetune/weights/checkpoint-30.pth`, nhưng inventory thấy file tại repository `weights/checkpoint-30.pth`; cần truyền override đúng hoặc xác nhận layout trước khi dựng.
- Expert1/2 raw best checkpoints không lưu schema version, base hash hay training config; cần external run manifest để loại trừ checkpoint nhầm expert/run.
- Expert2 full fine-tuned state cho phép strict load nhưng không chứng minh nó khớp chính xác pretrained base đã dùng; source hiện không lưu fingerprint trong raw best payload.
- Expert3 checkpoint loader xác minh names nhưng không preflight shapes; behavior của `copy_` với shape không khớp cần xác nhận, kể cả trường hợp shape broadcast được. `weights_only=True` fallback trên PyTorch cũ dùng default pickle loader.
- Expert4 best payload không có kind discriminator và inference không so sánh saved config hoặc fingerprint của CSVs (chỉ pretrained files/source); caller phải chọn best path và xác nhận expected run.
- Expert4 CLAP loader cố ý `strict=False`; source bảo vệ required text branch/projection keys, nhưng không reject mọi unexpected key. Không dùng nó như full CLAP load.
- `MoescoreExpert3`/inference source imports `MOESCORE...`; package-root invocation may need explicit package context. Đây là import concern cần giải quyết khi integration/runtime check, không đổi trong audit.
- Expert3 pretrained text encoder uses `transformers.from_pretrained`; cần model/tokenizer cache và chế độ offline được chuẩn bị trước, tránh tải tự động.
- `torch.load(weights_only=False)` xuất hiện ở MGA/M2D pretrained và Expert4 CLAP/resume; chỉ dùng các artifacts tin cậy.

## 12. Đề xuất cho LOCAL-02.3

- Chốt chính xác contract input batch/raw caption/audio resampling và shape score/features cho adapter; checkpoint loading phụ thuộc class/factory và nguồn resource đã xác định ở đây.
- Chốt adapter config per expert: base paths, fine-tuned best paths, Expert2 freeze policy, Expert3 concat assignment và dynamic D, Expert4 resource fingerprints/source directory.
- Chốt tên/key namespace của từng adapter trước khi quyết định strict-load vào current wrapper; Expert2 và Expert4 có top-level import namespace riêng.
- Chốt cách biểu diễn expert features và tổng feature dimension/gate input. Expert3 dynamic D ảnh hưởng gate; không thể suy từ checkpoint path hoặc legacy 7168.
- Chốt fail-fast behavior nếu pretrained file/hash/source/config thiếu hoặc khác. Không cho fallback âm thầm sang pretrained-only state.
- Giữ tách biệt expert restoration khỏi MoE freeze policy: restore fine-tuned values trước, sau đó freeze tất cả expert parameters.

## 13. Tham chiếu nguồn

- Expert1 save/init: `expert1_finetune/Train_Expert1.py:L98-L137,L147-L180,L292-L310`; generic loader `model/MOE_model.py:L5-L7,L33-L58`.
- Expert2 base + wrapper: `expert2_finetune/expert2_model.py:L12-L99`; config `expert2_finetune/expert2_finetune.yaml:L1-L20`; training/eval/save/resume `expert2_finetune/Train_Expert2.py:L103-L115,L127-L175,L202-L218`.
- Expert2 architecture: `expert2_finetune/MGA-CLAP/settings/inference_example.yaml:L1-L36`; `expert2_finetune/MGA-CLAP/models/ase_model.py:L95-L144`.
- Expert3 model/hash/dim: `expert3_finetune/XACLE_2027_baselinemodel/models/moescore_expert3.py:L25-L59,L98-L172`; PortableM2D loads `weights_only=False`: `.../models/m2d/vendor/portable_m2d.py:L266-L275,L545-L601`.
- Expert3 checkpoint schema/load: `expert3_finetune/XACLE_2027_baselinemodel/utils/expert3_checkpoint.py:L13-L93,L130-L156`; trainer `.../train_expert3.py:L386-L414,L483-L493`; inference `.../inference_expert3.py:L49-L56`.
- Expert3 config path/hash: `expert3_finetune/XACLE_2027_baselinemodel/train_expert3.py:L32,L45-L65`; `.../configs/expert3_moescore.json:L15-L16,L40`.
- Expert4 base loaders: `expert4_finetune/model.py:L16-L18`; `audio_encoder.py:L21-L38`; `text_encoder.py:L8-L29,L62-L109`.
- Expert4 state/fingerprints/resume: `expert4_finetune/trainer.py:L30-L48,L120-L160,L224-L265,L320-L340`; inference `expert4_finetune/inference.py:L50-L59`; defaults `expert4_finetune/expert4_colab.yaml:L2-L12`.
- Legacy MOE loader: `model/MOE_model.py:L5-L7,L28-L60`.

## Kết luận

**PASS — STATIC AUDIT COMPLETE.** Save/load formats and restoration sequences are documented from source; local inventory confirms pretrained files exist and fine-tuned artifacts are absent. This status is not proof that any fine-tuned checkpoint loads. All actual artifact, tensor-shape and inference checks remain pending until the external fine-tuned checkpoints are obtained. Không có production code hay weight file nào được sửa/đọc nội dung.
