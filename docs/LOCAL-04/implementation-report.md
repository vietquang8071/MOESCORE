# LOCAL-04 — Báo cáo triển khai Expert 1 Adapter (LAION-CLAP)

## 1. Tóm tắt

Đã triển khai `Expert1Adapter` dựa trên class fine-tune gốc, dùng chung `ExpertAdapter` từ LOCAL-03. Adapter ánh xạ batch, gọi encoder đúng một lần, giữ nguyên dot product, ghép `[audio, text]`, chuẩn hóa shape score thành `[B,1]`, kiểm tra output bằng validator chung, rồi đóng băng và giữ expert ở eval mode.

Các test giả và toàn bộ regression suite đều pass. Checkpoint fine-tuned do người dùng cung cấp hiện có tại `weights/finetuned_weights/expert1_best.pt` (606 MB); pretrained LAION-CLAP có tại `weights/laion-clap.pt` (1.8 GB). Chỉ kiểm tra metadata file, không deserialize, khởi tạo mô hình hay chạy inference. Vì vậy việc restore checkpoint thật và parity số học còn chờ xác minh riêng.

**Implementation:** `IMPLEMENTED_AND_UNIT_TESTED`  
**Runtime:** `REAL_CHECKPOINT_RESTORED` chưa xác minh; `NUMERICAL_PARITY_VERIFIED` chưa xác minh.

## 2. Trạng thái repo ban đầu

- Branch: `main`; hai remote đang có sẵn và không bị thay đổi.
- Trước khi sửa, `git status --short` chỉ có các thư mục/file chưa được track: LOCAL-03 (`configs/`, `model/expert_adapters/`, `tests/`), các báo cáo LOCAL-02/03, `docs/dataset-location.md` và `weights/`. Không có file tracked nào báo modified.
- Bộ test nền trước thay đổi: **41/41 PASS** bằng `/home/vitquay1708/miniconda3/envs/nlp/bin/python3` (Python 3.10, PyTorch 2.14.1+cu130).
- Test nền có cảnh báo đã tồn tại từ dependency: `timm.models.layers` deprecated và thông báo Hugging Face Hub chưa xác thực. Không có model/checkpoint nào được khởi tạo trong lần baseline.
- Trạng thái checkpoint trong audit LOCAL-02.2 đã cũ so với working tree hiện tại: audit ghi artifact fine-tuned chưa có; lần này `ls -lh` xác nhận file Expert1 ở đường dẫn người dùng đưa. Nội dung checkpoint chưa được kiểm tra.

## 3. Hợp đồng Expert 1 trong source

- `CLAPAudioTextEncoder.__init__(checkpoint, device)` tạo `laion_clap.CLAP_Module(enable_fusion=False, device=...)`, sau đó gọi `load_ckpt(checkpoint)`. Source đóng/mở gradient theo nhóm tham số khi fine-tune tại [Train_Expert1.py](../../expert1_finetune/Train_Expert1.py#L98) (L98–115).
- `CLAPAudioTextEncoder.forward(audio, captions)` gọi API audio và text rồi trả cặp embedding. `Expert1.forward(batch)` truyền `clap_wavs`, `caption_tokens`, tính `(audio_features * text_features).sum(dim=-1)` và chỉ trả `[B]` ([Train_Expert1.py](../../expert1_finetune/Train_Expert1.py#L125), forward L117–137).
- Training entry point khởi tạo `Expert1(args.checkpoint, device)` ([Train_Expert1.py](../../expert1_finetune/Train_Expert1.py#L216)). Best checkpoint được lưu từ toàn bộ `expert.state_dict()` dưới dạng raw tensor state dict, không bọc trong khóa `state_dict` hay `model_state_dict` ([Train_Expert1.py](../../expert1_finetune/Train_Expert1.py#L297), save L296–300).
- Checkpoint được khởi tạo lại theo thứ tự: tạo Expert1 để nạp pretrained CLAP, rồi strict-load state fine-tuned. Checkpoint fine-tuned không có fingerprint của pretrained base; strict-load không chứng minh nguồn gốc pretrained.
- Input gốc tại ranh giới encoder là waveform và danh sách caption thô. Dataset gốc trả `[B,1,480000]` float32 48 kHz và `caption_tokens` thực chất là chuỗi; adapter nhận `expert1_wavs`/`captions_raw`, không resample, cắt hay token hóa lại.

**Đối chiếu dependency LAION-CLAP trong môi trường `nlp`:** source của `CLAP_Module` đưa `joint_embed_shape` từ model core ra ở `model.model`; constructor `CLAP` có thuộc tính `joint_embed_shape` (mặc định 512) và projection audio/text đều chiếu về chiều này. File `HTSAT-tiny.json` không override thuộc tính. Vì vậy adapter lấy chiều từ model đã dựng, không mã hóa cứng. Trong phiên bản source cài đặt hiện tại, `get_audio_embedding()` và `get_text_embedding()` đều gọi `F.normalize(..., dim=-1)` (`laion_clap/clap_module/model.py:L698–742`); điểm này bổ sung bằng chứng mà các audit trước chưa thấy khi chỉ xét wrapper. Đây là bằng chứng source của dependency cục bộ; chưa chạy inference với artifact thật.

Pretrained `load_ckpt()` của dependency gọi loader dùng `torch.load(..., weights_only=False)` (`laion_clap/clap_module/hook.py:L112–114`, `clap_module/factory.py:L53–70`). Chỉ dùng pretrained file đáng tin cậy. Fine-tuned raw state dict do adapter nạp riêng bằng `weights_only=True`.

## 4. Kiến trúc adapter đã triển khai

File mới [expert1_adapter.py](../../model/expert_adapters/expert1_adapter.py#L1) định nghĩa `Expert1Adapter(ExpertAdapter)`.

- Constructor nhận `model: nn.Module` đã dựng sẵn và `embedding_dim` tùy chọn; không tạo bản sao Expert1. Model được đăng ký thành child module, mọi parameter bị freeze và model được đặt eval (class L84, constructor L100–132, freeze L279–281).
- `feature_dim` của base là `2 * embedding_dim`. Khi không truyền chiều, adapter đọc `joint_embed_shape` theo hierarchy gốc `Expert1 → CLAPAudioTextEncoder → CLAP_Module → CLAP`. Nếu caller truyền chiều, nó phải là số nguyên dương và khớp metadata nếu metadata tồn tại (adapter L100–141; base property L85–91).
- Concrete adapter cài `_forward(batch)` để giữ luồng kiểm tra `forward()` cuối cùng của base; base gọi `_forward()` rồi validate tuple bằng `validate_expert_output` ([base.py](../../model/expert_adapters/base.py#L94), forward/validator L94–117).
- Adapter thêm **không parameter/trainable layer**. `train(mode)` cho phép trạng thái của adapter thay đổi nhưng đặt lại toàn bộ expert con về eval, kể cả khi parent gọi `moe.train()` (adapter L283–286).
- Public import hiện tại: `from model.expert_adapters.expert1_adapter import Expert1Adapter`. Không sửa `__init__.py`, nên `import model.expert_adapters` vẫn không kéo LAION hoặc source model vào process.

## 5. Ánh xạ input

| Batch chung | Input encoder gốc | Kiểm tra |
|---|---|---|
| `expert1_wavs` | Tham số `audio` cho `CLAPAudioTextEncoder` (gốc đặt tên batch key là `clap_wavs`) | Tensor float32 `[B,1,480000]`, `B>0`, hữu hạn, một channel, đúng device với model |
| `captions_raw` | Tham số `captions` cho encoder (gốc đặt tên batch key là `caption_tokens`) | Sequence Python gồm đúng B chuỗi; không được truyền token ID; caption rỗng/whitespace-only bị từ chối theo điều kiện dataset gốc |

Chuỗi được truyền nguyên văn; `.strip()` chỉ dùng để phát hiện caption rỗng, không dùng để biến đổi nội dung. Encoder gốc tự squeeze channel trước API LAION và tự tokenize. Adapter không sửa waveform; producer sau này phải bảo đảm 48 kHz mono, center crop khi dài và repeat/zero-pad khi ngắn như validation preprocessing gốc.

## 6. Hợp đồng output

- `score`: `[B,1]`, được tạo từ đúng biểu thức gốc `(audio_embedding * text_embedding).sum(dim=-1)` rồi thêm chiều cuối. Không calibration, clip, sigmoid/tanh, temperature hoặc đổi thang điểm.
- `features`: `[B,2*D_CLAP]`, tạo bằng `torch.cat([audio_embedding, text_embedding], dim=-1)`. Thứ tự audio trước text sau; không chuẩn hóa hoặc chiếu feature ở adapter.
- `D_CLAP` lấy từ `joint_embed_shape` của model instance; trong source LAION-CLAP cài tại môi trường hiện tại, giá trị dự kiến là 512, nên `feature_dim` dự kiến 1024. Đây là số chiều suy ra từ source/config đang cài, **chưa được xác nhận bằng output của model đã nạp checkpoint thật**. Chiều embedding thực tế được kiểm tra mỗi forward.
- Validator LOCAL-03 xác nhận score chính xác `[B,1]`, feature `[B,feature_dim]`, cùng batch/device, floating dtype và giá trị hữu hạn. Adapter không tự reshape output sai ngoài việc thêm chiều cho score gốc.

## 7. Trích feature một lượt

`_forward()` gọi duy nhất `self.model.model(audio, captions)` — tức `CLAPAudioTextEncoder.forward()` — rồi dùng chính hai tensor trả về để tính score và concat feature (adapter L357–373). Không gọi `Expert1.forward()` trước đó và không gọi encoder lần hai. Test `test_returns_original_dot_score_and_audio_then_text_features` kiểm tra call count bằng 1, dot product tham chiếu độc lập, thứ tự feature, caption giữ nguyên và batch B=1,2,4.

## 8. Chiều feature

Nguồn tin cậy là `CLAP` core sau khi original Expert1 được dựng, cụ thể `joint_embed_shape`; adapter không lấy chiều từ tên checkpoint, không đoán hằng số. Nếu metadata không có, caller phải truyền `embedding_dim` dương có provenance tin cậy. Output embedding đầu tiên được đối chiếu với chiều đã khai báo; sai chiều gây lỗi trước khi gate được dùng. Factory sau khi strict restore sẽ dựng adapter với metadata từ chính model.

## 9. Khôi phục checkpoint

API:

```python
adapter = Expert1Adapter.from_finetuned(
    pretrained_checkpoint="/absolute/path/weights/laion-clap.pt",
    finetuned_checkpoint="/absolute/path/weights/finetuned_weights/expert1_best.pt",
    device="cpu",  # hoặc "cuda:0" / "auto"
)
```

Đường dẫn phải là absolute đã resolve; `from_config(resolved_config)` nhận toàn bộ config đã qua `load_moe_config()` và `resolve_moe_paths()` hoặc section `expert1` đã resolve. Nó xác nhận `enable_fusion: false` và chỉ kiểm tra hai resource Expert1; không gọi full `validate_moe_resources()` nên placeholder/missing checkpoint của E2–E4 không chặn riêng E1 (adapter `from_finetuned` L174–231; `from_config` bắt đầu L233).

Factory kiểm tra file tồn tại/đọc được trước, import `Train_Expert1.Expert1` chỉ khi gọi factory, dựng model gốc trên CPU (constructor tự load pretrained đúng một lần, fusion tắt), nạp raw fine-tuned state với `torch.load(map_location="cpu", weights_only=True)` và `load_state_dict(strict=True)`, rồi freeze/eval và chuyển đến device đích. `from_config()` chuyển `runtime.offline` vào factory; mặc định `offline=True` tạm đặt `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` và cờ runtime của Hugging Face quanh import/khởi tạo, sau đó khôi phục trạng thái môi trường. Khi offline, tokenizer/model text phải có trong cache; thiếu cache sẽ fail thay vì tải. Bundle có key lạ/thiếu, tensor shape sai, format không phải raw tensor state dict, file hỏng hoặc thiếu resource đều fail rõ ràng. Không có fallback pretrained-only (adapter loader L145–172, factory L174–231, offline helper L40–81).

Fine-tuned file hiện đã có trên local, nhưng task này không mở/deserialise file. `weights_only=True` đã được kiểm bằng state dict tổng hợp nhỏ, không được tính là kiểm tra artifact thật. Pretrained identity vẫn chưa chứng minh được vì raw Expert1 state dict không lưu SHA/base metadata.

## 10. Trạng thái đóng băng

Adapter gọi `requires_grad_(False)` và `.eval()` trên toàn expert sau khi nhận model. Override `train()` gọi hành vi chuẩn của `nn.Module` rồi đưa child expert trở lại eval; adapter có thể ở `training=True` trong lúc child Expert1 và dropout con luôn `training=False`. `adapter.eval()` vẫn hoạt động. Adapter không bọc `no_grad()` cưỡng bức; MoE sau này có thể dùng `torch.no_grad()` quanh expert, còn gate được tính ngoài context đó.

## 11. Unit/regression test

| Lệnh (đều chạy qua Conda `nlp`) | Kết quả |
|---|---|
| `python3 -m unittest discover -s tests -p "test_expert1_adapter.py" -v` | **PASS**, 22 test |
| `python3 -m unittest discover -s tests -p "test_adapter_base.py" -v` | **PASS**, 13 test |
| `python3 -m unittest discover -s tests -p "test_moe_config.py" -v` | **PASS**, 21 test |
| `python3 -m unittest discover -s tests -p "test_expert_import_isolation.py" -v` | **PASS**, 6 test |
| `python3 -m unittest discover -s tests -p "test_imports.py" -v` | **PASS**, 1 test; còn warning `timm`/Hub như baseline |
| `python3 -m unittest discover -s tests -p "test_*.py" -v` | **PASS**, 63 test, 0 failure |
| `python3 -m compileall -q model/expert_adapters tests/test_expert1_adapter.py` | **PASS** |

Lệnh trên được gọi với `conda run -n nlp --no-capture-output`. Test mới phủ import nhẹ, B=1/2/4, score/feature đúng, một encoder call, input xấu, dimension metadata, finite/device/dtype, config path scope, strict raw state restore và train/eval freeze. Không test nào cần checkpoint thật.

Lần chạy TDD ban đầu fail đúng vì module adapter chưa tồn tại. Một fixture batch-mismatch ban đầu chưa tạo output sai batch do stub cắt đúng batch đầu vào; nguyên nhân nằm ở fixture. Fixture được sửa để giữ batch sai cố ý. Sau đó test phát hiện `from_config()` chưa truyền `runtime.offline`; đã thêm offline context có kiểm tra env và trạng thái Hugging Face được cache. Bộ test cuối cùng pass.

## 12. Quy trình xác minh checkpoint thật (chưa chạy)

Khi cho phép runtime verification và cache text encoder cục bộ sẵn sàng:

1. Đọc config bằng `load_moe_config`, resolve bằng `resolve_moe_paths`; ghi đè riêng `experts.expert1.finetuned_checkpoint` thành đường dẫn tuyệt đối `weights/finetuned_weights/expert1_best.pt` nếu YAML vẫn dùng placeholder.
2. Gọi `Expert1Adapter.from_config(resolved, device="cpu")` để kiểm tra restore strict trên CPU trước; chỉ chuyển GPU nếu cần.
3. Xác nhận mọi parameter `requires_grad=False`, adapter/model `.training=False`, `embedding_dim` đọc từ CLAP core và `feature_dim == 2*embedding_dim`.
4. Dùng cùng batch xác định `[B,1,480000]` và raw captions cho model gốc và adapter trong eval/no-grad. So sánh `adapter_score.squeeze(-1)` với `original_expert(batch_goc)` trước; chọn tolerance trước khi chạy.
5. Dùng encoder hook hoặc direct encoder reference để so embeddings và feature order; xác nhận một lần gọi encoder trong mỗi `adapter(batch)`.
6. Kiểm tra B=1 và B>1, dtype/device, hữu hạn; ghi lại phiên bản LAION, fingerprint pretrained nếu có manifest đáng tin, không suy fingerprint từ việc state dict load thành công.

Không chạy bước nào ở đây. Khởi tạo Expert1 sẽ đọc các checkpoint lớn và có thể cần tài nguyên tokenizer/model RoBERTa trong cache Hugging Face; cần bảo đảm chế độ offline/cached resources trong tiến trình xác minh để tránh tải ngoài ý muốn.

## 13. Rủi ro và giới hạn

- Checkpoint fine-tuned đã có local nhưng chưa được deserialize: **strict restore thật chưa xác minh**.
- Dự kiến D=512 theo source/config package đang cài; width output thực tế chưa được đo qua model.
- Normalization được xác nhận tĩnh từ source của LAION dependency cục bộ, nhưng embedding values và prediction thực tế chưa được so sánh.
- Chưa xác minh parity score standalone/adaptor bằng fine-tuned weights.
- Raw fine-tuned artifact không mang fingerprint pretrained base; strict key/shape match không chứng minh đúng pretrained file.
- LAION pretrained loader trong dependency hiện tại dùng `weights_only=False`; chỉ dùng artifact pretrained đáng tin cậy. Fine-tuned raw state của adapter dùng `weights_only=True`.
- Offline mode tạm thay đổi environment flags và Hugging Face runtime constant trong lúc model được khởi tạo; model construction nên được chạy tuần tự, không đồng thời với một luồng khác đang dùng Hub.
- `from_config` yêu cầu path đã resolve; hãy chạy schema validation/path resolution của LOCAL-03 trước. Chỉ kiểm tra tài nguyên E1, không xác nhận toàn bộ cấu hình E2–E4.
- Kiểm import Expert1 module nhẹ đã qua. Chưa khởi tạo model thật để xác nhận tokenizer/cache/model construction; đây là runtime-pending, không phải import blocker.

## 14. File tạo mới

- `model/expert_adapters/expert1_adapter.py`
- `tests/test_expert1_adapter.py`
- `docs/LOCAL-04/implementation-report.md`

Không sửa `model/expert_adapters/__init__.py` vì import trực tiếp submodule giữ package nhẹ. Không sửa bất kỳ file bị cấm nào, source Expert1, cấu hình, loss, train loop, entropy coefficient `0.2`, remote hay checkpoint. Không tải/nạp checkpoint, không chạy training/inference, không commit/push. Các file untracked có trước được giữ nguyên.

## 15. Sẵn sàng cho LOCAL-05

Có thể bắt đầu triển khai Expert2 Adapter dựa trên cùng base interface. LOCAL-04 đạt mức `IMPLEMENTED_AND_UNIT_TESTED`; trước khi tuyên bố tích hợp runtime hoàn tất, cần strict-restore Expert1 artifact thật, xác minh chiều feature thực tế và chạy score/feature parity trong eval mode. Checkpoint có mặt nên blocker hiện tại không phải thiếu file; xác minh runtime được để riêng theo giới hạn task.

**Trạng thái:** **PASS — IMPLEMENTED AND UNIT TESTED; RUNTIME PARITY PENDING.**
