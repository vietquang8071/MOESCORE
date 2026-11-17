# Báo cáo adapter M2D-LoRA E34

## Kết quả

Đã thêm adapter suy luận độc lập `M2DLoRAAdapter`, kế thừa `ExpertAdapter`. Adapter giữ nguyên đường suy luận E34: M2D-CLAP tạo embedding audio/text 768 chiều; `inter_feats()` tạo vector tương tác 3073 chiều; head E34 gồm 10 logit CORAL và một đầu ra hồi quy quyết định điểm số. Điểm được khôi phục theo `select_on`/`mix` và `label_stats` trong checkpoint, sau đó đổi sang thang MOESCORE bằng `(score_raw - 5) / 5`. Không dùng cosine làm điểm dự đoán và không dùng nhãn trong suy luận.

Đối chiếu AST cho thấy `PortableM2D`, `BertXEncoder`, `get_text_encoder` và các thành phần runtime liên quan trong notebook trùng với runtime PortableM2D đã có trong MOESCORE. Vì vậy adapter tái sử dụng mã nguồn này, không chạy notebook và không đưa thêm bản sao runtime M2D. Checkpoint được đặt qua symlink tạm trong thư mục mang tên model mà PortableM2D yêu cầu; nội dung checkpoint không bị sửa.

## Kiểm tra checkpoint và suy luận thật

| Thành phần | Trạng thái | Chi tiết |
|---|---|---|
| Backbone M2D | PASS | Đã nạp `weights/pretrained/checkpoint-30.pth`; PortableM2D báo các khóa khớp đầy đủ. Sample rate của model là 16 kHz. |
| E34 `best.pt` | PASS | Đã xác nhận payload có `adapter`, `head`, `config`, `label_stats`; khôi phục đủ 24 tham số LoRA theo tên và shape chính xác; head nạp strict với 6 tensor. |
| Cặp dữ liệu | PASS | `train/00000.wav`, caption: “A water vehicle travels through the water with wind noise and a man talking”. |
| Suy luận | PASS | Score chuẩn hóa `0.4165346026420593`, shape `(1, 1)`; features shape `(1, 3073)`, float32 trên CPU. Cả model và head ở eval mode, mọi tham số đã freeze. |

Số adapter suy luận thật thành công trong milestone này: **1/1**. Score không bị clip; mọi dự đoán ngoài khoảng danh nghĩa vẫn được giữ theo hành vi E34.

Đường dẫn thực tế:

- Backbone: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/pretrained/checkpoint-30.pth`
- Fine-tuned E34: `/home/vitquay1708/Working_Space/Research/2027/code/MOESCORE/weights/best.pt`
- Audio: `/home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/wav/train/00000.wav`
- Text encoder cache: model `google-bert/bert-base-uncased` đã có trong Hugging Face cache cục bộ. Chạy ở chế độ offline, không tải model.

Checkpoint `best.pt` được đọc với `torch.load(weights_only=True)`. PortableM2D nguồn gốc E34 tự nạp checkpoint backbone bằng `weights_only=False`; file backbone là checkpoint cục bộ đã chỉ định rõ trong cấu hình. Payload Colab `checkpoint_path` chỉ là metadata và không được dùng thay cho đường dẫn backbone cục bộ.

## Môi trường và kiểm thử

- Python: `/home/vitquay1708/miniconda3/envs/nlp/bin/python` (Python 3.10).
- PyTorch `2.14.1+cu130`; CUDA không khả dụng, nên chạy trên CPU.
- Runtime inference đã dùng Transformers `4.57.6`, timm `1.0.28`, nnAudio `0.3.4`, einops `0.8.2` và cache BERT cục bộ.
- Kiểm thử tập trung: **8/8 PASS**, chạy bằng `python -m unittest discover -s tests -p 'test_m2d_lora_adapter.py' -v`.
- Các test bao phủ import lazy, layout interaction, khôi phục CORAL/reg/mix và chuẩn hóa score, target LoRA, kiểm tra thiếu/thừa/sai shape tham số, frozen/eval, batch output và HF cache path.
- Môi trường không cài `pytest`; không cần cài thêm để chạy bộ test mới vì test dùng `unittest` tích hợp.

## Cách dùng

```python
from model.expert_adapters.m2d_lora_adapter import M2DLoRAAdapter

config = {
    "runtime": {
        "device": "auto",  # hoặc "cpu" / "cuda"
        "offline": True,
        "cache_dir": None,
    },
    "base_checkpoint": "/absolute/path/to/checkpoint-30.pth",
    "finetuned_checkpoint": "/absolute/path/to/best.pt",
}

adapter = M2DLoRAAdapter.from_config(config)
score, features = adapter(batch)
print(score.shape)     # (B, 1)
print(features.shape)  # (B, 3073)
```

Đầu vào dùng `batch["expert3_wavs"]` với shape `[B,1,160000]`, float32, 16 kHz và `batch["captions_raw"]`. Adapter chuyển audio đến device của model, giữ nguyên caption đầu vào, và không yêu cầu labels hay generator metadata. `runtime.cache_dir` có thể là HF_HOME thông thường hoặc thư mục cache trực tiếp; adapter xử lý thư mục con `hub` mà BERT loader E34 cần.

## Tệp và tích hợp về sau

Tệp được tạo trong milestone này:

- `model/expert_adapters/m2d_lora_runtime.py`
- `model/expert_adapters/m2d_lora_adapter.py`
- `tests/test_m2d_lora_adapter.py`
- `docs/Ex1/m2d_lora_adapter_report.md`

Không sửa adapter Top 1/Top 2, registry, gate, feature store, notebook hoặc checkpoint. Khi tích hợp milestone sau, đăng ký khóa `m2d_lora` trỏ tới module `model.expert_adapters.m2d_lora_adapter`, class `M2DLoRAAdapter`, với `feature_dim=3073`; cập nhật thứ tự/contract MoE trong đợt tích hợp đó. Factory độc lập hiện đã nhận cấu hình trực tiếp và cũng hỗ trợ section `experts.m2d_lora` cùng `runtime`.

Không còn blocker cho adapter độc lập; inference thật đã PASS trên cặp train nói trên.
