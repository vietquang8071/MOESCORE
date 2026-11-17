# Báo cáo inference thật: bốn adapter pretrained Top 2

**Ngày chạy:** 17/11/2026  
**Repository:** MOESCORE  
**Phạm vi:** nạp pretrained và inference trên một cặp audio–caption; không train, không trích xuất toàn bộ dataset, không tích hợp gate chín expert.

## Tóm tắt kết quả

Đã chạy tuần tự script với cùng một cặp dữ liệu train cho cả bốn adapter. Lượt `--adapter all` mới nhất đạt **4/4 PASS**, exit code `0`. MS-CLAP dùng checkpoint local `CLAP_weights_2023.pth` và đã trả embedding/score thật. Lần thử trước bằng checkpoint Hub mặc định vẫn thất bại do DNS tới Hugging Face; lần inference thành công dùng lại `model_fp` để nạp checkpoint local.

| Expert | Checkpoint loaded | Cosine score | Score shape | Feature shape | Status |
|---|---|---:|---|---|---|
| M2D-CLAP | Có: `weights/pretrained/checkpoint-30.pth`; PortableM2D báo mọi key khớp | `0.3516223431` | `(1, 1)` | `(1, 1536)` | PASS |
| MS-CLAP 2023 | Có: `weights/pretrained/CLAP_weights_2023.pth` qua `model_fp` | `0.6419963837` | `(1, 1)` | `(1, 2048)` | PASS |
| LAION-CLAP | Có: `weights/pretrained/laion-clap.pt`, nạp bằng `CLAP_Module.load_ckpt` | `0.5418541431` | `(1, 1)` | `(1, 1024)` | PASS |
| MGA-CLAP | Có: `weights/pretrained/mga-clap.pt`; loader khớp 415/415 tensor của model | `0.4798489511` | `(1, 1)` | `(1, 2048)` | PASS |

Với cả bốn model, score và features là float32, finite, cùng trên CPU; score nằm trong `[-1,1]`; model ở eval mode và mọi parameter đều frozen. Độ rộng mỗi nhánh được xác nhận từ feature thực tế: M2D `768+768`, MS-CLAP `1024+1024`, LAION `512+512`, MGA `1024+1024`.

Score MS-CLAP thay đổi giữa các lần chạy: người dùng báo `0.6286894083`, lần chạy riêng trong môi trường này cho `0.6202629805`, và lượt tổng hợp cuối cho `0.6419963837`. Package MS-CLAP chọn ngẫu nhiên đoạn audio 7 giây từ WAV dài 9.844 giây; các lần chạy khác crop có thể cho cosine khác. Các lần chạy đều giữ cùng file/caption, checkpoint, shape và validation output.

Lượt tổng hợp cuối chạy `--adapter all` trả exit code `0`; các lần chạy trước khi GPT-2 cache sẵn sàng từng trả `1` với MS-CLAP BLOCKED. Trạng thái hiện tại là cả bốn adapter đều PASS.

## Dữ liệu dùng chung

- Dataset root được script phát hiện dưới thư mục truyền vào: `XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset/`.
- Split: `train`; metadata: `meta_data/train_average.csv`.
- Audio: `wav/train/00000.wav`, mono 16 kHz, thời lượng `9.844 s`.
- Caption cùng hàng: `A water vehicle travels through the water with wind noise and a man talking`.
- Dùng preprocessing của `UnifiedXACLEDataset`/`collate_unified`: waveform 16 kHz cho M2D, 32 kHz cho MGA, 48 kHz cho LAION. MS-CLAP nhận đường dẫn WAV gốc và API package tự áp dụng preprocessing 44,1 kHz/7 giây. Không đưa label vào batch inference.

## Môi trường và tài nguyên

- Python: `/home/vitquay1708/miniconda3/envs/nlp/bin/python`, Python 3.10.21.
- PyTorch: `2.14.1+cu130`; CUDA không khả dụng (`torch.cuda.is_available() == False`, 0 GPU); device chạy là CPU.
- Do không có GPU khả dụng, lượt này không xác minh runtime CUDA; script chuyển waveform tới device đã chọn khi có CUDA.
- Transformers `4.57.6`; `msclap` `1.3.4`; `laion-clap` `1.1.7`; `timm` `1.0.28`; `einops` `0.8.2`; `nnAudio` `0.3.4`; `torchlibrosa` `0.1.0`; PyYAML `6.0.3`.
- Checkpoint local có sẵn: `checkpoint-30.pth` (M2D), `CLAP_weights_2023.pth` (MS-CLAP), `laion-clap.pt` (LAION), `mga-clap.pt` (MGA). Không giải nén hoặc tải checkpoint trong lượt này.
- Checkpoint LAION là bundle native có `state_dict`, không phải checkpoint Transformers. Nó được nạp bằng API tương ứng của package LAION với HTSAT-tiny, non-fusion; hai nhánh trả embedding joint 512 chiều. Đây là checkpoint dưới `weights/pretrained`, không phải checkpoint fine-tuned Top 1.
- MGA restore từ YAML nguồn `expert2_finetune/MGA-CLAP/settings/inference_example.yaml`, đặt `audio_args.sr=32000`. Cảnh báo loader ghi nhận một key checkpoint dư, nhưng mọi 415 tensor của model đều được khớp; inference thành công.
- Script cấp thư mục Numba cache tạm trong lúc dựng MS-CLAP/MGA để tránh lỗi cache của `librosa` trong môi trường hiện tại; thư mục được dọn khi model kết thúc.

### Ghi chú về cách nạp MS-CLAP

Đã thử cách Hub mặc định không truyền `model_fp`; package gọi `hf_hub_download("microsoft/msclap", "CLAP_weights_2023.pth")`, nhưng request `https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth` thất bại vì DNS. Không có checkpoint nào được tải bằng lượt đó.

Sau đó working copy dùng `model_fp=str(checkpoint)` lại. Adapter nạp `weights/pretrained/CLAP_weights_2023.pth` và inference offline thành công; cache hiện có GPT-2 resources cần thiết cho text encoder. Đây là cấu hình của lượt `--adapter all` PASS 4/4.

## Script và cách chạy lại

Script hỗ trợ tự tìm layout dataset lồng nhau, chỉ định adapter hoặc chạy tuần tự tất cả, cấu hình đường dẫn project/checkpoint/dataset, chọn split/index/device và trả mã khác 0 nếu có FAIL/BLOCKED.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/vitquay1708/miniconda3/envs/nlp/bin/python \
scripts/smoke_test_top2_real_inference.py \
  --dataset-root /path/to/XACLE_Challenge_2027_development_dataset \
  --adapter all
```

Chọn riêng một adapter, ví dụ:

```bash
python scripts/smoke_test_top2_real_inference.py \
  --dataset-root /path/to/XACLE_Challenge_2027_development_dataset \
  --adapter top2_mga --device cpu
```

## Kiểm tra đã thực hiện

- Chạy inference thật tuần tự trên cùng cặp train bằng `--adapter all`: **4 PASS, exit code `0`**.
- Chạy lại riêng MS-CLAP theo lệnh bạn gửi: PASS, score `0.6202629805`; lần chạy độc lập trước đó do người dùng gửi score `0.6286894083`.
- Thử dựng MS-CLAP online theo Hub mặc định (không truyền `model_fp`): DNS thất bại khi truy cập checkpoint Hub; sau đó nạp checkpoint local thành công.
- `python -m unittest discover -s tests -p test_top2_adapters.py -v`: **7/7 pass**, gồm kiểm tra mock contract/freeze, factory dùng checkpoint local, và M2D symlink parent.
- `py_compile` cho script, bốn adapter và test: pass.
- `scripts/smoke_test_top2_real_inference.py --help`: pass.
- Không chạy toàn bộ dataset hoặc bộ test toàn repository.

## Files tạo/sửa trong lượt xác minh

- Tạo `scripts/smoke_test_top2_real_inference.py`.
- Sửa `model/expert_adapters/top2_m2d_adapter.py`: giữ đường dẫn logic qua symlink để PortableM2D đọc tên thư mục model.
- `model/expert_adapters/top2_msclap_adapter.py` hiện truyền checkpoint local qua `model_fp`; lần thử Hub mặc định trước đó không tải được do DNS.
- Sửa `model/expert_adapters/top2_laion_adapter.py`: hỗ trợ checkpoint native LAION local bằng `CLAP_Module` và embedding paired.
- Sửa `tests/test_top2_adapters.py`: regression cho factory MS-CLAP/LAION và symlink checkpoint M2D.
- Tạo báo cáo này và cập nhật `docs/Ex1/bao-cao-four-top2-adapters.md`.

Các adapter Top 2 còn lại, adapter Top 1, registry/gate, trainer, feature store và repository `EnsembleSVR-XACLE` không bị sửa trong lượt này. Thay đổi tracked có sẵn ở `expert1_finetune/Train_Expert1.py` được giữ nguyên.
