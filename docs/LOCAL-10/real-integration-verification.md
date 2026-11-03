# Báo cáo xác minh tích hợp đầu ra thật của bốn mô hình chuyên gia với mạng cổng MoE

## 1. Tóm tắt

**Kết quả: PASS — đã xác minh luồng đầu ra thật của bốn mô hình chuyên gia đi vào mạng cổng MoE.**

Phân mức ghi trong JSON: `REAL_FOUR_EXPERT_OUTPUT_EXTRACTION=PASS`, `REAL_OUTPUT_ASSEMBLY=PASS`, `REAL_OUTPUT_GATING_FORWARD=PASS`, `REAL_OUTPUT_GATING_BACKWARD=PASS`. Chạy đồng thời cả bốn mô hình có trạng thái `NOT_ATTEMPTED_RESOURCE_RISK`.

Mỗi mô hình chuyên gia được dựng bằng hàm tạo hiện có với checkpoint tiền huấn luyện và checkpoint đã tinh chỉnh thật, rồi suy luận trên cùng bốn tệp WAV trong các tiến trình riêng. Điểm và đặc trưng CPU FP32 được ghép theo danh tính mẫu, kiểm tra bằng hàm ghép đầu ra LOCAL-08 và đưa vào chính `GatingNetwork` LOCAL-10. Ma trận điểm `[4,4]`, đặc trưng `[4,7936]`, trọng số `[4,4]` và dự đoán `[4,1]` đều hợp lệ. Công thức tổng có trọng số khớp tham chiếu với sai số tuyệt đối lớn nhất `0.0`.

Cặp E2/E4 cũng được dựng trong cùng tiến trình, cùng tồn tại và suy luận trên một mẫu chung; kết quả khớp với tệp kết quả độc lập. **Chưa xác minh bốn mô hình thật cùng tồn tại một lúc.** Không có huấn luyện, cập nhật checkpoint hay hiệu chỉnh điểm.

## 2. Môi trường

| Thành phần | Giá trị quan sát |
|---|---|
| Conda | `nlp` |
| Python | `3.10.21` |
| PyTorch | `2.14.1+cu130` |
| CUDA runtime | `13.0` |
| GPU | NVIDIA GeForce GTX 1650, `3,895,132,160` byte VRAM (~3.63 GiB) |
| Chế độ số học | FP32 |
| Dữ liệu/HF | Chạy ngoại tuyến; không tải tài nguyên |
| Tập dữ liệu | `/home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset` |

Các phiên bản thư viện chính trong JSON: `torchaudio 2.11.0+cu130`, `transformers 5.17.0`, `laion-clap 1.1.7`, `numpy 1.26.4`, `scipy 1.15.3`, `soundfile 0.14.0`.

## 3. Trạng thái repository trước và sau

Nhánh hiện tại là `main`; các kho từ xa không thay đổi. Trước khi bắt đầu, thư mục làm việc Git đã có thay đổi và nhiều tệp milestone chưa được theo dõi, gồm `M expert1_finetune/Train_Expert1.py`. Các mục này được giữ nguyên; không stage, commit hoặc push.

Mốc kiểm thử trước chỉnh sửa là **219 bài đạt**. Sau khi thêm 9 bài kiểm thử LOCAL-10.1, toàn bộ bộ kiểm thử đạt 228 bài.

## 4. Kiểm tra mã nguồn và giao diện

Trình xác minh sử dụng các thành phần có sẵn:

- Hàm tạo: `create_expert_adapter(...)` trong `model/expert_adapters/registry.py`.
- Thứ tự ổn định: `expert1`, `expert2`, `expert3`, `expert4`.
- Xử lý dữ liệu: `UnifiedXACLEDataset` và `collate_unified` trong `data/`.
- Ghép đầu ra: `validate_and_assemble_expert_outputs(...)` trong bộ đăng ký.
- Cổng và phép tính đầu ra: `GatingNetwork` cùng `MoEForwardIntegration.forward_from_expert_outputs(...)`.
- Bộ chuyển đổi nhận các trường `expert1_wavs`, `expert2_wavs`, `expert3_wavs`, `expert4_wavs`, `expert4_wav_lens` và `captions_raw`.

Các bộ chuyển đổi trả điểm `[B,1]` và đặc trưng tương ứng `[B,1024]`, `[B,2048]`, `[B,3840]`, `[B,1024]`. E4 chuyển điểm gốc `[B]` sang giao diện `[B,1]` mà không đổi giá trị.

Đã thêm `forward_from_expert_outputs(...)` vào `MoEForwardIntegration` để tái sử dụng hàm ghép đầu ra và mạng cổng với các đầu ra đã được xác thực. Trong bước ghép tuần tự, bốn bộ chuyển đổi giữ chỗ không được gọi và không tạo đầu ra chuyên gia; điểm/đặc trưng thật từ tệp kết quả là dữ liệu duy nhất đi vào mạng cổng. Cách này giữ nguyên phép tính và không đòi hỏi cả bốn mô hình cùng nằm trong bộ nhớ.

## 5. Mẫu thật được chọn

Trình xác minh chọn theo `seed=2027`, phân hoạch `train`, băm ổn định từ danh tính mẫu; thuật toán ưu tiên có mặt đủ nhóm âm thanh ngắn, đúng 10 giây và dài hơn 10 giây. Nhãn và trường `model_id` không tham gia xếp hạng chọn mẫu. Bảng kê dùng chung có SHA-256:

```text
da37ac6af4ff578fa2dbde1a583208f26a5f905a76e32e185e5666101e78404d
```

| Vị trí CSV (dòng, tính cả dòng tiêu đề) | Danh tính mẫu | Tệp | Âm thanh nguồn | Nhóm | Mô tả gốc |
|---:|---|---|---|---|---|
| 4669 | `train:04667.wav` | `04667.wav` | 16 kHz, đơn kênh, 146000 mẫu, 9.125 giây | Ngắn hơn 10 giây | Mô tả gốc được giữ nguyên trong bảng kê |
| 3826 | `train:03824.wav` | `03824.wav` | 16 kHz, đơn kênh, 160000 mẫu, 10 giây | Đúng 10 giây | Mô tả được giữ nguyên trong bảng kê |
| 1376 | `train:01374.wav` | `01374.wav` | 16 kHz, đơn kênh, 161792 mẫu, 10.112 giây | Dài hơn 10 giây | Mô tả được giữ nguyên trong bảng kê |
| 6094 | `train:06092.wav` | `06092.wav` | 16 kHz, đơn kênh, 160000 mẫu, 10 giây | Đúng 10 giây | Mô tả được giữ nguyên trong bảng kê |

Bảng kê còn ghi phân hoạch, đường dẫn WAV đã chuẩn hóa, SHA-256 WAV/mô tả và siêu dữ liệu nguồn để kiểm toán. Cùng một bảng kê được dùng cho cả bốn tiến trình. Mô tả, nhãn và các trường `model_id`/`model_name` không được đưa vào mạng cổng; mô tả chỉ được chuyển tới bộ chuyển đổi theo hợp đồng đã xác minh. Không dùng các phân hoạch `validation`/`test`.

## 6. Nguồn và khôi phục checkpoint

Mọi tệp đều tồn tại, đọc được và siêu dữ liệu trước/sau lượt chạy không đổi. Việc khôi phục do hàm tạo hiện có thực hiện; không có phương án dự phòng chỉ dùng checkpoint tiền huấn luyện.

| Mô hình | Lớp gốc | Checkpoint tiền huấn luyện | Checkpoint đã tinh chỉnh | Số tham số | Tham số được huấn luyện |
|---|---|---|---|---:|---:|
| E1 | `expert1_finetune.Train_Expert1.Expert1` | `weights/laion-clap.pt` (1,863,587,645 byte) | `weights/finetuned_weights/expert1_best.pt` (635,384,245 byte) | 158,334,169 | 0 |
| E2 | `expert2_finetune.expert2_model.Expert2` | `weights/mga-clap.pt` (1,711,356,348 byte) | `weights/finetuned_weights/expert2_best.pt` (594,417,402 byte) | 148,092,376 | 0 |
| E3 | `MoescoreExpert3` trong mã nguồn M2D đã xác minh | `weights/checkpoint-30.pth` (1,694,677,655 byte) | `weights/finetuned_weights/expert3_best.pt` (135,760,942 byte) | 200,556,547 | 0 |
| E4 | `expert4_finetune.model.Expert4` | BEATs (361,499,833 byte) và LAION-CLAP (1,863,587,645 byte) | `weights/finetuned_weights/expert4_best.pt` (155,455,153 byte) | 220,212,605 | 0 |

Hàm tạo báo khôi phục thành công bằng trình nạp gốc cho cả bốn mô hình. Mọi bộ chuyển đổi đều xác nhận `all_parameters_frozen=true`, `all_modules_eval=true`. Checkpoint nền E3 có SHA-256 được mã nguồn/hàm tạo ràng buộc với giá trị tin cậy `238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1`; không thay đường dẫn hoặc trình nạp.

E4 dùng bản UniLM độc lập tại `/home/vitquay1708/Working_Space/Research/2027/dependencies/unilm_expert4_original`, commit `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`, fingerprint `fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487`. Kiểm tra trước xác nhận đúng định danh. Trong lượt E2/E4, các bí danh BEATs tạm thời được khôi phục về đúng module đã có trước khi hàm tạo kết thúc (`beats_aliases_restored=true`). Module MGA vẫn ở cây `expert2_finetune/MGA-CLAP`; BEATs/backbone/modules vẫn thuộc bản UniLM đã xác minh.

### Nguồn module đã nạp

| Nhánh | Nguồn module quan sát được |
|---|---|
| E1 | Mô hình gốc từ `expert1_finetune/Train_Expert1.py`; không gian tên `laion_clap` từ gói trong Conda `nlp`. |
| E2 | `models`, `models.ase_model`, `models.audio_encoder`, `models.text_encoder` từ `expert2_finetune/MGA-CLAP/models/`. |
| E3 | `MoescoreExpert3` và `portable_m2d` từ `expert3_finetune/XACLE_2027_baselinemodel/`; BERT/bộ tách token dùng bộ nhớ đệm cục bộ đúng mã `google-bert/bert-base-uncased`. |
| E4 | Mô hình/SeqCoAttn từ `expert4_finetune/`; `BEATs.py`, `backbone.py`, `modules.py` từ bản UniLM độc lập; CLAP từ gói `laion_clap` trong Conda. |

Kiểm tra nguồn E2/E4 diễn ra ngay trong tiến trình dựng cặp, gồm nguồn module và danh tính đối tượng của bí danh. Không phát hiện module MGA bị thay thế hoặc bí danh BEATs còn sót sau khi phạm vi nhập kết thúc.

## 7. Suy luận thật từng mô hình chuyên gia

Mỗi mô hình chuyên gia chạy trong tiến trình riêng và chỉ được dựng một lần ở lượt trích xuất thành công; cùng một mô hình chạy B=1, B=4, rồi lặp B=1 để kiểm tra độ ổn định. Với E1/E2/E4, lượt tổng hợp cuối dùng lại tệp kết quả được tạo từ các lượt tính thật trước đó sau khi kiểm tra trạng thái, bảng kê, danh tính, kích thước, mã băm tệp và siêu dữ liệu checkpoint. E3 được chạy lại sau khi sửa đường dẫn bộ nhớ đệm và tạo tệp kết quả thật. Không tiến trình nào thay đầu ra bằng bộ giả lập, số ngẫu nhiên hoặc giá trị suy ra từ nhãn.

| Mô hình | Điểm thật | Đặc trưng thật | FP32/hữu hạn | Lặp lại B=1 | B=1 so với phần tử đầu B=4 |
|---|---|---|---|---|---|
| E1 | `[4,1]` | `[4,1024]` | Có | Khớp chính xác | max abs: điểm `1.19e-7`, đặc trưng `1.15e-7`; đạt `1e-6/1e-5` |
| E2 | `[4,1]` | `[4,2048]` | Có | Khớp chính xác | điểm `0`; đặc trưng `7.82e-8`; đạt `1e-6/1e-5` |
| E3 | `[4,1]` | `[4,3840]` | Có | Khớp chính xác | điểm `5.96e-8`; đặc trưng `1.64e-7`; đạt `1e-6/1e-5` |
| E4 | `[4,1]` | `[4,1024]` | Có | Khớp chính xác | điểm `0`; đặc trưng `4.47e-7`; đạt `1e-6/1e-5` |

Ngưỡng dung sai giữ nguyên `atol=1e-6`, `rtol=1e-5`. Có sai khác rất nhỏ giữa lượt B=1 và B=4; lặp cùng B=1 trong `eval` cho kết quả giống từng bit. Đây không phải sai khác về mẫu hoặc chính sách tiền xử lý.

## 8. Tích hợp E2/E4 đồng thời

Trước khi thử, còn khoảng `3,734,241,280` byte VRAM trống và `9,590,226,944` byte RAM khả dụng; vượt ngưỡng an toàn do trình xác minh đặt ra là 2 GiB VRAM và 6 GiB RAM. E2 rồi E4 được dựng trong cùng tiến trình và cả hai còn hoạt động trong lúc suy luận mẫu `train:04667.wav`.

| Nội dung kiểm tra | E2 | E4 |
|---|---:|---:|
| Điểm | `[1,1]` | `[1,1]` |
| Đặc trưng | `[1,2048]` | `[1,1024]` |
| Đóng băng/chế độ đánh giá | Đạt | Đạt |
| Điểm so với tệp kết quả độc lập | Sai số tuyệt đối 0 | Sai số tuyệt đối 0 |
| Đặc trưng so với tệp kết quả độc lập | Sai số tuyệt đối lớn nhất `7.82e-8` | Sai số tuyệt đối lớn nhất `4.47e-7` |
| Trong ngưỡng dung sai `1e-6/1e-5` | Có | Có |

Hai mô hình cùng tồn tại và suy luận thành công. Dấu vết nguồn nhập xác nhận MGA từ đúng cây; BEATs, `backbone`, `modules` từ UniLM gốc. Bí danh BEATs và danh tính module cũ được giữ nguyên sau khi kết thúc phạm vi nhập. Đây là kiểm tra cặp E2/E4, không chứng minh bốn mô hình có thể cùng tồn tại.

## 9. Đối chiếu danh tính và ghép đầu ra

Tất cả tệp kết quả có cùng mã SHA của bảng kê, cùng thứ tự bốn mã mẫu, mã SHA của mô tả và WAV. Mỗi mô hình ghi đúng tên và chỉ số thứ tự. Trình đọc từ chối bảng kê sai, tên mô hình sai, danh tính sai hoặc tensor không hữu hạn; NPZ được đọc với `allow_pickle=False`, chuỗi lưu dạng Unicode và tensor số giữ nguyên FP32.

Hàm ghép đầu ra hiện có cho kết quả:

```text
scores   [4,4]
features [4,7936]
```

Biên đặc trưng được giữ đúng và đối chiếu chính xác với các tệp đầu ra:

| Mô hình | Đoạn cột | Số chiều |
|---|---:|---:|
| E1 | `[0:1024]` | 1024 |
| E2 | `[1024:3072]` | 2048 |
| E3 | `[3072:6912]` | 3840 |
| E4 | `[6912:7936]` | 1024 |

`score_preservation` và `feature_preservation` đều true cho từng mô hình. Không cắt, chiếu, đổi thứ tự hoặc chuẩn hóa điểm/đặc trưng.

## 10. Cổng MoE và phép tính điểm

Mạng cổng được khởi tạo từ `model.moe_gating_core.GatingNetwork`, nhận đặc trưng đã ghép `[4,7936]` và trả trọng số `[4,4]`. Tích hợp dùng nguyên mạng cổng LOCAL-10 và biểu thức bốn hạng trong `forward_from_expert_outputs(...)`.

| Mẫu | E1 | E2 | E3 | E4 | Điểm MoE |
|---|---:|---:|---:|---:|---:|
| `train:04667.wav` | 0.5585800 | 0.5956577 | 0.6540775 | 0.8188562 | 0.6800984 |
| `train:03824.wav` | 0.3223053 | 0.3294120 | 0.6274151 | 0.2744917 | 0.3567162 |
| `train:01374.wav` | 0.6599074 | 0.6926699 | 0.6789916 | 0.6043854 | 0.6512675 |
| `train:06092.wav` | 0.6818105 | 0.5851617 | 0.6722600 | 0.7608966 | 0.6851016 |

Trọng số theo đúng thứ tự E1/E2/E3/E4:

| Mẫu | w1 | w2 | w3 | w4 | Tổng |
|---|---:|---:|---:|---:|---:|
| `train:04667.wav` | 0.1988612 | 0.2714302 | 0.1603130 | 0.3693955 | 1.0000000 |
| `train:03824.wav` | 0.2242310 | 0.2404509 | 0.1651846 | 0.3701334 | 1.0000000 |
| `train:01374.wav` | 0.1977819 | 0.2618501 | 0.1713474 | 0.3690207 | 1.0000000 |
| `train:06092.wav` | 0.2130471 | 0.2503228 | 0.1687290 | 0.3679011 | 1.0000000 |

Sai số lớn nhất của tổng trọng số so với 1 là `1.1920929e-7`; mọi trọng số không âm. Dự đoán khớp công thức bốn hạng tham chiếu với `max_abs_diff=0.0`, đồng thời nằm giữa điểm nhỏ nhất và lớn nhất của từng mẫu. Khoảng cách L1 lớn nhất giữa các vector trọng số mẫu là `0.0619586`; đây là quan sát trên mạng cổng chưa huấn luyện, không phải bằng chứng về chất lượng phân bổ chuyên gia đã học.

Không có nhãn, `model_id` hoặc `model_name` trong lời gọi mạng cổng. Tensor đầu vào không bị thay đổi. Không hiệu chỉnh hoặc chuẩn hóa điểm.

## 11. Gradient trên đầu ra thật đã tách khỏi mô hình chuyên gia

Điểm và đặc trưng được lưu trên CPU, không có gradient nối về mô hình chuyên gia (`requires_grad=false`). Phép lan truyền ngược dùng một hàm mục tiêu vô hướng kiểm tra, tạo từ điểm MoE và trọng số:

- Mạng cổng có 4,066,820 tham số có thể huấn luyện; ghi nhận gradient ở 6 tensor.
- Tất cả gradient hữu hạn và có ít nhất một gradient khác 0.
- Mô hình chuyên gia có 0 tham số được huấn luyện và không bị cập nhật.
- Không gọi bộ tối ưu, không chạy bước huấn luyện, không dùng nhãn thật để tính hàm mục tiêu và không lưu checkpoint.

Đây chỉ là kiểm tra khả năng truyền gradient vào mạng cổng, không phải huấn luyện mạng cổng.

## 12. Thống kê mô tả điểm thật

Thống kê dùng cùng bốn mẫu `train` đã chọn. Độ lệch chuẩn là độ lệch chuẩn tổng thể. Mẫu chẩn đoán nhỏ này không đại diện phân phối toàn bộ dữ liệu và không dùng để hiệu chỉnh.

| Mô hình | Số mẫu | Nhỏ nhất | Lớn nhất | Trung bình | Độ lệch chuẩn | Hữu hạn | `< -1` | `> 1` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 4 | 0.3223053 | 0.6818105 | 0.5556508 | 0.1425170 | 4/4 (100%) | 0 | 0 |
| E2 | 4 | 0.3294120 | 0.6926699 | 0.5507253 | 0.1344736 | 4/4 (100%) | 0 | 0 |
| E3 | 4 | 0.6274151 | 0.6789916 | 0.6581860 | 0.0199667 | 4/4 (100%) | 0 | 0 |
| E4 | 4 | 0.2744917 | 0.8188562 | 0.6146575 | 0.2114836 | 4/4 (100%) | 0 | 0 |

Trong bốn mẫu này, E3 có độ phân tán quan sát nhỏ hơn, E4 lớn hơn. Do cỡ mẫu nhỏ, không kết luận đây là khác biệt hiệu chuẩn hoặc đặc tính của toàn bộ tập. Đặc biệt, mã nguồn E3 cho phép điểm hồi quy không bị chặn; các giá trị trong `[-1,1]` ở mẫu này không chứng minh giới hạn đó. Không thêm chuẩn hóa hoặc cắt ngưỡng.

## 13. Nhãn và hàm mất mát

Tập dữ liệu Unified tiếp tục giữ nhãn gốc `[0,10]`; nhãn chỉ nằm trong bảng kê để nhận diện/truy xuất nguồn, không truyền vào mô hình chuyên gia hoặc mạng cổng. Tương thích với hàm mất mát huấn luyện vẫn thuộc phạm vi LOCAL-11:

```text
nhãn gốc [0,10] → (nhãn - 5) / 5 → mục tiêu chuẩn hóa [-1,1]
```

Phép đổi ngược ở suy luận theo mã nguồn gốc là `prediction * 5 + 5`. LOCAL-10.1 không thực hiện các phép đổi này, không thay hàm mất mát và không ép điểm MoE vào `[-1,1]`.

## 14. Bộ nhớ

Các mô hình chuyên gia được đo riêng theo tiến trình; không cộng các đỉnh riêng để ước lượng chính xác bộ nhớ khi chạy đồng thời. Đơn vị bảng là MiB (1024² byte); JSON có số byte đầy đủ.

| Tiến trình | RSS đỉnh | `CUDA allocated` đỉnh | `CUDA reserved` đỉnh | VRAM trống sau nhóm mẫu |
|---|---:|---:|---:|---:|
| E1 | 3340.5 | 757.6 | 875.9 | 2665.2 |
| E2 | 3295.8 | 731.3 | 855.9 | 2683.2 |
| E3 | 2968.4 | 820.7 | 917.8 | 2623.4 |
| E4 | 3144.9 | 1165.5 | 1400.0 | 2134.0 |
| Cặp E2→E4 cùng tiến trình | 3287.0 | 1523.0 | 1635.0 | 1843.5 |

Tại lúc kiểm tra cặp, RAM khả dụng khoảng 8.93 GiB; sau khi dựng và suy luận hai mô hình còn khoảng 6.89 GiB. GPU có tổng 3.63 GiB. Không có lỗi hết bộ nhớ GPU. Không thử dựng cả bốn mô hình chuyên gia trong cùng tiến trình: trên GTX 1650 chưa xác lập được khoảng dự phòng an toàn cho bốn hàm tạo. Đây là quyết định tránh rủi ro, không phải lỗi tương thích đã quan sát. Trạng thái đồng thời bốn mô hình là `NOT_ATTEMPTED_RESOURCE_RISK`.

## 15. Lỗi trình xác minh được phát hiện và xử lý

Lần thử E3 đầu tiên dừng trong hàm tạo vì bộ tách token không tìm thấy tệp trong bộ nhớ đệm ngoại tuyến. Nguyên nhân là trình xác minh truyền thư mục cấp trên `~/.cache/huggingface` vào trường `cache_dir`, trong khi mã nguồn gốc cần thư mục Hub `~/.cache/huggingface/hub`. Đây là lỗi cấu hình của trình xác minh, không phải lỗi checkpoint hoặc không gian tên. Đã sửa kiểm tra trước để truyền đúng `cache_dir` như hàm tạo, thêm lựa chọn `--hf-cache-dir` (vẫn nhận `--hf-home` làm bí danh), và thêm kiểm thử hồi quy cho quy tắc này. Sau sửa, E3 nạp checkpoint nền và checkpoint tinh chỉnh, rồi suy luận đủ bốn mẫu ngoại tuyến. Không sửa bộ chuyển đổi hoặc mã nguồn M2D.

## 16. Kiểm thử và kết quả cuối

- Mốc trước thay đổi: 219 bài đã được báo là đạt.
- Kiểm thử mới LOCAL-10.1: 9 bài đạt.
- Nhóm LOCAL-10 `test_moe*.py`: 38 bài đạt.
- Toàn bộ kiểm thử hồi quy: **228 bài đạt, 0 lỗi, 0 bỏ qua**, hoàn tất trong 117.295 giây.
- `py_compile` cho trình xác minh, kiểm thử mới và mô-đun tích hợp: đạt.
- Chế độ `--mode preflight`: đạt, không dựng mô hình.
- Chế độ `--mode sequential`: đạt, đầu ra thật của bốn mô hình được ghép và đưa qua mạng cổng.
- Chạy thật cặp E2/E4: đạt trên một mẫu chung.

Lệnh chạy toàn bộ kiểm thử hồi quy:

```bash
conda run -n nlp --no-capture-output python3 \
  -m unittest discover -s tests -p 'test_*.py' -v
```

## 17. Tệp và kết quả tạm

Tệp tạo mới hoặc sửa cho milestone:

- `scripts/verify_real_expert_to_gate.py` — giao diện dòng lệnh để kiểm tra trước, trích xuất tuần tự, xác minh nguồn/bảng kê, ghép đầu ra và đo tài nguyên.
- `tests/test_real_expert_output_manifest.py` — kiểm thử bảng kê, tệp FP32, kiểm tra bộ nhớ đệm và lỗi danh tính.
- `tests/test_real_output_gating_integration.py` — kiểm thử đường đi từ đầu ra đã xác thực qua hàm ghép và mạng cổng.
- `model/moe_forward_integration.py` — thêm `forward_from_expert_outputs(...)` để kiểm tra đầu ra thật đã trích xuất mà không cần giữ bốn mô hình.
- `docs/LOCAL-10/real-integration-verification.md` — báo cáo này.
- `docs/superpowers/plans/2026-11-02-local101-real-output-gating.md` — kế hoạch thực hiện.

Tệp kết quả tạm thời, không nằm trong Git:

- JSON chính: `/tmp/moescore_local101_runtime.json`
- JSON kiểm tra trước: `/tmp/moescore_local101_preflight.json`
- Bảng kê cùng tệp NPZ và báo cáo từng mô hình: `/tmp/moescore_local101/`

Siêu dữ liệu checkpoint trước/sau được giữ nguyên; không checkpoint mô hình nào bị sửa.

## 18. Giới hạn còn lại

- Chỉ dùng bốn mẫu `train`; thống kê điểm không mô tả toàn bộ tập.
- E1/E2/E3/E4 được trích xuất tuần tự; bốn mô hình không cùng tồn tại trong một tiến trình.
- Chỉ E2/E4 được thử đồng thời trực tiếp, trên một mẫu.
- B=1/B=4 ổn định trong ngưỡng dung sai đã nêu; sai khác phép tính rất nhỏ giữa cỡ lô khác nhau không bằng 0 từng bit.
- Mạng cổng chưa được huấn luyện; trọng số quan sát chỉ phản ánh lần khởi tạo có seed.
- Chưa dự đoán/đánh giá toàn tập, tính SRCC, tinh chỉnh theo validation hoặc hiệu chuẩn.
- Chưa triển khai quy trình huấn luyện hoặc chọn chiến lược bộ nhớ cho huấn luyện.

## 19. Mức sẵn sàng cho LOCAL-11

Biên **đầu ra chuyên gia thật → hàm ghép → mạng cổng** đã được xác minh với dữ liệu/checkpoint thật và có thể làm nền cho LOCAL-11. LOCAL-11 vẫn cần xác định rõ nơi đổi nhãn gốc `[0,10]` sang mục tiêu hàm mất mát chuẩn hóa `[-1,1]`, chiến lược tài nguyên khi huấn luyện trên GTX 1650 và quy trình chỉ dùng phân hoạch `train` để cập nhật mạng cổng. Kết quả này không xác nhận bốn mô hình cùng tồn tại đồng thời, không xác nhận mạng cổng đã học và không xác nhận hệ MoE sẵn sàng huấn luyện toàn phần.
