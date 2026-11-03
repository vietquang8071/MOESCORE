# LOCAL-09.1 — Báo cáo kiểm chứng tính tương đương số học của tiền xử lý trên dữ liệu thật

## 1. Tóm tắt

**ĐẠT — ĐÃ KIỂM CHỨNG ĐỘ TƯƠNG ĐỒNG CỦA TIỀN XỬ LÝ TRÊN DỮ LIỆU THẬT.** Đã quét siêu dữ liệu của cả ba tập huấn luyện, validation và kiểm thử, đồng thời đọc phần đầu của tệp WAV trong 13.500 bản ghi. Dữ liệu PCM chỉ được giải mã cho 10 mẫu thuộc tập huấn luyện và validation, được chọn xác định theo seed; tập kiểm thử chỉ dùng để kiểm tra tính toàn vẹn siêu dữ liệu và thông tin WAV. Cả bốn nhánh xử lý gốc và Unified khớp chính xác trên 10/10 mẫu cho mỗi chuyên gia. Caption gốc, nhãn, tên tệp, tập dữ liệu, thứ tự mẫu, độ dài hợp lệ của E4 và DataLoader CPU thực tế với cỡ lô B=1/B=2 đều đạt.

Không phát hiện lỗi trong `data/unified_dataset.py` hoặc `data/unified_collate.py`, vì vậy không cần sửa cách xử lý dữ liệu. Chương trình kiểm chứng được điều chỉnh sau lần chạy đầu để so sánh đúng chế độ huấn luyện/đánh giá của E1, chạy được từ dòng lệnh và không giải mã PCM của tập kiểm thử. Không khởi tạo mô hình, không nạp tệp trọng số, không dùng GPU, không chạy suy luận và không tính chỉ số đánh giá.

## 2. Môi trường thực thi

| Thành phần | Giá trị quan sát được |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.14.1+cu130 |
| TorchAudio | 2.11.0+cu130 |
| SciPy | 1.15.3 |
| SoundFile | 0.14.0 |
| NumPy | 1.26.4 |
| Hệ điều hành | Linux x86_64, kernel 6.17.0-40-generic |
| Chế độ ngoại tuyến | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` |
| Ngữ cảnh CUDA | Chưa khởi tạo |
| Khởi tạo mô hình / nạp tệp trọng số / suy luận GPU | `false` / `false` / `false` |

## 3. Đường dẫn tập dữ liệu và lệnh chạy

Thư mục gốc được kiểm tra:

```text
/home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset
```

Các thư mục `meta_data/{train,validation,test}` và `wav/{train,validation,test}` đều tồn tại và đọc được. Đã chạy chương trình kiểm chứng bằng lệnh:

```bash
conda run -n nlp --no-capture-output python3 scripts/verify_unified_dataset_runtime.py \
  --dataset-root /home/vitquay1708/Working_Space/Research/2027/data/XACLE_Challenge_2027_development_dataset/XACLE_Challenge_2027_development_dataset \
  --splits train,validation,test \
  --parity-splits train,validation \
  --seed 2027 --samples-per-split 2 --batch-sizes 1,2 --offline \
  --report-json /tmp/moescore_local091_runtime.json
```

Chương trình kiểm chứng đọc toàn bộ CSV để rà soát siêu dữ liệu, dùng `soundfile.info` để đọc thông tin đầu tệp và chỉ giải mã PCM cho các mẫu được chọn thuộc tập huấn luyện/validation. Tập kiểm thử không được đưa qua các nhánh tiền xử lý gốc/Unified; chương trình chỉ kiểm tra siêu dữ liệu, sự tồn tại của tệp và thông tin WAV.

## 4. Kiểm tra siêu dữ liệu

Tiêu đề cột thực tế của cả ba CSV điểm trung bình:

```text
wav_file_name,text,average_score,model_ID,model_name
```

Trong bộ dữ liệu cục bộ này không có tên cột điểm dạng bí danh alignment; cột thực tế là `average_score`. Tên cột được xác định từ dòng tiêu đề, không dựa vào vị trí.

| Tập dữ liệu | Số dòng điểm trung bình | Số WAV duy nhất | Nhãn nhỏ nhất–lớn nhất | Điểm trung bình | Thiếu WAV | Tên tệp/cặp trùng | Mô tả rỗng | Nhãn lỗi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Huấn luyện | 7.500 | 7.500 | 0–10 | 7,0345 | 0 | 0 / 0 | 0 | 0 |
| Validation | 3.000 | 3.000 | 0–10 | 6,91925 | 0 | 0 / 0 | 0 | 0 |
| Kiểm thử | 3.000 | 3.000 | 0–10 | 7,15542 | 0 | 0 / 0 | 0 | 0 |

Số dòng khớp với số liệu 7.500/3.000/3.000 trong tài liệu thử thách. Không có dòng CSV thiếu tên tệp, mô tả hoặc nhãn; không có dòng thừa trường, đường dẫn không hợp lệ, định dạng tên tệp bất thường hoặc phần đầu WAV không đọc được. Cả 13.500 tệp đều ở định dạng `WAV`, kiểu mẫu `PCM_16`, tần số 16.000 Hz, mono.

Ba CSV điểm từng người nghe cũng được kiểm tra cấu trúc và số dòng; chúng không được dùng để tạo thêm mẫu:

| Tệp | Số dòng | Tiêu đề cột thực tế |
|---|---:|---|
| `train.csv` | 30.000 | `wav_file_name,text,score,listener_id,model_ID,model_name` |
| `validation.csv` | 12.000 | `wav_file_name,Caption,REL,listener id,model_ID,model_name` |
| `test.csv` | 24.000 | `wav_file_name,text,score,listener_id,model_ID,model_name` |

Không có dòng nào chứa trường thừa hoặc trường rỗng trong các CSV điểm từng người nghe. Unified Dataset dùng CSV điểm trung bình, không biến điểm của từng người nghe thành mẫu bổ sung.

## 5. Tính toàn vẹn các tập và thống kê âm thanh

| Tập dữ liệu | WAV ngắn hơn 10 giây | Đúng 10 giây | Dài hơn 10 giây | Tần số lấy mẫu | Số kênh | Kiểu PCM |
|---|---:|---:|---:|---|---|---|
| Huấn luyện | 353 | 5.397 | 1.750 | 16 kHz | Mono | PCM_16 |
| Validation | 123 | 2.127 | 750 | 16 kHz | Mono | PCM_16 |
| Kiểm thử | 39 | 1.918 | 1.043 | 16 kHz | Mono | PCM_16 |
| **Tổng** | **515** | **9.442** | **3.543** | **16 kHz** | **Mono** | **PCM_16** |

Các số liệu thời lượng được lấy từ phần đầu WAV của toàn bộ tập. Mẫu ngắn, đúng 10 giây và mẫu dài đều được đưa vào so sánh PCM của tập huấn luyện/validation. Không có WAV khác 16 kHz, âm thanh stereo/đa kênh hoặc mô tả có khoảng trắng ở đầu/cuối.

Không có tên WAV hoặc cặp `(tên tệp, mô tả)` trùng giữa hai tập bất kỳ. Mô tả trùng nhưng thuộc cặp khác xuất hiện ở huấn luyện/validation: 11; huấn luyện/kiểm thử: 1; validation/kiểm thử: 0. Đây là thống kê cảnh báo, không phải bằng chứng rò rỉ dữ liệu vì cặp âm thanh–mô tả vẫn khác nhau. Cả 9 `model_ID` xuất hiện ở nhiều tập; đây là siêu dữ liệu của mô hình tạo âm thanh, được giữ để kiểm tra và không truyền làm đầu vào cho bộ định tuyến MoE.

## 6. Các nhánh tiền xử lý gốc được đối chiếu

| Chuyên gia | Lớp dữ liệu/hàm ghép lô gốc và mã nguồn | Bộ giải mã và cách xử lý âm thanh đã xác nhận |
|---|---|---|
| E1 | `Expert1Dataset`, `collate_batch` — [`Train_Expert1.py`](../../expert1_finetune/Train_Expert1.py#L29) | `torchaudio.load`; lấy trung bình các kênh; `torchaudio.functional.resample` tới 48 kHz; 480.000 mẫu. Âm thanh ngắn: lặp `floor(target/L)` lần rồi đệm số 0 bên phải. Âm thanh dài: cắt ngẫu nhiên khi huấn luyện, cắt giữa khi đánh giá. Giữ caption nguyên gốc. |
| E2 | `Expert2Dataset`, `collate_expert2` — [`expert2_data.py`](../../expert2_finetune/expert2_data.py#L18) | `scipy.io.wavfile.read`; đổi PCM số nguyên sang số thực theo dtype; lấy trung bình kênh; `resample_poly` theo GCD tới 32 kHz; 320.000 mẫu. Âm thanh dài cắt từ đầu; âm thanh ngắn đệm số 0 bên phải. Caption được xử lý bằng `.strip()`. |
| E3 | `MoescoreExpert3Dataset`, `collate_fn` — [`moescore_expert3_dataset.py`](../../expert3_finetune/XACLE_2027_baselinemodel/datasets/moescore_expert3_dataset.py#L17) | `soundfile.read(dtype="float32", always_2d=True)`; lấy trung bình kênh; `resample_poly` theo GCD tới 16 kHz; 160.000 mẫu. Âm thanh dài cắt từ đầu; âm thanh ngắn đệm số 0 bên phải. Caption được xử lý bằng `.strip()`. |
| E4 | `Expert4Dataset`, `collate_expert4` — [`dataset.py`](../../expert4_finetune/dataset.py#L23) | Đọc WAV bằng SciPy; xử lý PCM có dấu/không dấu theo mã nguồn; lấy trung bình kênh; `resample_poly` với `Fraction` tới 16 kHz; cắt từ đầu ở mốc 160.000 mẫu. Mỗi mục dữ liệu giữ độ dài trước khi đệm; `collate_expert4` đệm bên phải tới `max_samples=160000` và giữ `wav_lens`. Caption được xử lý bằng `.strip()`. |

Bộ dữ liệu thống nhất nằm tại [`unified_dataset.py`](../../data/unified_dataset.py#L57): E1–E4 dùng các nhánh giải mã/chuyển đổi riêng `_expert1_audio`…`_expert4_audio` (dòng 245–330). Hàm [`collate_unified`](../../data/unified_collate.py#L43) chỉ kiểm tra và ghép tensor trên CPU; không cắt, đổi tần số lấy mẫu, chuẩn hóa mô tả hoặc thay đổi độ dài hợp lệ.

## 7. Chọn mẫu dữ liệu thật

Mẫu được chọn theo thứ tự xác định bằng SHA-256 của `(seed=2027, split, filename)`, sau đó bổ sung để có âm thanh ngắn, đúng 10 giây và dài khi dữ liệu cho phép. Không chọn theo nhãn hoặc kết quả mô hình. Chỉ các mẫu huấn luyện và validation được đưa qua so sánh PCM.

| Tập / dòng CSV | Tệp WAV | Số khung ở 16 kHz | Lý do chọn | Nhãn | model_ID |
|---|---|---:|---|---:|---:|
| Huấn luyện / 1087 | `01085.wav` | 161.792 | Âm thanh dài | 2,5 | 4 |
| Huấn luyện / 2361 | `02359.wav` | 160.000 | Đúng 10 giây | 0,5 | 0 |
| Huấn luyện / 3826 | `03824.wav` | 160.000 | Chọn theo mã băm | 5,25 | 0 |
| Huấn luyện / 4669 | `04667.wav` | 146.000 | Chọn theo mã băm | 4,25 | 0 |
| Huấn luyện / 5895 | `05893.wav` | 157.536 | Âm thanh ngắn | 2,75 | 0 |
| Validation / 547 | `08045.wav` | 160.000 | Đúng 10 giây | 10,0 | 0 |
| Validation / 1583 | `09081.wav` | 161.792 | Âm thanh dài | 5,0 | 4 |
| Validation / 2113 | `09611.wav` | 160.000 | Chọn theo mã băm | 7,25 | 3 |
| Validation / 2305 | `09803.wav` | 160.000 | Chọn theo mã băm | 6,75 | 1 |
| Validation / 2557 | `10055.wav` | 147.920 | Âm thanh ngắn | 4,5 | 0 |

Số dòng CSV bắt đầu từ dòng dữ liệu đầu tiên sau dòng tiêu đề, đánh số từ 1. Trích đoạn mô tả và toàn bộ siêu dữ liệu có trong `/tmp/moescore_local091_runtime.json`. Dữ liệu thật không có tần số lấy mẫu khác 16 kHz, âm thanh stereo hoặc mô tả có khoảng trắng ở hai đầu; các trường hợp đó chỉ được kiểm tra bằng kiểm thử tổng hợp. Vì đầu vào thực tế là 16 kHz, E1/E2 vẫn được đổi tần số lên mức đích 48/32 kHz.

## 8. Độ tương đồng E1 và cắt ngẫu nhiên

- Tập huấn luyện: 5 dạng sóng được so giữa tiền xử lý gốc `training=True` và phép cắt ngẫu nhiên của Unified tại cùng vị trí bắt đầu; cả 5/5 khớp chính xác.
- Tập validation: 5 dạng sóng được so với phép cắt giữa của tiền xử lý gốc `training=False`; cả 5/5 khớp chính xác. WAV dài `09081.wav` được cắt giữa; WAV ngắn giữ đúng quy tắc lặp rồi đệm.
- WAV huấn luyện dài `01085.wav`: 161.792 khung nguồn thành 485.376 mẫu sau khi đổi lên 48 kHz; khoảng vị trí cắt hợp lệ là `[0, 5376]`. Với `seed=2027, epoch=0, index=1085`, vị trí cắt là `3645`. Khi đặt phép cắt huấn luyện gốc dùng đúng vị trí này, dạng sóng `[1,480000]` khớp từng bit.
- Cùng `(seed, epoch, index)` luôn cho cùng vị trí cắt; 16 epoch tạo ra nhiều vị trí khác nhau. Trạng thái bộ sinh số ngẫu nhiên toàn cục của Python/NumPy/Torch không đổi.
- Bộ sinh số ngẫu nhiên cục bộ của Unified và nguồn gốc đều chọn số nguyên trên cùng khoảng có tính cả hai đầu mút. Tuy vậy, hai lần gọi huấn luyện độc lập không được đảm bảo chọn cùng vị trí. Vì vậy, chỉ kết luận hai dạng sóng huấn luyện bằng nhau từng mẫu khi cố định cùng vị trí cắt.

## 9. Độ tương đồng E2

Cả 10 dạng sóng thuộc tập huấn luyện/validation đều bằng chính xác với `Expert2Dataset`; sai số tuyệt đối lớn nhất/trung bình là `0.0`, số phần tử sai khác là `0`. Đã giữ bộ giải mã SciPy, phép đổi PCM, lấy trung bình kênh, đổi tần số polyphase lên 32 kHz, cắt 10 giây đầu và đệm số 0. Caption sau `.strip()` khớp mã nguồn; CSV thật không có khoảng trắng đầu/cuối.

## 10. Độ tương đồng E3

Cả 10 dạng sóng thuộc tập huấn luyện/validation đều bằng chính xác với `MoescoreExpert3Dataset`; sai số tuyệt đối lớn nhất/trung bình là `0.0`, số phần tử sai khác là `0`. Đã giữ bộ giải mã SoundFile float32, lấy trung bình kênh, đổi tần số polyphase, cắt từ đầu và đệm số 0. Quy tắc `.strip()` cho caption cũng khớp.

## 11. Độ dài, dạng sóng và ghép lô E4

Phần waveform hợp lệ của cả 10 mục do `Expert4Dataset` gốc trả về đều khớp với Unified. Độ dài từ mã gốc khớp `expert4_wav_lens` ở cả 10/10 mẫu; phần đệm cố định còn lại của Unified chỉ chứa số 0.

Các batch có độ dài hỗn hợp từ WAV thật:

| Tập dữ liệu | Cỡ lô | `wav_lens` gốc | Kích thước gốc/Unified | Kết quả phần âm thanh hợp lệ |
|---|---:|---|---|---|
| Huấn luyện | B=1 | `[146000]` | `[1,1,160000]` / `[1,1,160000]` | Khớp chính xác |
| Huấn luyện | B=2 | `[160000,146000]` | `[2,1,160000]` / `[2,1,160000]` | Khớp chính xác |
| Validation | B=1 | `[147920]` | `[1,1,160000]` / `[1,1,160000]` | Khớp chính xác |
| Validation | B=2 | `[160000,147920]` | `[2,1,160000]` / `[2,1,160000]` | Khớp chính xác |

Trong phiên bản mã nguồn này, `collate_expert4` đệm từng mục dữ liệu tới `max_samples=160000`; độ rộng không được đặt theo mẫu dài nhất trong lô. Vì vậy, kích thước lô từ mã gốc và Unified bằng nhau, nhưng giá trị vùng đệm và độ dài vẫn được kiểm tra riêng. Phần đệm của cả hai phía đều bằng 0; không có trường hợp thay độ dài hợp lệ bằng 160.000.

## 12. Độ tương đồng của caption

Với cả 10 dòng thuộc tập huấn luyện/validation, caption trong CSV khớp nguyên văn với `captions_raw`. Nhánh E1 giữ caption nguyên gốc; E2/E3/E4 gọi `.strip()` đúng như mã nguồn. Tuy nhiên, caption thật không có khoảng trắng ở đầu/cuối nên lần kiểm tra này không đo được tác động thực tế của `.strip()`; kiểm thử tổng hợp đã kiểm tra trường hợp đó. Không chuyển chữ thường hoặc chuẩn hóa Unicode.

## 13. Nhãn và định danh mẫu

Ở cả 10 bản ghi được đối chiếu, tên WAV, dòng CSV, tập dữ liệu, caption, nhãn và siêu dữ liệu mô hình vẫn thuộc cùng một mẫu. Nhãn từ cả bốn bộ dữ liệu gốc bằng chính xác nhãn vô hướng `float32` của Unified; thang 0–10 được giữ nguyên, không chuẩn hóa, dịch hoặc cắt ngưỡng. CSV điểm trung bình của tập kiểm thử cũng có cột `average_score` với 3.000 dòng; nếu CSV không có cột điểm, Unified Dataset sẽ không tự tạo nhãn.

## 14. Bộ nạp dữ liệu CPU trên dữ liệu thật

DataLoader của tập validation dùng `num_workers=0`, `shuffle=False`, `collate_fn=collate_unified`:

| Cỡ lô | Tệp theo thứ tự | Nhãn | Độ dài E4 | Kết quả |
|---:|---|---|---|---|
| B=1 | `10055.wav` | `[4.5]` | `[147920]` | ĐẠT |
| B=2 | `08045.wav`, `10055.wav` | `[10.0,4.5]` | `[160000,147920]` | ĐẠT |

Trong cả hai lô, waveform trên CPU có dtype `float32`; kích thước lần lượt là E1 `[B,1,480000]`, E2 `[B,1,320000]`, E3/E4 `[B,1,160000]`; độ dài E4 là `int64 [B]`. Caption, nhãn, tên tệp và mã mẫu giữ đúng thứ tự. Không dùng CUDA, adapter chuyên gia hoặc suy luận mô hình.

## 15. Tổng hợp sai khác số

Ngưỡng được ấn định trước là `atol=1e-6`, `rtol=1e-5`. Do hai phía dùng cùng bộ giải mã và bộ đổi tần số lấy mẫu gốc, kết quả kỳ vọng là bằng chính xác. Có tổng cộng 10 bản ghi thuộc tập huấn luyện/validation cho mỗi nhánh:

| Chuyên gia | Số lần so sánh | Bằng chính xác | Đạt `allclose` | Sai số tuyệt đối lớn nhất | Sai số tuyệt đối trung bình |
|---|---:|---:|---:|---:|---:|
| E1 | 10 | 10 | 10 | 0 | 0 |
| E2 | 10 | 10 | 10 | 0 | 0 |
| E3 | 10 | 10 | 10 | 0 | 0 |
| Phần hợp lệ E4 | 10 | 10 | 10 | 0 | 0 |

Caption và nhãn cũng khớp chính xác 10/10. Đây là kết quả đối chiếu trên các mẫu được chọn, không phải giải mã và so sánh PCM của toàn bộ WAV.

## 16. Diễn biến và điều chỉnh chương trình kiểm chứng

Lần chạy dòng lệnh đầu tiên không thêm thư mục gốc kho mã vào `sys.path`, dẫn tới `ModuleNotFoundError: data`; chương trình kiểm chứng đã được chỉnh để thêm thư mục gốc một cách tường minh. Lần so sánh ban đầu ghép nhầm phép cắt ngẫu nhiên của Unified ở chế độ huấn luyện với phép cắt giữa của mã gốc ở chế độ đánh giá, nên chỉ E1 có vị trí cắt khác. Phép kiểm tra được chỉnh để mã gốc dùng cùng vị trí cắt `(seed,epoch,index)` rồi so sánh lại; kết quả khớp chính xác. Không thay đổi Unified Dataset hoặc Collate.

## 17. Trường hợp không có trong dữ liệu thật

- Âm thanh stereo/đa kênh: 0/13.500; đã có kiểm thử tổng hợp.
- Tần số lấy mẫu nguồn khác 16 kHz: 0/13.500; đã có kiểm thử tổng hợp. Âm thanh thật 16 kHz vẫn được E1/E2 đổi tần số lên mức đích 48/32 kHz.
- Caption có khoảng trắng đầu/cuối: 0/13.500; kiểm thử tổng hợp xác minh ranh giới chuẩn hóa.
- Âm thanh ngắn, đúng 10 giây và dài: đã có mẫu thật thuộc tập huấn luyện/validation cho cả ba nhóm.
- PCM của tập kiểm thử: không giải mã trong phạm vi đối chiếu; chỉ kiểm tra sự tồn tại của tệp, siêu dữ liệu và header/thống kê WAV.

## 18. Kiểm thử chương trình kiểm chứng và hồi quy

- Kiểm thử mới `test_unified_runtime_verification.py`: **8/8 ĐẠT**.
- Nhóm kiểm thử `test_unified*.py`: **43/43 ĐẠT**, bao gồm 8 kiểm thử mới.
- Kiểm tra cú pháp bằng `py_compile`: **ĐẠT**.
- Mốc kiểm thử trước thay đổi: **195 kiểm thử**; toàn bộ lệnh chạy kết thúc với mã 0, khớp báo cáo trước đó về 195/195.
- Kết quả cuối: **203/203 ĐẠT**, 0 lỗi, 0 bỏ qua (`Ran 203 tests in 108.099s`).
- Khi quét và chạy chương trình có cảnh báo API sắp bị loại bỏ (`FutureWarning`) từ `timm.models.layers`; cảnh báo không ảnh hưởng kết quả đối chiếu hoặc kiểm thử.

Lệnh suite đầy đủ:

```bash
conda run -n nlp --no-capture-output python3 -m unittest discover \
  -s tests -p 'test_*.py' -v
```

## 19. Tệp được tạo

- `scripts/verify_unified_dataset_runtime.py` — quét siêu dữ liệu/header WAV, so sánh mã gốc với Unified theo cách xác định, kiểm tra phép cắt ngẫu nhiên E1, độ dài/ghép lô E4, DataLoader và xuất JSON.
- `tests/test_unified_runtime_verification.py` — kiểm thử hàm so sánh, siêu dữ liệu, phép cắt, đệm E4 và báo cáo.
- `docs/LOCAL-09/runtime-verification.md` — báo cáo này.
- `/tmp/moescore_local091_runtime.json` — kết quả JSON cho máy đọc.

Không sửa `data/unified_dataset.py`, `data/unified_collate.py`, mã nguồn chuyên gia gốc, mô hình, bộ điều hợp hoặc checkpoint. Các thay đổi đã có từ trước trong cây làm việc được giữ nguyên.

## 20. Hạn chế còn lại và mức sẵn sàng cho LOCAL-10

Các nhánh dữ liệu/ghép lô đã có bằng chứng thực nghiệm khớp số học với mã nguồn gốc trên tập huấn luyện/validation; mã mẫu, nhãn và độ dài E4 được giữ đúng. Kết quả của 10 mẫu không đại diện cho việc so sánh PCM trên toàn bộ 13.500 WAV. PCM của tập kiểm thử, âm thanh stereo, tần số lấy mẫu khác thường và caption có khoảng trắng chưa được đối chiếu bằng dữ liệu thật; hành vi phần mềm tương ứng đã được kiểm tra bằng dữ liệu dựng riêng cho kiểm thử.

**Bộ dữ liệu thống nhất đã sẵn sàng cho bước tích hợp tiếp theo trong phạm vi dữ liệu của LOCAL-09.1.** Kết quả này không xác nhận suy luận chuyên gia, khả năng chứa đồng thời bốn mô hình trên GPU, lượt truyền xuôi/bộ định tuyến MoE, quá trình huấn luyện hoặc chỉ số đánh giá trên tập validation. LOCAL-10 cần kiểm tra riêng các phần đó.
