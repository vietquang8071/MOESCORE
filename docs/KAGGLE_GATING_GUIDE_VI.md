# Hướng dẫn huấn luyện cổng MOESCORE trên Kaggle cho XACLE 2027

## 1. Mục đích và thành phần

Mở `notebooks/kaggle_train_moe_gating.ipynb` và chạy các ô từ trên xuống. Notebook dùng mã hiện tại của repository để phục hồi bốn expert đã fine-tune, trích xuất score và feature, huấn luyện riêng gating network, chọn checkpoint theo SRCC validation, rồi đánh giá validation và test. Kết quả chỉ được ghi khi các bước tương ứng thực sự chạy thành công.

Phương pháp tham khảo từ [repository MOESCORE gốc](https://github.com/S-Orion/MOESCORE) và bài báo `MOESCORE.pdf` trong repository: chuyên gia được fine-tune riêng, sau đó đóng băng khi huấn luyện cổng phụ thuộc đặc trưng từng mẫu. Mã hiện tại và cấu hình đã chốt cho XACLE 2027 là căn cứ thực thi. Không lấy kết quả công bố năm 2026 làm kết quả của notebook này.

Các thành phần dùng lại:

- `create_expert_adapter`: phục hồi checkpoint theo từng expert.
- `UnifiedXACLEDataset`, `collate_unified`: bốn nhánh xử lý âm thanh đã được xác minh.
- `MoeFeatureStoreWriter`: lưu shard NPY và manifest có checksum.
- `scripts/train_moe_gating.py`: optimizer, loss, scheduler, checkpoint và resume hiện hành.
- `MoeGateEvaluator`, `compute_metrics`: suy luận và tính metric trên toàn bộ split.

Thành phần bổ sung là `scripts/extract_moe_feature_store.py` để trích xuất toàn tập theo shard và `scripts/kaggle_moe_pipeline.py` để chuẩn bị cấu hình, kiểm tra môi trường và xuất đánh giá cuối. Notebook không viết lại vòng huấn luyện.

## 2. Chuẩn bị dữ liệu đầu vào

### 2.1. Bộ dữ liệu XACLE

Thêm bộ dữ liệu được phép sử dụng vào Kaggle Input. Một trong hai cách bố trí sau được hỗ trợ:

```text
/kaggle/input/<slug>/XACLE_dataset/meta_data/
/kaggle/input/<slug>/XACLE_dataset/wav/
```

hoặc:

```text
/kaggle/input/<slug>/meta_data/
/kaggle/input/<slug>/wav/
```

Root được chọn phải có cấu trúc:

```text
XACLE_dataset/
├── meta_data/
│   ├── train_average.csv
│   ├── validation_average.csv
│   ├── test_average.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── wav/
    ├── train/
    ├── validation/
    └── test/
```

Ba tệp `*_average.csv` chứa cặp audio–text và nhãn trung bình. Các tệp còn lại chứa đánh giá từng người nghe và không được dùng làm các mẫu bổ sung. Notebook yêu cầu đúng `wav_file_name`, `text`, `average_score` cho train/validation. Cột nhãn test chỉ được đọc khi đánh giá cuối. Metadata như `model_id`, `model_ID`, `model_name` không trở thành đầu vào cổng.

Số cặp dự kiến là train 7.500, validation 3.000, test 3.000. Notebook dừng nếu số lượng sai, tên WAV trùng, caption rỗng, đường dẫn vượt khỏi thư mục split, thiếu WAV hoặc nhãn train/validation không hữu hạn trong [0,10]. Tính hợp lệ của nhãn test được kiểm tra riêng sau suy luận cuối.

Nếu tìm thấy nhiều root hợp lệ, điền `DATASET_ROOT` ở ô cấu hình; notebook không tự chọn một root bất kỳ.

### 2.2. Tám checkpoint đã chốt

Theo danh sách tệp thực tế đã xác nhận, notebook dùng `WEIGHTS_ROOT = Path('/kaggle/input/datasets/qnv8071/xacle-weights/weights')`:

```text
weights/
├── pretrained/
│   ├── laion-clap.pt
│   ├── mga-clap.pt
│   ├── checkpoint-30.pth
│   └── BEATs_iter3_plus_AS2M.pt
└── finetuned_weights/
    ├── expert1_best.pt
    ├── expert2_best.pt
    ├── expert3_best.pt
    └── expert4_best.pt
```

Giữ nguyên các tệp hiện có. Nếu vị trí mount khác, sửa `WEIGHTS_ROOT`; nếu tên tệp/thư mục khác, sửa các đường dẫn trong `EXPERT_CHECKPOINTS`. Mapping này được truyền vào `generate_configs()` để cả cấu hình YAML lưu trên đĩa và cấu hình runtime thống nhất. E1/E2 khởi tạo pretrained rồi khôi phục đầy đủ fine-tuned state. E3/E4 khởi tạo base/pretrained rồi khôi phục phần fine-tuned theo loader gốc. Factory tiếp tục kiểm tra schema, key, shape và fingerprint; không có fallback sang pretrained-only. Chỉ dùng checkpoint đáng tin cậy, vì một số loader gốc cần giải tuần tự checkpoint Python của dự án.

### 2.3. Mã nguồn

E2 dùng `expert2_finetune/MGA-CLAP/`, E3 dùng `expert3_finetune/XACLE_2027_baselinemodel/`. Cả hai đã nằm trong MOESCORE; không cần upload riêng.

E4 ưu tiên `expert4_finetune/unilm/`. Nguồn đi kèm khác nguồn huấn luyện gốc tại hai import tương đối trong `BEATs.py` và `backbone.py`. Bộ kiểm tra chỉ đảo đúng hai import này khi tính fingerprint và yêu cầu toàn bộ ba tệp, gồm `modules.py`, khôi phục SHA-256 gốc `fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487`, gắn với commit huấn luyện `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`. Mã số Git hiện tại vẫn được giữ đúng giá trị quan sát; không giả lập commit gốc. Nếu còn khác biệt khác, phục hồi bị từ chối.

Khi tắt mạng, đính kèm một bản source đầy đủ của repository chứa các script và notebook mới, rồi đặt `REPOSITORY_SOURCE` tới thư mục đó. Nếu bật mạng, có thể để notebook clone repository; hãy đảm bảo phiên bản trên GitHub đã chứa các tệp mới. `REPO_REVISION` cho phép chọn commit cụ thể trong bản clone mới. Notebook in Git SHA để ghi nhận nguồn thực thi.

### 2.4. Cache mô hình văn bản và phụ thuộc

Các checkpoint audio không thay thế toàn bộ tài nguyên tokenizer/model mà loader gốc cần. Chuẩn bị cache của đúng ba định danh mà nguồn expert gọi:

- `roberta-base`;
- `bert-base-uncased` cho E2: cần config và tokenizer; trọng số text encoder được phục hồi từ checkpoint MGA;
- `google-bert/bert-base-uncased`.

Điền `HF_CACHE_DIR` tới thư mục hub cache chứa trực tiếp `models--roberta-base/`, `models--bert-base-uncased/` và `models--google-bert--bert-base-uncased/`, gồm `refs/`, `snapshots/`, `blobs/` và các tài nguyên cần thiết. BERT dùng tên ngắn và tên có namespace có thể được lưu trong hai thư mục cache khác nhau; có một tên không bảo đảm tên còn lại hoạt động offline. Khi tạo dataset cache, cần bảo toàn nội dung các liên kết; không để liên kết trỏ ra ngoài dataset đã upload.

Notebook đặt cùng thư mục cho `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` và `runtime.cache_dir`, vì loader E3 truyền `cache_dir` tường minh. Không trỏ nhầm vào thư mục cha của hub cache. Tự dò chỉ dùng một số vị trí giới hạn trong Kaggle Input; nếu nhiều cache hợp lệ, phải chọn tường minh.

Nếu có mạng và được phép tải các tài nguyên này, bật `ALLOW_CACHE_DOWNLOAD=True`; notebook chuẩn bị đúng các định danh nêu trên. Đặt cache tải mới trong `/kaggle/working/hf_cache`, vì Kaggle Input chỉ đọc. Chuẩn bị/kiểm cache chạy qua `scripts/prepare_moe_text_cache.py` trong tiến trình mới, tránh cờ offline hoặc đường dẫn cache đã nạp trước đó trong kernel. E2 được kiểm bằng đúng `BertConfig` và `BertTokenizer` với cách tìm cache mặc định của loader gốc. Sau chuẩn bị, extraction vẫn offline.

Lệnh chuẩn bị cache riêng khi có mạng:

```bash
python scripts/prepare_moe_text_cache.py \
  --cache-dir /kaggle/working/hf_cache --download \
  --report-json /kaggle/working/moe_text_cache.json
```

Bỏ `--download` để chỉ kiểm tra offline. Nếu Expert 2 báo thiếu config trong cache trong khi preflight cũ đã PASS, kiểm tra riêng định danh `bert-base-uncased`; không đổi nó thành model ID khác trong YAML expert.

Tệp chung `requirements-kaggle.txt` khai báo các thư viện cho cả bốn expert và pipeline cổng, gồm `torchlibrosa`, `nnAudio` và thư viện vẽ biểu đồ. Một số phiên bản xử lý audio/model được chốt theo môi trường cục bộ đã xác minh; đây chưa phải bằng chứng một môi trường Kaggle mới đã chạy thành công toàn bộ.

Ô cấu hình mặc định `INSTALL_REQUIREMENTS=True`. Sau khi chuẩn bị source và trước khi import pipeline, notebook gọi `scripts/install_kaggle_dependencies.py` để cài tệp requirements. Installer sinh `moe_runtime_constraints.txt`, khóa phiên bản hiện có của PyTorch, TorchAudio, TorchVision, Triton và các gói CUDA `nvidia-*`, rồi đưa constraints vào pip. Nếu dependency mới xung đột với bộ GPU hiện có, pip phải báo lỗi thay vì âm thầm thay nó. Báo cáo gói đã cài/thay đổi được ghi tại `/kaggle/working/moe_dependency_install.json`.

Lệnh tương đương khi có Internet:

```bash
python scripts/install_kaggle_dependencies.py \
  --constraints /kaggle/working/moe_runtime_constraints.txt \
  --report-json /kaggle/working/moe_dependency_install.json
```

Khi chạy offline, đặt `WHEELHOUSE` tới thư mục wheel đã gắn trong Kaggle Input. Installer thêm `--no-index --find-links`, nên không tìm gói qua mạng:

```bash
python scripts/install_kaggle_dependencies.py \
  --constraints /kaggle/working/moe_runtime_constraints.txt \
  --wheelhouse /kaggle/input/<wheel-slug>/wheelhouse
```

Wheelhouse cần đủ cả dependency gián tiếp và tương thích phiên bản Python/hệ điều hành của Kaggle; chỉ chứa các wheel được liệt kê trực tiếp trong requirements là chưa đủ. Có thể chuẩn bị wheelhouse ở một môi trường có mạng tương thích bằng `python -m pip download -r requirements-kaggle.txt --dest wheelhouse`, sau đó kiểm tra dung lượng và các dependency GPU lớn trước khi upload. Các gói GPU đã có trong Kaggle tiếp tục bị khóa bởi constraints lúc cài.

Đặt `INSTALL_REQUIREMENTS=False` khi môi trường đã được chuẩn bị đầy đủ. `EXTRA_PACKAGES` chỉ dành cho gói bổ sung ngoài requirements. Không đưa PyTorch/CUDA vào danh sách này. Nếu pip thay phiên bản một thư viện đã được kernel nạp, notebook dừng và yêu cầu khởi động lại kernel Python rồi chạy từ đầu; không cần tạo một phiên Kaggle mới. NumPy được chốt dưới phiên bản 2 theo yêu cầu của LAION-CLAP, nên trường hợp này có thể xảy ra khi image Kaggle dùng NumPy 2.

Preflight kiểm tra import các thư viện, gồm `torchlibrosa`, `nnAudio` và Matplotlib, rồi factory tiếp tục kiểm tra các phụ thuộc sâu hơn. Cách dùng requirements và constraints dựa trên [tài liệu pip](https://pip.pypa.io/en/stable/cli/pip_install/).

Môi trường cục bộ đã dùng Python 3.10, NumPy 1.26, SciPy 1.15, cùng PyTorch/TorchAudio đồng bộ. Đây là thông tin đối chiếu, không phải yêu cầu cài lại image Kaggle giống hệt. Ghi phiên bản thực tế bằng ô chẩn đoán và chạy `smoke` trước extraction lớn.

## 3. Tạo và cấu hình notebook Kaggle

1. Tạo notebook Python mới trong Kaggle và nhập tệp `kaggle_train_moe_gating.ipynb` bằng chức năng nhập notebook của giao diện.
2. Mở phần cài đặt notebook, chọn bộ tăng tốc GPU ở mục `Accelerator`. Tên lựa chọn GPU phụ thuộc tài khoản và tài nguyên sẵn có.
3. Bật tùy chọn Internet nếu cần clone source hoặc tải tài nguyên đã được cho phép. Khi mọi tài nguyên đã mount cục bộ, đặt các cờ tải về thành `False` và chạy offline.
4. Dùng chức năng thêm dữ liệu đầu vào, thường được ghi là `Add Input`, để đính kèm XACLE, `xacle-weights`, cache Hugging Face và source nếu chạy offline.
5. Điền các đường dẫn trong ô cấu hình. Giữ mặc định:

   ```python
   PROJECT_ROOT = Path('/kaggle/working/MOESCORE')
   FEATURE_STORE_ROOT = Path('/kaggle/working/moe_feature_store')
   OUTPUT_DIR = Path('/kaggle/working/moe_gating')
   ```

6. Chọn `RUN_MODE='preflight'`, chạy từ trên xuống để kiểm tra GPU, VRAM, Input, số cặp, WAV, checkpoint, nguồn UniLM và cache. Đọc `/kaggle/working/moe_preflight/preflight.json` nếu cần tra chi tiết.
7. Chọn `smoke` để chạy tối đa bốn mẫu đầu mỗi split qua bốn expert thật, rồi đúng ba bước optimizer cổng trên CPU. Store smoke dùng root riêng có hậu tố `_smoke`. Đây chỉ là kiểm tra chức năng.
8. Khi smoke thành công, chọn `full` và chạy từ trên xuống để thực hiện toàn bộ quy trình. Không cần gọi ô theo thứ tự thủ công khác.

Tên hoặc vị trí nút trên Kaggle có thể thay đổi. Các bước cần đạt là gắn Input, bật GPU và chọn chính sách mạng; ô chẩn đoán xác nhận trạng thái thật thay vì dựa vào tên nút.

## 4. Các chế độ chạy

| `RUN_MODE` | Hành vi và điều kiện |
|---|---|
| `preflight` | Kiểm tra môi trường, dữ liệu và tài nguyên; không dựng expert. |
| `smoke` | Trích xuất thật tối đa bốn mẫu mỗi split vào store riêng và chạy ba bước gate; không tạo kết quả toàn tập. |
| `extract` | Trích xuất và xác minh đầy đủ ba store; chưa huấn luyện gate. |
| `train` | Yêu cầu store đầy đủ, bắt đầu huấn luyện mới, từ chối output đã có checkpoint; sau đó đánh giá cuối. |
| `resume` | Yêu cầu store tương thích và checkpoint latest; tiếp tục huấn luyện rồi đánh giá cuối. |
| `evaluate` | Yêu cầu best checkpoint và store đầy đủ; chỉ đánh giá validation/test và xuất kết quả. |
| `full` | Trích xuất hoặc tái sử dụng store hợp lệ, huấn luyện hoặc tiếp tục latest, đánh giá cuối và xuất artifact. |

Tất cả chế độ của notebook Kaggle đều yêu cầu GPU và các Input đã khai báo. Gate-only có thể chạy CPU bằng CLI bên ngoài notebook; notebook này ưu tiên quy trình Kaggle GPU theo yêu cầu.

## 5. Trích xuất feature toàn tập

Mỗi subprocess giữ đúng một expert sống và xử lý cả train, validation, test trước khi kết thúc. Tiến trình tiếp theo chỉ bắt đầu sau khi tiến trình trước đã thoát, nhờ đó bộ nhớ GPU được giải phóng cùng toàn bộ CUDA context.

Mỗi batch dùng `torch.inference_mode()`, giữ FP32, đúng waveform/caption của Unified Dataset và đúng độ dài hợp lệ E4. Output chuyển về CPU rồi lưu shard NPZ trung gian không có object/pickle. Khi đủ bốn expert cho một shard, assembler hiện hành ghép score `[N,4]` và feature `[N,7936]`; writer hiện hành lưu store NPY hoàn chỉnh.

```text
moe_feature_store/
├── _extraction/
│   ├── job.json
│   ├── train_manifest.json
│   ├── validation_manifest.json
│   ├── test_manifest.json
│   ├── expert1_runtime.json
│   └── expert1/... expert2/... expert3/... expert4/...
├── train/manifest.json, source_manifest.json, shards/...
├── validation/manifest.json, source_manifest.json, shards/...
└── test/manifest.json, source_manifest.json, shards/...
```

Manifest liên kết ID, filename, caption nguyên gốc, hash WAV, hash checkpoint và nguồn preprocessing. Thứ tự CSV được bảo toàn. Hash nội dung checkpoint dùng để nhận diện khi đổi vị trí mount giữa các phiên; đường dẫn hiện tại xuất trong cấu hình/báo cáo tài nguyên. Không dùng tên file như bằng chứng duy nhất về checkpoint.

`EXTRACTION_BATCH_SIZES` mặc định bằng 1 cho cả bốn expert. Bắt đầu nhỏ và chỉ tăng sau khi đo tài nguyên thực tế. `SHARD_SIZE=128` giới hạn số output CPU lưu trong mỗi shard; nó khác batch size inference. Không chuyển FP16, quantization hoặc offload khi gặp OOM.

Feature cache dùng preprocessing đánh giá xác định. Với E1, clip dài được center crop; không tạo random crop mới mỗi epoch. Đây là lựa chọn huấn luyện cổng trên output expert đóng băng, cần ghi nhận khi so sánh với huấn luyện online có augmentation.

Tiến độ hiển thị bằng tqdm và báo cáo tài nguyên ghi RSS, RAM khả dụng, CUDA allocated/reserved/peak/free tại các mốc. Một GPU được dùng mỗi lần; không thực hiện distributed inference hoặc nạp đồng thời cả bốn expert.

Lệnh tương đương ngoài notebook:

```bash
python scripts/extract_moe_feature_store.py \
  --config /kaggle/working/moe_expert_config.yaml \
  --dataset-root /kaggle/input/<slug>/XACLE_dataset \
  --output-root /kaggle/working/moe_feature_store \
  --device cuda:0 --batch-sizes 1,1,1,1 --shard-size 128
```

### Khôi phục extraction

Chạy lại cùng lệnh/mode sau gián đoạn. Shard có đủ metadata, checksum và identity được xác minh rồi bỏ qua; worker không dựng model nếu toàn bộ shard của expert đã có. Shard chưa commit đầy đủ được tạo lại. Shard đã commit nhưng hỏng checksum gây lỗi rõ ràng; giữ bản sao để điều tra, rồi chỉ loại bỏ cặp shard lỗi khỏi bản writable nếu đã xác định cần tính lại. Không sửa checksum để ép chấp nhận dữ liệu.

Có thể giảm riêng batch inference sau OOM và chạy lại; identity nội dung không bao gồm batch size. Giữ nguyên shard size, dữ liệu, checkpoint, caption và mã preprocessing. Đổi các thành phần đó cần root mới. Nếu split đã hoàn chỉnh và khớp provenance, extractor tái sử dụng mà không nạp lại model.

Muốn dùng store từ phiên trước, đặt `EXISTING_FEATURE_STORE_ROOT` tới root đã gắn qua Kaggle Input. Notebook xác minh rồi sao chép sang root writable. Bản input nên giữ đủ manifest, source manifest, shard và `_extraction` nếu muốn tiếp tục extraction dở. Cấu hình hiện tại phải tương thích nguồn/checkpoint đã ghi.

## 6. Huấn luyện gate

Notebook gọi trực tiếp:

```bash
python scripts/train_moe_gating.py \
  --config /kaggle/working/moe_gating_config.yaml \
  --report-json /kaggle/working/moe_gating/training_run.json
```

Cấu hình được sinh theo schema hiện tại:

| Tham số | Giá trị |
|---|---:|
| Seed | 2027 |
| Batch train / validation / inference | 8 / 8 / 64 |
| Số epoch tối đa | 50 |
| Learning rate | 0.00001 |
| Weight decay | 0.0001 |
| Diversity weight | 0.2 |
| Early-stop patience | 50 |
| Chu kỳ checkpoint theo step | 1000 |
| Loss weights | `[1.0, 0.5]` |
| Clipped MSE tau | 0.25 |
| Contrastive margin | 0.1 |
| Class balancing | `false` |

Kiến trúc giữ `Linear(7936,512) → LayerNorm → ReLU → Dropout(0.1) → Linear(512,4) → Softmax`. Dự đoán là tổng score chuyên gia nhân weight theo từng mẫu. Không thêm score calibration hoặc regression head.

Trainer đọc score/feature cố định từ store, vì vậy không nạp lại checkpoint expert. Chỉ tham số gate vào AdamW. Nhãn raw giữ [0,10] trong store, được chuyển `(raw-5)/5` tại loss boundary; prediction `[B,1]` được chuyển thành vector `[B]` khi gọi loss để tránh broadcasting sai. Scheduler hiện hành nhận loss validation sau mỗi epoch. Checkpoint tốt nhất vẫn chọn bằng SRCC validation; test không tham gia.

Log từng epoch ghi train loss, metric validation, global step và SRCC tốt nhất. Loss ngắn hạn không chứng minh chất lượng tổng quát; so sánh bằng protocol validation đã định. Nếu SRCC không xác định, không tạo checkpoint tốt nhất giả hoặc dùng latest thay thế âm thầm.

`latest_gate.pt` lưu gate, optimizer, scheduler, RNG, sampler, epoch và global step cùng config/provenance để tiếp tục. `best_gate.pt` là snapshot tại SRCC validation tốt nhất. Cả hai không chứa bốn expert pretrained.

### Khôi phục huấn luyện

Trong cùng phiên, chọn `resume`; mặc định dùng `OUTPUT_DIR/latest_gate.pt`. Có thể đặt `RESUME_CHECKPOINT` tường minh. Ở phiên mới, gắn output cũ làm Input và đặt `RESTORE_OUTPUT_ROOT` tới thư mục chứa cả latest và best; notebook sao chép sang `OUTPUT_DIR` trống. Giữ store cùng identity, cấu hình tương thích và checkpoint expert gốc.

```bash
python scripts/train_moe_gating.py \
  --config /kaggle/working/moe_gating_config.yaml \
  --resume /kaggle/working/moe_gating/latest_gate.pt \
  --report-json /kaggle/working/moe_gating/training_run.json
```

Lịch sử epoch cũ được ghép với phần vừa tiếp tục theo số epoch vào `training_history.json`. Không chỉ sao chép best nếu muốn resume chính xác; cần latest, store và metadata đi kèm.

## 7. Validation, test và metric

Sau huấn luyện, notebook nạp `best_gate.pt`, kiểm checkpoint provenance, preprocessing contract và identity validation. Gate ở eval/no-grad; không tạo optimizer trong bước đánh giá cuối.

Validation dùng toàn bộ 3.000 mẫu. Test suy luận toàn bộ 3.000 mẫu bằng batch 64 với store không nhãn. Prediction test đầy đủ độ chính xác được lưu trước khi đọc `test_average.csv`. Sau đó ghép nhãn theo sample ID, filename và caption; kiểm tập ID không thiếu/trùng/thừa. Thứ tự CSV nhãn khác thứ tự prediction vẫn ghép đúng theo ID.

Mọi metric gọi lại duy nhất `compute_metrics()` trong `training/moe_evaluator.py`, trên thang raw `pred*5+5`. Không lấy metric từ CSV submission đã làm tròn; không lấy trung bình correlation từng batch.

| Metric | Ý nghĩa | Hướng tốt hơn |
|---|---|---|
| SRCC | Tương quan thứ hạng Spearman | Cao hơn, tối đa 1 |
| LCC | Tương quan tuyến tính Pearson | Cao hơn, tối đa 1 |
| KTAU | Kendall tau-b có xử lý đồng hạng | Cao hơn, tối đa 1 |
| MSE | Bình phương sai số trung bình trên thang [0,10] | Thấp hơn, tối thiểu 0 |

MAE và các trạng thái correlation tiếp tục được giữ. Khi prediction hoặc nhãn hằng, correlation trả `null` cùng trạng thái; không thay bằng 0. Gate có thể cho raw prediction ngoài [0,10] nếu score expert vượt khoảng chuẩn hóa; pipeline giữ nguyên phép tính, không tự clipping.

Test labels chỉ phục vụ đánh giá sau cùng. Không dùng chúng để fine-tune, tạo feature, train gate, scheduler, chọn checkpoint, early stopping hoặc chỉnh hyperparameter sau khi xem test.

## 8. Kết quả và bảo toàn artifact

```text
/kaggle/working/moe_gating/
├── best_gate.pt
├── latest_gate.pt
├── metrics.json
├── validation_metrics.json
├── test_metrics.json
├── validation_predictions.csv
├── test_predictions.csv
├── submission.csv
├── training_history.json
├── training_config.yaml
├── training_run.json
└── training_curves.png
```

`metrics.json` chứa `validation` và `test`, mỗi phần có `sample_count`, `SRCC`, `LCC`, `KTAU`, `MSE`, MAE và trạng thái correlation. Báo cáo riêng có checkpoint SHA-256, epoch tốt nhất, số training step, seed, Git SHA, store identity, config và thời điểm đánh giá. Notebook in bảng hai split bằng dữ liệu đã tính thật.

`validation_predictions.csv` và `test_predictions.csv` giữ đầy đủ độ chính xác số thực. `submission.csv` giữ hành vi xuất hai chữ số hiện tại, header `wav_file_name,pred_score`. Cần đối chiếu quy cách nộp chính thức 2027 trước khi nộp; mã upstream chưa đủ làm bằng chứng xác nhận mọi yêu cầu submission 2027.

Notebook tạo `/kaggle/working/moe_gating_results.zip`. Tải tệp này hoặc lưu output phiên bằng chức năng lưu phiên bản/output của Kaggle. Muốn giữ feature store, có thể bật `SAVE_FEATURE_STORE_ARCHIVE`; cần thêm dung lượng cho archive, hoặc lưu thư mục store như artifact dataset riêng. Đừng giả định `/kaggle/working` tồn tại trong phiên mới không liên quan.

Ma trận feature FP32 chiếm khoảng 227 MiB train, 91 MiB validation và 91 MiB test. Shard trung gian giữ thêm một bản score/features, checkpoint/runtime/source cũng cần đĩa. Dung lượng VRAM/RAM phụ thuộc từng expert và batch size; xem báo cáo thực tế. Không cộng các peak độc lập để kết luận khả năng nạp đồng thời bốn expert.

Khi chuyển phiên: đính kèm artifact đã lưu vào Input, xác minh đủ checksum/manifest, copy sang Working nếu cần ghi tiếp, rồi chọn `resume`, `evaluate` hoặc `extract` phù hợp.

## 9. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| Không thấy hoặc nhiều dataset root | Kiểm `meta_data/` và `wav/`, điền `DATASET_ROOT` chính xác; không dùng root của tệp listener-level. |
| Sai số lượng cặp, thiếu WAV hoặc trùng ID | Kiểm lại dataset upload và split tương ứng. Không bỏ qua dòng để ép đủ shape. |
| Thiếu checkpoint | Kiểm đúng tám đường dẫn đã chốt, tên mount và quyền đọc; không đổi tên model khác để thay thế. |
| Checkpoint sai schema/key/shape/SHA | Dùng checkpoint đã xác minh đúng expert; đọc exception factory. Không dùng `strict=False` bổ sung hoặc bỏ fingerprint. |
| CUDA không khả dụng | Bật GPU trong cài đặt và khởi động lại phiên nếu cần; chạy lại ô môi trường. |
| CUDA OOM | Giảm batch riêng expert về 1; dừng tiến trình lỗi, xác nhận VRAM trống rồi chạy lại extraction. Giữ SHARD_SIZE và provenance. Nếu B=1 vẫn lỗi, cần GPU có đủ VRAM. |
| Thiếu hoặc xung đột phụ thuộc | Xem tên gói và traceback, chuẩn bị wheel tương thích hoặc cài gói tường minh. Không cài lại toàn bộ PyTorch/CUDA theo một requirements cũ. |
| Cache tokenizer/model không đủ | Điền đúng hub cache, kiểm đủ ba tên ở mục 2.4. Đặc biệt E2 cần `bert-base-uncased`, còn E3 gọi `google-bert/bert-base-uncased`. Chạy lại preflight có cho phép tải, hoặc CLI chuẩn bị cache trong tiến trình mới. Không thay model ID trong expert. |
| UniLM import/fingerprint không khớp | Dùng source đi kèm đúng revision; chỉ hai khác biệt import đã rà soát được chấp nhận. Đừng sửa model rồi cập nhật fingerprint cho khớp giả. |
| Provenance khác khi resume | So checkpoint SHA, source/preprocessing hash, ID/CSV và config. Khôi phục đúng artifact hoặc dùng root mới cho một lượt trích xuất mới. |
| Shard checksum sai | Giữ log và bản lỗi; khôi phục shard tốt từ artifact gốc hoặc tính lại đúng shard trên bản writable. Không sửa checksum thủ công. |
| Extraction bị ngắt | Giữ `_extraction` và final shard, chạy lại cùng root; worker xác minh rồi bỏ qua phần đã commit. |
| Huấn luyện bị ngắt | Phục hồi latest và cả best cùng store/config, chọn resume; không chỉ nạp weight rồi bỏ optimizer/RNG/sampler. |
| Không thấy best/prediction | Kiểm train có hoàn tất epoch validation, SRCC có được định nghĩa, mode có đến đánh giá cuối và exception đầu tiên. Không dùng checkpoint giả thay best. |
| Correlation trả null | Kiểm số mẫu, prediction/nhãn hằng và trường `*_status`; đây có thể là metric không xác định hợp lệ. |
| Test thiếu nhãn hoặc ID/caption không khớp | Suy luận có thể đã được lưu, nhưng đánh giá test phải dừng. Đính kèm đúng `test_average.csv`, không zip theo vị trí hoặc tự tạo nhãn. |

Nếu cập nhật mã preprocessing, checkpoint hoặc dữ liệu sau khi đã tạo store, nên giữ artifact cũ cho tái lập và dùng root mới. Điều này tránh ghép output từ hai phiên bản khác nhau.

## 10. Phạm vi xác minh và quy định sử dụng

Notebook được cung cấp để chạy thật trên Kaggle; các test cục bộ dùng fixture nhỏ để kiểm phần mềm, không tạo dữ liệu challenge mới. Kết quả chạy kiểm thử và giới hạn thực tế ghi ở `docs/KAGGLE_IMPLEMENTATION_VERIFICATION_VI.md`. Không có kết luận chất lượng cuộc thi hoặc số liệu full Kaggle nếu chưa thực hiện phiên đó.

Kiến trúc cuối vẫn là cổng trainable phụ thuộc mẫu kết hợp expert đóng băng. Không dùng average/voting hoặc weight cố định. Cần tuân thủ quyền sử dụng dữ liệu và pretrained resources của cuộc thi. Việc cache frozen feature là cách triển khai kỹ thuật; nếu cách diễn giải quy định MoE nội bộ chưa rõ, cần xác nhận với ban tổ chức trước khi nộp. Hướng dẫn này không thay cho phê duyệt chính thức.

## 11. Danh sách bắt đầu nhanh

1. Đính kèm XACLE đúng cấu trúc, tám checkpoint đúng tên, source mới và cache text model.
2. Bật GPU, chọn chính sách Internet và kiểm đường dẫn ở ô cấu hình.
3. Chạy `preflight` từ trên xuống; xử lý toàn bộ lỗi tài nguyên.
4. Chạy `smoke`; kiểm output thật của bốn expert và ba bước gate giới hạn.
5. Chọn `full`; giữ loss `[1.0,0.5]`, diversity `0.2` và seed `2027`.
6. Kiểm store đủ 7.500/3.000/3.000, không có nhãn trong test store.
7. Theo dõi log epoch và best SRCC validation; dùng `resume` khi bị ngắt.
8. Chờ đánh giá cuối bằng best gate, đọc bảng SRCC/LCC/KTAU/MSE của validation/test.
9. Lưu checkpoint, config, manifest, JSON/CSV và feature store ra artifact bền vững trước khi kết thúc phiên.
