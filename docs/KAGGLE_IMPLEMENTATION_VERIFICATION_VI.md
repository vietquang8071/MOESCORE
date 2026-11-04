# Báo cáo xác minh triển khai notebook Kaggle MOESCORE

## 1. Kết quả và phạm vi

Đã triển khai notebook `notebooks/kaggle_train_moe_gating.ipynb`, hướng dẫn `docs/KAGGLE_GATING_GUIDE_VI.md`, extractor toàn tập theo shard, và bước đánh giá cuối sử dụng lại evaluator/metric hiện có.

Kết quả phần mềm: **275/275 regression test PASS**. Có **7 test mới**, bao gồm nhiều nhánh kiểm tra identity, resume, hỏng artifact, nhãn test, notebook và báo cáo metric. Một phép thử riêng trên **một WAV thật của Expert 4** xác nhận nguồn UniLM đi kèm phục hồi checkpoint và suy luận thành công.

**Chưa chạy một phiên Kaggle toàn tập**, chưa trích xuất 13.500 cặp bằng extractor mới, chưa huấn luyện gate 50 epoch, chưa có metric validation/test toàn tập từ notebook này. Các test phần mềm dùng fixture nhỏ được ghi rõ dưới đây. Không chuyển số liệu fixture thành kết quả XACLE.

## 2. Căn cứ và trạng thái repository

- Nhánh ban đầu: `main`, working tree sạch trước triển khai.
- Commit ban đầu: `79d8b9a47e1dc2d64201ef71f15fb5ca15c47140` — `add lcc and ktau`.
- Đã đọc registry/factory, Unified Dataset/Collate, writer/reader store, trainer/evaluator/checkpoint/target transform, CLI train/evaluate/infer, cấu hình và tài liệu LOCAL-10/11.
- Đã đối chiếu bài báo `MOESCORE.pdf` và [repository phương pháp gốc](https://github.com/S-Orion/MOESCORE). Bài báo mô tả fine-tune expert độc lập, đóng băng expert và tối ưu gate. Mã hiện tại quyết định kiến trúc và cấu hình thực thi.
- `build_moe_feature_store.py` chỉ nhập artifact giới hạn từ verifier cũ; không có extractor toàn tập trước thay đổi này.
- Giữ cấu hình đã chốt: batch 8/8/64, 50 epoch, learning rate 1e-5, weight decay 1e-4, loss weights `[1.0,0.5]`, diversity 0.2. Chỉ sửa comment cấu hình từng mô tả hệ số cũ để tránh hiểu nhầm.

## 3. Thành phần triển khai

Notebook có **33 ô**, gồm 20 phần nội dung và các ô Python gọi thành phần tái sử dụng. Các chế độ `preflight`, `smoke`, `extract`, `train`, `resume`, `evaluate`, `full` đều có nhánh thực thi tương ứng. Mặc định là `full`.

`scripts/extract_moe_feature_store.py`:

1. Kiểm CSV pair-level và số lượng, tạo manifest cố định theo thứ tự CSV, hash WAV/checkpoint/source.
2. Loại cột nhãn khỏi metadata test trước khi khởi tạo Unified Dataset; không chuyển nhãn hoặc model metadata vào expert forward.
3. Chạy một subprocess cho mỗi expert, dùng factory hiện có, giữ expert cho cả ba split rồi thoát trước expert tiếp theo.
4. Xác minh frozen/eval, checkpoint không đổi trong restoration, module origins và FP32 output.
5. Xử lý từng mini-batch trong inference mode; lưu shard CPU nguyên tử kèm checksum và sample ID.
6. Resume xác minh shard đã commit, từ chối hỏng checksum, ID sai hoặc provenance thay đổi. Cho phép giảm inference batch size sau OOM, giữ nguyên shard size và dữ liệu nguồn.
7. Dùng assembler hiện có và `MoeFeatureStoreWriter` hiện có để tạo store `[N,4]`, `[N,7936]`; store test không chứa labels.
8. Bỏ qua dựng model khi shard/store hoàn chỉnh đã được xác minh tương thích.

`scripts/kaggle_moe_pipeline.py` chứa tìm root dữ liệu an toàn, sinh config theo schema hiện tại, kiểm tài nguyên/cache, kiểm store nguồn, chạy smoke ba bước giới hạn, đánh giá best checkpoint và xuất kết quả. Notebook gọi nguyên `scripts/train_moe_gating.py`, không viết lại vòng huấn luyện.

## 4. Nguồn UniLM và kiểm chứng Expert 4 thật

Đối chiếu trực tiếp nguồn đi kèm với clone gốc đã xác minh cho thấy:

- `BEATs.py`: `from backbone import (` khác `from .backbone import (`.
- `backbone.py`: `from modules import (` khác `from .modules import (`.
- Ba tệp được hash theo đúng thứ tự và tên mà loader gốc sử dụng; mọi byte khác phải giữ nguyên để khôi phục đúng hash gốc.

`expert4_finetune/source_compatibility.py` chỉ chấp nhận fingerprint huấn luyện đã rà soát:

```text
git_commit: 31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd
sha256: fac0525f18089a7163625272d58ba94e2adee203c4e7898492e003b6194e7487
```

Nếu identity nguyên bản không khớp, loader chỉ cho phép nguồn có hash byte sau đảo hai import khớp chính xác fingerprint này. Không sửa metadata checkpoint, không giả lập Git commit, không thay state dict và không mở fallback pretrained-only. Test xác nhận một sửa đổi nguồn ngoài phạm vi, kể cả byte bổ sung, bị từ chối.

Đã thực thi `/tmp/moescore_kaggle_e4_smoke.py` bằng Conda `nlp`, offline, FP32, `cuda:0` trên GTX 1650. Dùng factory thật, checkpoint thật và Unified Dataset cho `validation:07500.wav`:

| Hạng mục | Quan sát |
|---|---|
| Phục hồi checkpoint qua loader gốc | PASS |
| Nguồn UniLM đi kèm, equivalence import | PASS |
| Score shape | `[1,1]` |
| Feature shape | `[1,1024]` |
| Score | `0.7870628237724304` |
| Score và toàn bộ feature hữu hạn | Có |
| Toàn bộ tham số frozen, mọi module eval | Có |
| Sai khác tuyệt đối lớn nhất giữa hai score lặp | `0.0` |
| Sai khác tuyệt đối lớn nhất giữa hai feature lặp | `0.0` |

Đây là phép thử một expert trên một mẫu, không phải metric đánh giá hoặc bằng chứng bốn expert chạy đồng thời.

Đo tài nguyên của phép thử này, đơn vị byte:

| Mốc | RSS | CUDA allocated | CUDA reserved | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| Trước factory | 513396736 | 0 | 0 | 0 | 0 |
| Sau factory | 2377129984 | 883738112 | 958398464 | 883738112 | 958398464 |
| Sau forward | 2623086592 | 911781888 | 1080033280 | 1000438784 | 1080033280 |

VRAM tổng Torch quan sát: `3895132160` byte. JSON chi tiết: `/tmp/moescore_kaggle_e4_smoke.json`; log: `/tmp/moescore_kaggle_e4_smoke.log`. Đây là artifact tạm cục bộ, có thể mất khi dọn `/tmp`.

## 5. Ranh giới huấn luyện, validation và test

Kiến trúc gate, phép tổng bốn score có weight, target transform, loss, optimizer, scheduler và chọn checkpoint giữ nguyên. `train_moe_gating.py` chỉ thêm log epoch ra stderr. Best checkpoint tiếp tục chọn duy nhất bằng SRCC validation.

`MoeGateEvaluator.predict()` có tùy chọn keyword `round_scores=False`; mặc định vẫn `True`, bảo toàn hành vi submission hiện hành. Bước đánh giá cuối dùng option mới để tính metric trước khi làm tròn, không viết lại phép fusion hoặc metric.

Validation chạy evaluator hiện có trên toàn bộ loader. Test chạy label-free trước, ghi prediction đầy đủ độ chính xác, rồi mới đọc `average_score` để ghép nhãn theo ID, filename và caption. Hàm `compute_metrics()` hiện có tính SRCC, LCC, KTAU, MSE và MAE trên thang raw. Báo cáo xuất thêm tên hiển thị viết hoa; các khóa và trạng thái gốc vẫn giữ trong `metrics_raw_scale`.

Kiểm tra cuối còn so checkpoint expert provenance, validation store identity và toàn bộ preprocessing identity giữa validation/test. Không gọi optimizer trong đánh giá. `test_predictions.csv` giữ số đầy đủ độ chính xác; `submission.csv` giữ hai chữ số như luồng cũ.

Các test chạy một lượt train fixture bốn mẫu, hai optimizer step với loss `[1.0,0.5]`, chọn best, đánh giá hai split fixture và xuất JSON/CSV. CSV nhãn test bị đảo thứ tự để xác nhận ghép theo ID. Test khác chặn mọi lời gọi hàm đọc nhãn test trong lúc `run_training()` chạy. Đây là test phần mềm; không huấn luyện trên dữ liệu challenge thật trong tác vụ này.

## 6. Kiểm thử đã thực hiện

Kiểm thử tập trung:

```bash
conda run -n nlp --no-capture-output python3 \
  -m unittest discover -s tests -p 'test_kaggle*.py' -v
```

Kết quả: **7/7 PASS**, lần chạy tập trung cuối trước regression khoảng 4,8 giây. Bao phủ:

- Tìm dataset, từ chối nhiều candidate và cho chọn root tường minh.
- Source BEATs tương đương đúng hai import, từ chối identity khác hoặc sửa nguồn khác.
- Projection test không đọc giá trị nhãn để xử lý; fixture chứa giá trị score không hợp lệ vẫn trích xuất test label-free.
- Bốn stub expert đi qua preprocessing thật của fixture, shards, assembler và writer thật.
- Gián đoạn sau shard đầu và tiếp tục; không dựng lại expert khi mọi shard hợp lệ.
- Thay batch size sau gián đoạn, kiểm hash chunk, hash manifest và checkpoint thay đổi.
- Notebook JSON, compile tất cả code cell; thực thi các code cell theo thứ tự ở mode preflight với GPU/cache giả lập trong test.
- Full-precision prediction giữ giá trị khác bản submission đã làm tròn.
- Huấn luyện fixture, chọn best theo validation SRCC, đánh giá toàn split, test label matching, đủ metric JSON và rejection khi preprocessing identity test khác.

Xác minh `.ipynb` bằng `nbformat` 5.9.1 có sẵn trong Python hệ thống:

```bash
/usr/bin/python3 -c 'import nbformat; n=nbformat.read("notebooks/kaggle_train_moe_gating.ipynb", as_version=4); nbformat.validate(n)'
```

Kết quả: **PASS**, 33 ô hợp lệ. Mọi ô Python được `compile()` thành công. Môi trường `nlp` không có nbformat; không cài thêm gói, và unit test thông thường dùng JSON/compile để không tạo phụ thuộc runtime mới. Việc kiểm schema bằng nbformat vẫn đã được thực thi riêng như trên.

Compile các tệp Python mới/sửa bằng `python3 -m py_compile`: **PASS**. `git diff --check`: **PASS**.

Regression đầy đủ:

```bash
conda run -n nlp --no-capture-output python3 \
  -m unittest discover -s tests -p 'test_*.py' -v
```

Kết quả thực tế: **275 test, 275 PASS, 0 lỗi, 0 test bị bỏ qua**, thời gian `112.634s`, exit code `0`. Bộ test hiện có được giữ nguyên; thêm bảy test. Log: `/tmp/moescore_kaggle_regression.log`. Có cảnh báo `FutureWarning` từ import timm ở test hiện có; không làm test thất bại.

## 7. Tệp tạo và sửa

Tạo:

- `notebooks/kaggle_train_moe_gating.ipynb`
- `docs/KAGGLE_GATING_GUIDE_VI.md`
- `docs/KAGGLE_IMPLEMENTATION_VERIFICATION_VI.md`
- `scripts/extract_moe_feature_store.py`
- `scripts/kaggle_moe_pipeline.py`
- `expert4_finetune/source_compatibility.py`
- `tests/test_kaggle_feature_extraction.py`
- `tests/test_kaggle_notebook_pipeline.py`

Sửa:

- `expert4_finetune/inference.py`: fallback kiểm chứng equivalence đúng hai import, chỉ cho fingerprint đã rà soát.
- `training/moe_evaluator.py`: option prediction đầy đủ độ chính xác, mặc định cũ giữ nguyên.
- `scripts/train_moe_gating.py`: log tiến độ epoch ra stderr.
- `configs/moe_cloud.example.yaml`: sửa comment về hệ số loss đã chốt; giá trị YAML không đổi.
- `docs/LOCAL-11/cloud-execution-plan.md`: dẫn tới hướng dẫn mới và phân biệt trạng thái lịch sử.

Không thay kiến trúc, score, attention mask, checkpoint expert hoặc công thức loss. Không stage/commit/push trong tác vụ này.

## 8. Giới hạn và bước chạy tiếp

Chưa thực thi đủ 33 ô ở mode full trên Kaggle. Test thứ tự ô hiện kiểm preflight bằng tài nguyên giả lập; đường đi extraction/train/evaluation được kiểm độc lập bằng fixture, và E4 có một phép thử thật riêng. Chưa đo thời gian/dung lượng của extractor trên 13.500 cặp, chưa thử OOM hay gián đoạn phiên Kaggle thật, chưa có metric validation/test toàn tập từ notebook mới.

Kaggle cần đúng tám checkpoint, source hiện tại, data đầy đủ và cache text model. Image Kaggle khác môi trường cục bộ có thể cần wheel/phụ thuộc tương thích; notebook chỉ báo lỗi hoặc cài gói khi người dùng cấu hình tường minh. Không tự thay PyTorch/CUDA. Full extraction vẫn tính cả bốn nhánh preprocessing Unified cho mỗi item trước khi route nhánh cần dùng; đây là lựa chọn bảo toàn đường đi đã xác minh và có thể tốn CPU.

Chạy `preflight`, sau đó `smoke`, rồi `full` theo hướng dẫn. Lưu feature store, checkpoint, config và manifest ra artifact bền vững giữa các phiên. Quy cách submission 2027 và cách diễn giải quy định MoE/cache cần được đối chiếu với ban tổ chức trước khi nộp; notebook không tuyên bố phê duyệt chính thức.

## 9. Bổ sung requirements và cài đặt trong notebook

Đã bổ sung `requirements-kaggle.txt`, `scripts/install_kaggle_dependencies.py` và `tests/test_kaggle_dependencies.py`. Notebook mặc định `INSTALL_REQUIREMENTS=True`, gọi installer trước khi import pipeline. Preflight bổ sung kiểm tra `torchlibrosa`, `nnAudio` và Matplotlib; hướng dẫn sử dụng được cập nhật cho cài online/offline và trường hợp cần khởi động lại kernel.

Tệp requirements chốt các phiên bản xử lý âm thanh/mô hình theo môi trường cục bộ đã xác minh. Installer dùng constraints giữ nguyên PyTorch, TorchAudio, TorchVision, Triton và các distribution `nvidia-*` hiện có. Gói thêm vào `EXTRA_PACKAGES` không được thay bộ GPU hoặc truyền cờ pip. Khi cung cấp `WHEELHOUSE`, pip dùng `--no-index --find-links`. Nếu package vừa được thay phiên bản đang nằm trong kernel, notebook dừng và yêu cầu khởi động lại kernel rồi chạy từ đầu.

Kiểm thử sau bổ sung:

- `python3 -m unittest discover -s tests -p 'test_kaggle*.py' -v`: **11/11 PASS**, `6.422s`, gồm bốn test installer mới và bảy test trước đó. Test gọi pip qua mock, không thực hiện cài đặt.
- Pip `--dry-run --no-index` với requirements và constraints GPU hiện tại: **PASS**, mọi yêu cầu đã được đáp ứng trong môi trường `nlp`. Không tải hoặc cài gói.
- `nbformat.validate`, compile mọi code cell và compile Python của installer/test: **PASS**.
- `git diff --check`: **PASS** trước cập nhật tài liệu này.

Regression 275 test ở mục 6 là lần chạy đầy đủ trước phần bổ sung requirements. Sau bổ sung chỉ chạy lại nhóm Kaggle phù hợp, không khẳng định đã chạy lại toàn suite hoặc đã thử cài trong một image Kaggle sạch. Log mới: `/tmp/moescore_kaggle_dependencies_tests.log`, `/tmp/moescore_requirements_dryrun.log`.

## 10. Đồng bộ đường dẫn theo dataset weight thực tế

Người dùng cung cấp danh sách tám file trong `/kaggle/input/datasets/qnv8071/xacle-weights/weights`. Thư mục fine-tuned thực tế là `finetuned_weights/`; checkpoint E3/E4 tương ứng mang tên `pretrained/checkpoint-30.pth` và `pretrained/BEATs_iter3_plus_AS2M.pt`.

Tại thời điểm sửa, commit `5da3bc7` của người dùng đã cập nhật các tên này trong generator, trong khi mapping kiểm tra checkpoint ở notebook còn dùng tên cũ. Đã giữ thay đổi của người dùng, cập nhật root và mapping notebook, đồng thời bổ sung tham số tùy chọn `expert_checkpoints` cho generator để chính mapping người dùng được áp dụng vào cả YAML đã lưu và cấu hình runtime. Không đổi tên hoặc chỉnh nội dung checkpoint.

Đã cập nhật fixture và hướng dẫn theo cấu trúc thực tế, thêm regression test so sánh checkpoint mapping người dùng với YAML/runtime. Kết quả: **12/12 test `test_kaggle*.py` PASS**, thời gian `6.349s`; notebook hợp lệ theo nbformat và mọi code cell compile thành công; `git diff --check` PASS trước cập nhật báo cáo này. Log: `/tmp/moescore_checkpoint_paths_tests.log`. Không có phép thử inference hoặc full Kaggle mới trong phần sửa đường dẫn này.

## 11. Sửa cache BERT của Expert 2

Traceback Kaggle cho thấy factory E2 dừng tại `BertConfig.from_pretrained('bert-base-uncased')` do cache thiếu file trong chế độ offline. Nguồn E2 xác nhận ID tên ngắn này, trong khi preflight cũ chỉ kiểm `roberta-base` và `google-bert/bert-base-uncased`. Vì vậy preflight cũ chưa bảo đảm cache đáp ứng đúng cách lookup của E2.

Đã tạo `scripts/prepare_moe_text_cache.py`, cập nhật `check_text_cache()` gọi tiến trình mới với cache/environment tường minh. Script chuẩn bị và kiểm cả ba ID theo đúng nguồn; E2 chỉ cần config/tokenizer vì trọng số được phục hồi từ MGA. Kiểm E2 dùng đúng `BertConfig` và `BertTokenizer` với lookup mặc định, không đổi kiến trúc hoặc tên model trong source expert. Chuẩn bị online diễn ra riêng, adapter tiếp tục offline.

Thêm bốn test trong `tests/test_kaggle_text_cache.py`: tiến trình/environment riêng, chuẩn bị đúng alias E2 mà không tải trọng số BERT dư, từ chối cache thiếu alias E2 dù ID namespaced có sẵn, và validation offline không bật mạng. Test tiến trình đã được chạy trước sửa và thất bại đúng do helper cũ không tạo tiến trình mới; sau sửa đạt.

Nhóm Kaggle sau sửa: **16/16 PASS**, `5.012s`, log `/tmp/moescore_text_cache_tests.log`. Notebook hợp lệ theo nbformat và compile mọi code cell; compile Python mới/sửa và `git diff --check` PASS trước cập nhật tài liệu này. Chưa chạy lại toàn regression suite trong phần sửa cache.

Đã chạy CLI thật ở chế độ offline trên cache có sẵn `/home/vitquay1708/.cache/huggingface/hub`: cả `roberta-base`, `bert-base-uncased`, `google-bert/bert-base-uncased` đều PASS. E2 xác nhận tokenizer class `BertTokenizer`, model type `bert`. Không tải dữ liệu, không dựng encoder và không giải tuần tự checkpoint expert. Báo cáo `/tmp/moescore_text_cache_offline.json`; log `/tmp/moescore_text_cache_offline.log`. Đây là xác minh cache cục bộ, chưa phải lần chạy lại Expert 2 trong phiên Kaggle của người dùng.
