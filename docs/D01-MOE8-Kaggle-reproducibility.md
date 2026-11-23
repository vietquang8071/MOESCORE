# D01 MOE8 trên Kaggle — ghi chú tái lập

## Mã nguồn và kernel

- Notebook: `notebooks/XACLE2027_D01_MOE8_Kaggle.ipynb`
- Gói push: `kaggle/d01_moe8/`
- Kernel riêng tư: `qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics`
- Mã nguồn MOESCORE được checkout detached tại `2e6c247563b554116017c5f8630cda7ae257bc59`.
- Dataset IDs được Kaggle CLI xác minh: `qnv8071/moe8-bestgate`, `qnv8071/recovery`, `qnv8071/xacle-weights`, `qnv8071/xacle-2027-data`, `qnv8071/moescore-full9-feature-cache`.

Kernel chỉ gắn năm dataset trên. Dataset đánh giá riêng tư không được truy vấn, gắn, tải, mở, hoặc tham chiếu trong code. Kernel bật internet để checkout commit đã ghim và cài các dependency còn thiếu; T4 chỉ cần khi không có cache Expert8 phù hợp. Nếu có store Full8 tương thích hoặc cache Expert8 dùng được, inference gate và tính toán D01 chạy trên CPU.

## Quy trình và hợp đồng

Checkpoint public thực tế là format v7 (không phải format v5 của source commit ghim), nhưng giữ `full_8`, đúng thứ tự `expert1`…`expert8`, 8 điểm, 8 trọng số gate và feature width 33,032. Adapter D01 trong `scripts/diagnostics/d01_moe8.py` xác minh source hash liên quan trực tiếp đến gate/score/store, kiến trúc, fingerprints và provenance; sau đó dùng validator format v5 đã ghim để kiểm tra compatibility view và nạp state dict strict. Các source khác biệt được ghi trong `gate_source_code_audit`; store train/validation và preprocessing phải khớp identity của checkpoint.

Format v7 không lưu riêng overlap diagnostic/policy. Notebook tái lập diagnostic từ các sample ID và WAV SHA-256 nằm trong checkpoint, bật override repository chỉ khi hash overlap tồn tại, rồi đối chiếu diagnostic với manifests trước khi dựng store. Nguồn của chính sách này được ghi ở `train_validation_overlap_policy_origin`.

Nếu không tìm thấy Full8 tương thích, notebook tìm Full7/Full9 và Whisper/E5/E6 cache đã gắn, xác minh hàng theo sample ID, rồi gọi builder hiện có. Luồng D01 bật riêng `--allow-public-split-audio-overlap` cho Full7/Full8 assembly để giữ nguyên cả 13.500 hàng public khi WAV SHA-256 trùng giữa train, validation hoặc test; builder mặc định vẫn strict, và cờ này không được truyền vào huấn luyện/checkpoint fitting. `reports/audio_overlap.json` ghi tất cả SHA-256, sample IDs, số nhóm theo cặp split, chính sách override và cảnh báo bias. Notebook xác minh ID/count/thứ tự sau mỗi store; overlap chỉ bị loại khỏi phép gộp metric sensitivity tùy chọn, không loại khỏi feature store hay predictions. Nếu cache Expert8 hợp lệ không khớp provenance gate thì notebook mới gọi extractor Expert8 có resume trên Kaggle T4; không chạy lại expert khác.

Mỗi split có `source_manifest_sha256` riêng. D01 kiểm tra digest canonical với nội dung `source_manifest.json` tương ứng, rồi đối chiếu sample ID, filename và WAV/caption fingerprints theo đúng thứ tự của feature store; không yêu cầu ba split có cùng digest. Checkpoint identities và preprocessing contract vẫn phải khớp giữa split. Notebook Full8 dùng chung validator đã bỏ equality sai này và giữ kiểm tra provenance chéo split.

Điểm expert không được hiệu chỉnh, clamp hay chuẩn hóa lại. E1/E2 là similarity proxy, E3–E8 là regression score; raw-unit `5*s+5` chỉ là phép diễn giải. Tính toán envelope thực hiện trên normalized scores. Raw MSE bằng 25 lần normalized MSE. Oracle projection và MSE lower bound là chẩn đoán sample-wise, không phải hiệu năng gate có thể đạt.

Train được đánh dấu mô tả in-sample, validation mang lịch sử lựa chọn validation, và D01 không phải LOGO. Test labels—nếu metadata public phát hành—chỉ được đọc sau khi inference test không nhãn đã hoàn tất và chỉ cho báo cáo mô tả; nếu thiếu thì không có oracle/error/SRCC/MSE test. Không có huấn luyện hoặc optimizer trong notebook.

## Artifact khi kernel chạy xong

Kernel ghi vào `/kaggle/working/d01_moe8/`:

- `reports/audio_overlap.json`, `full7_build.json`, `full7_build.log`, `overlap_metric_sensitivity.json`
- `reports/summary.json`, `global_metrics.csv`, `per_bin_metrics.csv`, `generator_metrics.csv`, `structural_vs_gating.json`, `provenance.json`, `D01-MOE8-report.md`
- `predictions/{train,validation,test}_d01.csv`
- `plots/*.png` — các biểu đồ riêng theo split và biểu đồ generator khi có metadata.

Không có số liệu thực nghiệm nào được điền sẵn trong source; các kết luận được dựng từ artifact runtime.

## Kiểm thử trước push

Synthetic D01/overlap tests chạy bằng `python3 -m unittest tests.test_d01_moe8 tests.test_public_audio_overlap -v`. Các integration tests hiện có cần pytest và PyTorch; nếu thiếu hai dependency này ở local, ghi rõ và không thay bằng inference đắt tiền. Notebook kiểm tra gate provenance/shape và inference equivalence khi Kaggle chạy.

## Lệnh Kaggle CLI

```bash
kaggle kernels status qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics
kaggle kernels logs qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics
kaggle kernels pull qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics -p d01_moe8_kernel
kaggle kernels output qnv8071/xacle-2027-d01-moe8-convex-envelope-diagnostics -p d01_moe8_output
```

`kernels push` xác nhận upload và khởi chạy job, không chứng minh run đã hoàn tất. Chỉ dùng `kernels output` sau khi `kernels status` báo run hoàn tất.
