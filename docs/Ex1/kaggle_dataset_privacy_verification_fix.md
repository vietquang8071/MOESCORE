# Sửa xác minh quyền riêng tư khi upload cache lên Kaggle

## Tóm tắt

Lỗi xảy ra vì `check_private_upload_target()` gọi `KaggleApi.dataset_view()`, nhưng method này không có trong Kaggle API đang được tham chiếu. Thay đổi này dùng `dataset_metadata(dataset, path)` để đọc metadata từ API, chỉ cho phép tạo dataset mới khi nhận diện được HTTP 404, và chỉ cho phép cập nhật dataset hiện có khi metadata xác nhận `isPrivate` là `true`.

Không thực hiện upload thật trong quá trình kiểm tra.

## Thay đổi

- `scripts/kaggle_full9_artifacts.py`
  - Kiểm tra handle theo dạng `owner/dataset-slug`; từ chối username mẫu như `YOUR_KAGGLE_USERNAME` trước khi đọc file hoặc gọi API.
  - Giữ dependency injection qua `check_private_upload_target(handle, *, api=None)` và bổ sung `api=None` tùy chọn cho `upload_feature_cache(...)` để kiểm thử mà không cần credentials.
  - Thay `dataset_view()` bằng `dataset_metadata(dataset, path)`. Metadata được đọc từ file do Kaggle API ghi ra; chấp nhận `isPrivate` hoặc `is_private`. Thiếu trường visibility, API không tương thích, lỗi xác thực, timeout và lỗi server đều dừng upload.
  - Chỉ coi lỗi có HTTP status **chính xác 404** là dataset chưa tồn tại. Nội dung lỗi chỉ nói “404” nhưng không có status HTTP không đủ để cho phép tạo mới.
  - Trước upload, kiểm tra manifest, tên archive, kích thước và SHA-256 của `feature_cache.tar`.
  - Sau upload, tải `upload_manifest.json` và archive bằng `KaggleApi.dataset_download_file()` trực tiếp, tránh resolver/cache của Kagglehub trong notebook. Chỉ báo PASS khi manifest khớp hoàn toàn và SHA-256 archive từ xa khớp.
- `tests/test_kaggle_full9_artifacts_privacy.py`
  - Thêm 18 kiểm thử dùng API giả, không cần Internet hay Kaggle credentials.
- `tests/test_kaggle_full9_dual_gpu.py`
  - Cập nhật fake API của các bài test đóng gói/upload sang `dataset_metadata()` và `dataset_download_file()`.
- `docs/Ex1/kaggle_dataset_privacy_verification_fix.md`
  - Báo cáo thay đổi và các giới hạn xác minh.

Không thay đổi extraction, adapters, gating, feature store, cache identity, training hoặc evaluation.

## API và tương thích

Source của Kaggle CLI hiện có `KaggleApi.dataset_metadata(dataset, path)`: method gọi API lấy metadata, ghi `response.info` vào file và trả về đường dẫn file. Cùng source cung cấp `dataset_download_file(dataset, file_name, path, force, quiet, ..., unzip)`, được dùng để xác minh bytes đã upload. Không thấy method `dataset_view()` trong source này. Kaggle API metadata định nghĩa trường `is_private` dạng boolean; code xử lý cả tên JSON camelCase và snake_case. [Kaggle CLI source: metadata](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py), [Kaggle API dataset metadata schema](https://www.kaggle.com/docs/mcp)

Source Kagglehub 0.3.13 xác nhận `dataset_upload()` nhận `handle`, thư mục local và ghi chú phiên bản; hàm không có tham số visibility. Với dataset mới, Kagglehub tạo dataset mới; tài liệu Kaggle CLI xác nhận tạo dataset mặc định là private nếu không truyền tùy chọn `--public`. Dataset đã tồn tại được xử lý như tạo version, còn hàm của MOESCORE chặn dataset public trước khi gọi uploader. [Kagglehub 0.3.13 upload API](https://github.com/Kaggle/kagglehub/blob/v0.3.13/src/kagglehub/datasets.py), [Kaggle dataset creation defaults](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)

Trong source Kagglehub 0.3.13, `dataset_download()` đi qua resolver registry. Vì Kaggle Notebook đăng ký resolver riêng cho input/cache, code xác minh sau upload dùng API tải file trực tiếp thay vì `kagglehub.dataset_download()`. Điều này giữ việc xác minh trên một bản tải mới vào thư mục tạm, không dựa vào file được mount hoặc cache cũ. [Kagglehub dataset download resolver](https://github.com/Kaggle/kagglehub/blob/v0.3.13/src/kagglehub/datasets.py)

## Dependencies và giới hạn

Các requirements hiện khai báo `kaggle>=1.7,<2` và `kagglehub>=0.3,<1`; `kagglesdk` không được khai báo trực tiếp. Trong Python environment dùng để chạy test, cả `kaggle`, `kagglehub` và `kagglesdk` đều chưa cài. Vì vậy, method signatures/response format được đối chiếu với source chính thức, còn test hành vi chạy bằng fake API. Chưa xác nhận runtime của một Kaggle Notebook cụ thể; nếu package ở runtime không có `dataset_metadata()` hoặc `dataset_download_file()` thì code fail closed với lỗi tương thích rõ ràng.

## Kiểm thử

Kiểm thử bao phủ: HTTP 404 hợp lệ; private/public; metadata visibility thiếu; handle sai và username mẫu; lỗi xác thực; timeout/server error; lỗi API; chặn upload khi public; xác minh SHA-256 local/remote; manifest từ xa không khớp; archive từ xa không khớp; archive local bị thay đổi.

Không có Kaggle credentials trong môi trường local; không chạy upload thật và không tuyên bố upload đã thành công.
