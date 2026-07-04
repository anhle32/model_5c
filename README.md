# 🏦 Ứng dụng Dự báo Rủi ro Tín dụng theo Khung 5C

Ứng dụng web Streamlit được chuyển đổi từ notebook `mohinh.ipynb`, huấn luyện mô hình **Logistic Regression** (scikit-learn) để dự báo xác suất vỡ nợ **PD** (0 = không rủi ro, 1 = có rủi ro) của khách hàng dựa trên **24 tiêu chí thuộc khung 5C**, thang đo Likert 1–5:

| Nhóm | Ý nghĩa | Biến |
|---|---|---|
| TC | Tư cách (Character) | TC1–TC5 |
| NL | Năng lực (Capacity) | NL1–NL4 |
| DK | Điều kiện (Conditions) | DK1–DK5 |
| V | Vốn (Capital) | V1–V6 |
| TS | Tài sản đảm bảo (Collateral) | TS1–TS4 |

Pipeline tái hiện đúng notebook: đọc CSV → chọn X (24 biến) và y (PD) → `train_test_split(test_size=0.2, random_state=32)` → `LogisticRegression()` → kiểm định trên tập test → dự báo khách hàng mới kèm xác suất.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

Trình duyệt sẽ mở tại `http://localhost:8501`.

## Cấu trúc file dữ liệu đầu vào

- **File huấn luyện** (tải ở sidebar): CSV có cấu trúc như `5c.csv`, bắt buộc chứa 24 cột biến đầu vào (`TC1`–`TC5`, `NL1`–`NL4`, `DK1`–`DK5`, `V1`–`V6`, `TS1`–`TS4`) và cột mục tiêu `PD` (0/1). Các cột khác (ví dụ `Dấu thời gian`, `NN`) được bỏ qua khi huấn luyện.
- **File dự báo hàng loạt** (tab Sử dụng mô hình): CSV chứa **đúng 24 cột biến đầu vào** như trên; nếu thiếu cột, app sẽ báo lỗi và liệt kê cột thiếu.

## Mô tả các tab

1. **📋 Tổng quan dữ liệu** — số dòng/cột/dung lượng file, xem dữ liệu thô, thống kê mô tả 24 biến X và biến PD.
2. **📊 Trực quan hóa dữ liệu** — lưới biểu đồ 2×2 (mặc định: PD + đại diện các nhóm 5C); biểu đồ tự chọn loại theo kiểu biến (bar cho biến rời rạc Likert, bar phân phối lớp cho PD); có multiselect để đổi biến.
3. **🎯 Kết quả huấn luyện & Kiểm định** — Accuracy, Precision, Recall, F1, ROC-AUC; ma trận nhầm lẫn; đường cong ROC; classification report; bảng kết quả chấm điểm tập test. Chỉ hiển thị sau khi bấm nút huấn luyện.
4. **🔮 Sử dụng mô hình** — hai chế độ: (a) nhập trực tiếp 24 tiêu chí của một khách hàng (form theo nhóm 5C, mặc định = trung vị) → kết quả PD kèm xác suất; (b) tải file CSV dự báo hàng loạt → bảng kết quả + nút tải CSV (utf-8-sig).

## Cách vận hành

- Huấn luyện chỉ được kích hoạt **một lần** khi bấm nút **🚀 Huấn luyện mô hình** ở sidebar; mô hình, bộ tiền xử lý và bảng kết quả được lưu trong `st.session_state` nên chuyển tab không train lại.
- Siêu tham số mặc định trên giao diện lấy đúng từ notebook: `C=1.0`, `max_iter=100`, `solver='lbfgs'`, `test_size=0.2`, `random_state=32`.

## Ghi chú

- Notebook gốc **không sử dụng scaler/encoder**, nên app cũng không áp dụng tiền xử lý chuẩn hóa — dữ liệu Likert 1–5 được đưa thẳng vào mô hình (nhất quán với notebook). Trường `preprocessor` trong `session_state` được đặt `None` để nhất quán kiến trúc.
- Notebook không cố định siêu tham số nào của `LogisticRegression()` (dùng mặc định sklearn), nên các widget tham số trên sidebar lấy mặc định của sklearn làm giá trị khởi tạo.
- Với `max_iter=100` (mặc định notebook), solver `lbfgs` có thể cảnh báo chưa hội tụ trên một số dữ liệu — có thể tăng `max_iter` trên sidebar.
- Khuyến nghị dùng **Streamlit ≥ 1.38** (app dùng `st.container(height=...)`, `st.divider`, `st.metric`, tabs). Các tính năng mới hơn như `st.container(horizontal=True)` hay dynamic container yêu cầu Streamlit ≥ 1.55 nhưng không bắt buộc cho app này.
