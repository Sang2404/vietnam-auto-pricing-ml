# 🚗 Thuyết Trình Dự Án: Dự Đoán Giá Xe Ô Tô Cũ

---

## 1. MỤC TIÊU BÀI TOÁN

**Bài toán:** Dự đoán giá xe ô tô cũ dựa trên các đặc trưng của xe

**Dữ liệu:**
- 12,164 mẫu xe (sau lọc)
- 6 đặc trưng: Hãng xe, Dòng xe, Đời xe, Phiên bản, Màu xe, Công tơ mét
- Target: Giá xe (VNĐ)
- Chia tập: 80% Train / 20% Test

---

## 2. XỬ LÝ DỮ LIỆU

### 2.1 Lọc dữ liệu
- **Vấn đề:** Dữ liệu gốc có 14,000 mẫu, bao gồm xe "Mới" (giá cao, không phù hợp)
- **Giải pháp:** Lọc bỏ xe "Mới", giữ lại "Tốt", "Rất tốt", "Trung bình"
- **Kết quả:** Còn 12,164 mẫu xe phù hợp

### 2.2 Xử lý dữ liệu trống (Missing Values)
- **Vấn đề:** Một số cột có giá trị trống
- **Giải pháp:**
  - Cột text (Hãng xe, Dòng xe, v.v.): Dùng **mode** (giá trị xuất hiện nhiều nhất)
  - Cột số (Công tơ mét, Giá): Dùng **median** (giá trị giữa)
- **Lợi ích:** Giữ được toàn bộ dữ liệu, không mất mẫu

### 2.3 Feature Engineering (Tạo đặc trưng mới)
- **Tạo feature:** Tuổi xe = 2026 - Đời xe
- **Ý nghĩa:** Xe càng cũ (tuổi cao) thì giá càng rẻ
- **Lợi ích:** Giúp mô hình học được mối quan hệ giữa tuổi xe và giá

### 2.4 Mã hóa dữ liệu (Encoding)
- **Vấn đề:** Mô hình học máy chỉ hiểu số, không hiểu text
- **Giải pháp:**
  - **LabelEncoder** cho cột text (Hãng xe, Dòng xe, Phiên bản, Màu xe):
    - Chuyển mỗi giá trị text thành số (0, 1, 2, ...)
    - Ví dụ: Toyota=0, Honda=1, Hyundai=2, ...
  - **Ordinal Encoding** cho "Tình trạng" (có thứ tự):
    - Trung bình = 1, Tốt = 2, Rất tốt = 3
    - Số càng lớn = xe càng tốt = giá càng cao

### 2.5 Chia tập dữ liệu
- **Train set (80%):** 9,731 mẫu - dùng để huấn luyện mô hình
- **Test set (20%):** 2,433 mẫu - dùng để đánh giá mô hình
- **Lợi ích:** Đánh giá mô hình trên dữ liệu chưa từng thấy

---

## 3. CƠ SỞ LÝ THUYẾT VÀ ỨNG DỤNG TRONG DỰ ÁN

### 3.1 Linear Regression

**Nguyên lý:**
- Tìm đường thẳng (hoặc siêu phẳng) phù hợp nhất với dữ liệu
- Công thức: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
- Sử dụng phương pháp bình phương nhỏ nhất (Least Squares)

**Áp dụng trong dự án:**
- Giả định giá xe có mối quan hệ tuyến tính với các đặc trưng
- Ví dụ: Giá = a + b×Tuổi_xe + c×Công_tơ_mét + ...
- Tìm các hệ số a, b, c, ... sao cho sai số nhỏ nhất

**Ưu điểm:**
- Đơn giản, dễ hiểu
- Nhanh, không cần điều chỉnh tham số
- Dễ diễn giải kết quả

**Nhược điểm:**
- Giả định mối quan hệ tuyến tính (không phù hợp với dữ liệu phức tạp)
- Độ chính xác thấp hơn các mô hình phức tạp

---

### 3.2 Random Forest Regressor

**Nguyên lý:**
- Xây dựng 200 cây quyết định độc lập
- Mỗi cây được huấn luyện trên tập con ngẫu nhiên của dữ liệu
- Kết quả cuối cùng = trung bình của 200 cây

**Áp dụng trong dự án:**
- Mỗi cây học cách dự đoán giá từ các đặc trưng khác nhau
- Ví dụ: Cây 1 học từ (Hãng xe, Đời xe), Cây 2 học từ (Công tơ mét, Tình trạng)
- Lấy trung bình 200 cây → Dự đoán chính xác hơn

**Ưu điểm:**
- Xử lý tốt dữ liệu phi tuyến tính (giá xe không tăng tuyến tính)
- Ít bị overfitting (mô hình không học thuộc lòng dữ liệu)
- Xác định được feature nào quan trọng nhất

**Tham số:**
- n_estimators=200: Số cây (càng nhiều càng tốt, nhưng chậm hơn)
- max_depth=15: Độ sâu tối đa của cây (tránh overfitting)

---

### 3.3 XGBoost Regressor

**Nguyên lý:**
- Cải tiến Gradient Boosting với tối ưu hóa tốt hơn
- Xây dựng cây tuần tự (không độc lập)
- Cây thứ 2 sửa lỗi của cây thứ 1, cây thứ 3 sửa lỗi của cây 1+2, ...
- Có regularization tích hợp (tránh overfitting)

**Áp dụng trong dự án:**
- Cây 1: Dự đoán giá ban đầu (sai số lớn)
- Cây 2: Học cách sửa sai của cây 1
- Cây 3: Học cách sửa sai của cây 1+2
- Kết quả: Dự đoán ngày càng chính xác

**Ưu điểm:**
- Tốc độ huấn luyện nhanh
- Độ chính xác cao
- Xử lý dữ liệu lớn hiệu quả

**Tham số:**
- n_estimators=200: Số cây
- learning_rate=0.1: Tốc độ học (nhỏ = chậm nhưng chính xác hơn)
- max_depth=10: Độ sâu tối đa

---

## 4. PHƯƠNG PHÁP ĐÁNH GIÁ MÔ HÌNH

### Tiêu chí đánh giá:

**R² Score:** 
- Tỷ lệ phương sai được giải thích (0-1)
- R² = 0.9763 → Mô hình giải thích 97.63% phương sai giá xe
- Càng gần 1 càng tốt

**MAE (Mean Absolute Error):**
- Sai số tuyệt đối trung bình (đơn vị: VNĐ)
- MAE = 53.1M → Sai số trung bình 53 triệu đồng
- Càng thấp càng tốt

**MAPE (Mean Absolute Percentage Error):**
- Sai số phần trăm trung bình (%)
- MAPE = 10.22% → Sai số trung bình 10.22%
- Càng thấp càng tốt

### Kết quả so sánh:

| Thuật Toán | R² Score | MAE | MAPE |
|---|---|---|---|
| Linear Regression | 0.8234 | 156.2M VNĐ | 28.45% |
| **Random Forest** ⭐ | **0.9763** | **53.1M VNĐ** | **10.22%** |
| XGBoost | 0.9080 | 104.1M VNĐ | 20.15% |

**Kết luận:** Random Forest là mô hình tốt nhất

---

## 5. TRỰC QUAN DỮ LIỆU

- **Top 10 hãng xe:** Toyota, Honda, Hyundai, Kia, ...
- **Giá theo đời xe:** Xe mới (2024) đắt hơn xe cũ (2010)
- **Giá vs Công tơ mét:** Xe chạy nhiều (công tơ mét cao) thì rẻ hơn
- **Phân bố tình trạng:** Tốt > Rất tốt > Trung bình

---

## 6. XÂY DỰNG MÔ HÌNH

### Bước 1: Tiền xử lý dữ liệu
- Lọc bỏ xe "Mới"
- Xử lý dữ liệu trống (mode, median)

### Bước 2: Feature Engineering
- Tạo feature: Tuổi xe = 2026 - Đời xe

### Bước 3: Mã hóa dữ liệu
- LabelEncoder cho cột text
- Ordinal Encoding cho "Tình trạng"

### Bước 4: Chia tập dữ liệu
- Train: 80%, Test: 20%

### Bước 5: Huấn luyện 3 mô hình
- Huấn luyện trên tập train
- Đánh giá trên tập test

---

## 7. ỨNG DỤNG DỰ ĐOÁN

**Kết quả Random Forest:**
- R² = 0.9763: Giải thích 97.63% phương sai
- MAE = 53.1M VNĐ: Sai số trung bình
- MAPE = 10.22%: Sai số phần trăm

**Ứng dụng web:**
- Người dùng nhập thông tin xe (hãng, dòng, đời, v.v.)
- Dự đoán giá tự động
- Hiển thị kết quả

---

## 8. HIỂN THỊ KẾT QUẢ

**Bảng so sánh 3 mô hình:**
- R² Score, MAE, MAPE

**Biểu đồ:**
- So sánh R², MAE, MAPE
- Scatter plot: Giá thực tế vs dự đoán
- Feature Importance

**Kết luận:**
- Random Forest là mô hình tốt nhất
- Độ chính xác cao, sai số thấp
- Phù hợp triển khai thực tế
