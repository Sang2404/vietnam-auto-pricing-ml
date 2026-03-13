# 🚗 Dự Đoán Giá Xe Ô Tô Cũ - Machine Learning Project

## � Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Vấn Đề & Mục Tiêu](#vấn-đề--mục-tiêu)
3. [Dữ Liệu](#dữ-liệu)
4. [Phương Pháp](#phương-pháp)
5. [Kết Quả](#kết-quả)
6. [Cài Đặt & Chạy](#cài-đặt--chạy)
7. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)

---

## 🎯 Giới Thiệu

Dự án này xây dựng một ứng dụng web dự đoán giá xe ô tô cũ sử dụng **Machine Learning**.

**Mục đích:** Giúp người dùng ước tính giá trị hiện tại của xe ô tô cũ dựa trên các đặc trưng như:
- Hãng xe
- Dòng xe
- Năm sản xuất
- Phiên bản
- Màu sắc
- Số km đã chạy

---

## 🔍 Vấn Đề & Mục Tiêu

### Vấn Đề
- Người mua/bán xe cũ khó xác định giá trị thực của xe
- Không có công cụ tiêu chuẩn để dự đoán giá xe cũ
- Giá xe phụ thuộc vào nhiều yếu tố phức tạp

### Mục Tiêu
- ✅ Xây dựng mô hình dự đoán giá xe chính xác
- ✅ So sánh 3 thuật toán Machine Learning
- ✅ Tạo giao diện web thân thiện cho người dùng
- ✅ Đạt độ chính xác cao (R² > 0.99)

---

## 📊 Dữ Liệu

### Nguồn Dữ Liệu
- File: `xe_cu.csv`
- Tổng số mẫu: **9,999 xe**

### Các Đặc Trưng (Features)
| Đặc Trưng | Kiểu Dữ Liệu | Mô Tả |
|---|---|---|
| Hãng xe | Categorical | 11 hãng xe khác nhau |
| Dòng xe | Categorical | 78 dòng xe |
| Đời xe | Numerical | Năm sản xuất |
| Phiên bản | Categorical | 163 phiên bản |
| Màu xe | Categorical | 8 màu sắc |
| Công tơ mét (km) | Numerical | Số km đã chạy |
| **Giá (VNĐ)** | **Numerical** | **Target - Giá xe** |

### Chia Tập Dữ Liệu
- **Train:** 7,999 mẫu (80%)
- **Test:** 2,000 mẫu (20%)

---

## 🤖 Phương Pháp

### Quy Trình Xử Lý Dữ Liệu
1. **Đọc dữ liệu** từ CSV
2. **Xử lý giá trị trống** (fillna, dropna)
3. **Mã hóa dữ liệu** (Label Encoding cho dữ liệu categorical)
4. **Chia tập dữ liệu** (80/20 split)

### 3 Thuật Toán So Sánh

#### 1️⃣ Random Forest Regressor
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)
```

#### 2️⃣ Gradient Boosting Regressor
```python
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
```

#### 3️⃣ XGBoost Regressor
```python
XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
```

### Chỉ Số Đánh Giá

**R² Score (Coefficient of Determination)**
- Công thức: R² = 1 - (SS_res / SS_tot)
- Giá trị: 0 đến 1
- Ý nghĩa: Tỷ lệ phương sai được giải thích

**MAE (Mean Absolute Error)**
- Công thức: MAE = (1/n) × Σ|y_i - ŷ_i|
- Đơn vị: VNĐ
- Ý nghĩa: Sai số trung bình tuyệt đối

**MAPE (Mean Absolute Percentage Error)**
- Công thức: MAPE = (1/n) × Σ|((y_i - ŷ_i) / y_i) × 100|
- Đơn vị: %
- Ý nghĩa: Sai số phần trăm trung bình

---

## 📈 Kết Quả

### So Sánh 3 Thuật Toán

| Thuật Toán | R² (Test) | MAE (Test) | MAPE | Xếp Hạng |
|---|---|---|---|---|
| **Random Forest** | **0.9935** | **26,672,702 VNĐ** | **4.44%** | 🥇 |
| XGBoost | 0.9818 | 47,023,370 VNĐ | 8.29% | 🥈 |
| Gradient Boosting | 0.9812 | 47,430,914 VNĐ | 8.40% | 🥉 |

### 🏆 Model Tốt Nhất: Random Forest

**Hiệu Suất:**
- R² Score: **0.9935** (99.35% chính xác)
- MAE: **26.7 triệu đồng**
- MAPE: **4.44%**

**Ý Nghĩa:**
- Giải thích 99.35% phương sai của giá xe
- Sai số trung bình chỉ 26.7 triệu đồng
- Độ chính xác 95.56%

### Biểu Đồ So Sánh

```
R² Score (Test)
Random Forest:      ████████████████████ 0.9935 ⭐
XGBoost:            ███████████████████ 0.9818
Gradient Boosting:  ███████████████████ 0.9812

MAE (Test) - Càng thấp càng tốt
Random Forest:      ██████ 26.7M VNĐ ⭐
XGBoost:            ███████████ 47.0M VNĐ
Gradient Boosting:  ███████████ 47.4M VNĐ
```

---

## 🚀 Cài Đặt & Chạy

### Yêu Cầu
- Python 3.8+
- pip

### Bước 1: Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn Luyện Mô Hình
```bash
python train_model.py
```

**Output:**
- `models.pkl` - 3 mô hình
- `best_model_name.pkl` - Tên model tốt nhất
- `models_performance.pkl` - Hiệu suất 3 model
- `encoders.pkl` - Bộ encoder
- `feature_columns.pkl` - Danh sách features
- `processed_data_raw.pkl` - Dữ liệu gốc
- `training_info.pkl` - Thông tin huấn luyện

### Bước 3: Chạy Ứng Dụng Web
```bash
streamlit run app.py
```

**Truy cập:** http://localhost:8501

---

## 📁 Cấu Trúc Dự Án

```
vietnam-auto-pricing-ml/
├── 📄 README.md                    # Tài liệu này
├── 📄 requirements.txt             # Dependencies
│
├── 🐍 Python Scripts
│   ├── train_model.py              # Huấn luyện 3 model
│   ├── app.py                      # Ứng dụng Streamlit
│   └── generate_xe_cu_data.py      # Tạo dữ liệu
│
├── 📊 Dữ Liệu
│   └── xe_cu.csv                   # Dữ liệu gốc (9,999 xe)
│
└── 💾 Mô Hình & Encoder
    ├── models.pkl                  # 3 mô hình
    ├── best_model_name.pkl         # Tên model tốt nhất
    ├── models_performance.pkl      # Hiệu suất 3 model
    ├── encoders.pkl                # Bộ encoder
    ├── feature_columns.pkl         # Danh sách features
    ├── processed_data_raw.pkl      # Dữ liệu gốc
    └── training_info.pkl           # Thông tin huấn luyện
```

---

## 💡 Tính Năng Chính

### 1. Giao Diện Web Premium
- ✅ Thiết kế hiện đại, thân thiện
- ✅ Responsive design
- ✅ Màu sắc xanh dương chuyên nghiệp

### 2. Dự Đoán Giá Xe
- ✅ Nhập 6 thông tin xe
- ✅ Dự đoán tức thì
- ✅ Hiển thị kết quả từ model tốt nhất

### 3. Phân Tích Mô Hình
- ✅ Tab "Phân tích mô hình": Hiển thị R² của 3 model
- ✅ Tab "Bảng chi tiết": So sánh đầy đủ

### 4. Empty State
- ✅ Hướng dẫn nhẹ nhàng khi chưa dự đoán

---

## � Ví Dụ Sử Dụng

### Dự Đoán Giá Toyota Camry 2020

**Nhập thông tin:**
- Hãng xe: Toyota
- Dòng xe: Camry
- Đời xe: 2020
- Phiên bản: 2.5L
- Màu xe: Trắng
- Công tơ mét: 50,000 km

**Kết quả:**
- Giá dự đoán: **1,200,000,000 VNĐ** (1.2 tỷ)
- Thuật toán: Random Forest
- Độ tin cậy: 99.35%

---

## 🎓 Kết Luận

### Điểm Mạnh
- ✅ So sánh 3 thuật toán khác nhau
- ✅ Chọn model tốt nhất tự động
- ✅ Hiệu suất rất cao (R² = 0.9935)
- ✅ Giao diện web đẹp và dễ sử dụng
- ✅ Tài liệu đầy đủ

### Cải Tiến Tương Lai
- 🔄 Thêm các model khác (LightGBM, CatBoost)
- 🔄 Thêm cross-validation
- 🔄 Hiển thị feature importance
- 🔄 Thêm prediction intervals
- 🔄 Lưu lịch sử dự đoán

---

## 👨‍� Tác Giả
Dự án Machine Learning - Năm 4, HK2

## 📞 Liên Hệ
Nếu có câu hỏi, vui lòng liên hệ hoặc tạo issue.

---

**Sẵn sàng sử dụng!** 🚀
