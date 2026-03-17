# 🚗 Dự Đoán Giá Xe Ô Tô Cũ - Machine Learning

## 📋 Tóm Tắt

Dự án xây dựng mô hình Machine Learning dự đoán giá xe ô tô cũ bằng cách so sánh **3 thuật toán**: Linear Regression, Random Forest, XGBoost.

---

## 📊 Dữ Liệu

- **Tổng mẫu:** 14,000 xe (sau lọc: 12,164 xe)
- **Features:** 6 (Hãng xe, Dòng xe, Đời xe, Phiên bản, Màu xe, Công tơ mét)
- **Target:** Giá xe (VNĐ)
- **Chia tập:** 80% Train / 20% Test

---

## 🔧 Xử Lý Dữ Liệu

### 1. Lọc dữ liệu
- Bỏ xe "Mới", giữ lại "Tốt", "Rất tốt", "Trung bình"
- Kết quả: 12,164 mẫu xe

### 2. Xử lý dữ liệu trống
- Cột text: Dùng **mode** (giá trị xuất hiện nhiều nhất)
- Cột số: Dùng **median** (giá trị giữa)

### 3. Feature Engineering
- Tạo feature mới: **Tuổi xe = 2026 - Đời xe**

### 4. Mã hóa dữ liệu
- **LabelEncoder** cho cột text (Hãng xe, Dòng xe, Phiên bản, Màu xe)
- **Ordinal Encoding** cho "Tình trạng" (1=Trung bình, 2=Tốt, 3=Rất tốt)

---

## 🤖 3 Thuật Toán

### 1. Linear Regression
```python
LinearRegression()
```
- Mô hình hồi quy tuyến tính cơ bản
- Giả định mối quan hệ tuyến tính giữa features và target
- Ưu điểm: Đơn giản, nhanh, dễ diễn giải

### 2. Random Forest Regressor
```python
RandomForestRegressor(n_estimators=200, max_depth=15, 
                      min_samples_split=5, min_samples_leaf=2)
```
- Xây dựng 200 cây quyết định độc lập
- Lấy trung bình kết quả của tất cả cây
- Ưu điểm: Ít bị overfitting, xác định được feature quan trọng

### 3. XGBoost Regressor
```python
XGBRegressor(n_estimators=200, learning_rate=0.1, 
             max_depth=10, min_child_weight=3)
```
- Cải tiến Gradient Boosting với tối ưu hóa tốt hơn
- Xây dựng cây tuần tự, mỗi cây sửa lỗi của cây trước
- Ưu điểm: Tốc độ nhanh, độ chính xác cao

---

## 📈 Kết Quả

| Thuật Toán | R² Score | MAE | MAPE |
|---|---|---|---|
| Linear Regression | 0.0639 | 504M VNĐ | 72.01% |
| **XGBoost** ⭐ | **0.9124** | **146M VNĐ** | **17.72%** |
| Random Forest | 0.9014 | 150M VNĐ | 17.72% |

**Model tốt nhất: XGBoost**
- R² = 0.9124: Giải thích 91.24% phương sai giá xe
- MAE = 146M VNĐ: Sai số trung bình ~146 triệu đồng
- MAPE = 17.72%: Sai số phần trăm hợp lý

---

## 📁 Cấu Trúc Dự Án

```
├── train_model.py              # Script huấn luyện 3 mô hình
├── app.py                      # Ứng dụng web Streamlit
├── visualize_data.py           # Trực quan hóa dữ liệu
├── visualize_models.py         # Trực quan hóa kết quả mô hình
├── visualize_algorithms.py     # Trực quan hóa 3 thuật toán
├── xe_cu.csv                   # Dữ liệu gốc
├── model.pkl                   # Mô hình XGBoost
├── encoders.pkl                # Encoders cho mã hóa dữ liệu
├── feature_columns.pkl         # Danh sách các features
├── training_info.pkl           # Thông tin huấn luyện
├── data_visualization.png      # Biểu đồ phân tích dữ liệu
├── models_comparison.png       # Biểu đồ so sánh 3 mô hình
├── models_table.png            # Bảng so sánh hiệu suất
├── THUYẾT_TRÌNH.md            # File thuyết trình chi tiết
├── requirements.txt            # Dependencies
└── README.md                   # File này
```

---

## 🚀 Chạy Dự Án

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình
```bash
python train_model.py
```

### 3. Chạy ứng dụng web
```bash
streamlit run app.py
```

### 4. Trực quan hóa dữ liệu
```bash
python visualize_data.py
python visualize_models.py
python visualize_algorithms.py
```

---

## 📊 Biểu Đồ

### Phân Tích Dữ Liệu
- Top 10 hãng xe
- Giá trung bình theo đời xe
- Giá vs Công tơ mét
- Phân bố theo tình trạng

### So Sánh Mô Hình
- R² Score của 3 thuật toán
- MAE của 3 thuật toán
- MAPE của 3 thuật toán

---

## 🎯 Kết Luận

✓ **XGBoost** là mô hình tốt nhất cho bài toán này
- Độ chính xác cao (R² = 0.9124)
- Sai số thấp (MAE = 146M VNĐ)
- Phù hợp để triển khai trong thực tế

---

## 📝 Ghi Chú

- Dữ liệu được lọc bỏ xe "Mới" để tập trung vào xe cũ
- Sử dụng Random State = 42 để đảm bảo kết quả có thể tái tạo
- Mô hình được lưu dưới dạng pickle (.pkl) để sử dụng lại

---

## 👥 Tác Giả

Dự án Machine Learning - Dự đoán Giá Xe Ô Tô Cũ

---

## 📄 License

MIT License
