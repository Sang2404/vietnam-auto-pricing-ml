# 🚗 Ứng Dụng Dự Đoán Giá Xe Ô Tô Cũ - Báo Cáo Học Máy

## 📊 Tổng Quan Dự Án

Xây dựng mô hình Machine Learning dự đoán giá xe ô tô cũ tại Việt Nam (2016-2026) dựa trên các đặc trưng: hãng xe, dòng xe, năm sản xuất, phiên bản, màu sắc, và công tơ mét.

**Dataset:** 10.000 dòng dữ liệu từ 11 hãng xe phổ biến
**Mô hình:** Random Forest Regressor
**Kết quả:** R² Score = 0.9978, MAPE = 1.63%

---

## 🔄 QUY TRÌNH XỬ LÝ DỮ LIỆU

### **BƯỚC 1: SINH DỮ LIỆU (generate_xe_cu_data.py)**

#### 1.1 Xây Dựng Dictionary Car_DB
- **Dữ liệu:** 11 hãng xe × 50+ dòng xe × 2-3 phiên bản
- **Hãng xe:** Toyota, Honda, Mazda, Mitsubishi, Hyundai, Kia, Ford, VinFast, Mercedes-Benz, BMW, Lexus
- **Giá gốc:** Niêm yết từ thị trường Việt Nam (280 triệu - 2.85 tỷ VNĐ)

#### 1.2 Thuật Toán Tính Giá Xe Cũ
```
Giá xe cũ = Giá gốc × (1 - Hệ số khấu hao)^Tuổi xe - Bù/trừ ODO + Biến động ngẫu nhiên
```

**Hệ số khấu hao theo hãng xe (% mất giá/năm):**
- Xe Nhật (Toyota, Honda, Lexus): 6-7%/năm (giữ giá tốt nhất)
- Xe Nhật khác (Mazda, Mitsubishi): 7-8%/năm
- Xe Hàn/Mỹ/Châu Âu phổ thông (Hyundai, Kia, Ford): 8-10%/năm
- Xe Việt Nam (VinFast): 10-13%/năm
- Xe Sang Đức (Mercedes, BMW): 10-12%/năm

**Bù/trừ giá dựa trên ODO:**
- Mức tiêu chuẩn: 15.000 km/năm
- Công thức: Mỗi 1.000 km chênh lệch = ±0.5% giá trị xe
- Xe 2025-2026 (lướt): ODO < 20.000 km

**Biến động ngẫu nhiên:** ±3% để mô phỏng thị trường thực tế

#### 1.3 Đầu Ra
- File `xe_cu.csv`: 10.000 dòng × 7 cột
- Cột: Hãng xe, Dòng xe, Đời xe, Phiên bản, Màu xe, Công tơ mét (km), Giá (VNĐ)

---

### **BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (train_model.py)**

#### 2.1 Xử Lý Giá Trị Trống (Missing Values)
```
Cột "Màu xe": 476 giá trị trống (4.76%)
Phương pháp: Điền bằng Mode (giá trị xuất hiện nhiều nhất) = "Trắng"
Kết quả: 10.000 dòng → 10.000 dòng (không mất dữ liệu)
```

#### 2.2 Mã Hóa Dữ Liệu Dạng Chữ (Categorical Encoding)
**Phương pháp:** Label Encoding

| Cột | Số giá trị | Ví dụ |
|-----|-----------|-------|
| Hãng xe | 11 | Toyota→1, Honda→2, ... |
| Dòng xe | 50+ | Vios→1, Camry→2, ... |
| Phiên bản | 100+ | "Toyota Vios 1.5E MT"→1, ... |
| Màu xe | 8 | Trắng→1, Đen→2, ... |

**Lý do chọn Label Encoding:**
- Dữ liệu có thứ tự tự nhiên (hãng xe, dòng xe)
- Giảm chiều dữ liệu so với One-Hot Encoding
- Phù hợp với Random Forest (không nhạy cảm với thứ tự)

#### 2.3 Chuẩn Hóa Dữ Liệu (Normalization)
- **Không chuẩn hóa** vì Random Forest không nhạy cảm với tỷ lệ đặc trưng
- Giữ nguyên giá trị gốc để dễ diễn giải

#### 2.4 Chia Tập Dữ Liệu (Train-Test Split)
```
Tổng dữ liệu: 10.000 dòng
├─ Tập Train: 8.000 dòng (80%)
└─ Tập Test: 2.000 dòng (20%)

Phương pháp: train_test_split(test_size=0.2, random_state=42)
- random_state=42: Đảm bảo kết quả lặp lại
- Tỷ lệ 80-20: Chuẩn trong ML
```

#### 2.5 Đầu Ra Tiền Xử Lý
- Dữ liệu đã mã hóa: 10.000 × 6 features
- Features: [Hãng xe, Dòng xe, Đời xe, Phiên bản, Màu xe, Công tơ mét]
- Target: Giá (VNĐ)

---

### **BƯỚC 3: HUẤN LUYỆN MÔ HÌNH (train_model.py)**

#### 3.1 Lựa Chọn Thuật Toán: Random Forest Regressor

**Tại sao Random Forest?**
- Xử lý tốt dữ liệu hỗn hợp (số và phân loại)
- Không nhạy cảm với outliers
- Tự động tìm tương tác giữa features
- Cung cấp feature importance
- Hiệu suất cao trên dữ liệu lớn

#### 3.2 Cấu Hình Mô Hình
```python
RandomForestRegressor(
    n_estimators=100,      # Số cây trong rừng
    max_depth=20,          # Độ sâu tối đa mỗi cây
    min_samples_split=5,   # Số mẫu tối thiểu để chia node
    min_samples_leaf=2,    # Số mẫu tối thiểu ở leaf node
    random_state=42,       # Tái tạo kết quả
    n_jobs=-1              # Sử dụng tất cả CPU cores
)
```

**Giải thích tham số:**
- `n_estimators=100`: Cân bằng giữa độ chính xác và tốc độ
- `max_depth=20`: Tránh overfitting nhưng đủ sâu để học
- `min_samples_split=5`: Tránh cây quá chi tiết
- `min_samples_leaf=2`: Đảm bảo leaf nodes có ít nhất 2 mẫu

#### 3.3 Quá Trình Huấn Luyện
```
Input: X_train (8.000 × 6), y_train (8.000,)
↓
Xây dựng 100 cây quyết định (Decision Trees)
- Mỗi cây được huấn luyện trên random subset của dữ liệu
- Mỗi node tìm feature và threshold tốt nhất để chia dữ liệu
- Giảm MSE (Mean Squared Error) tại mỗi bước
↓
Output: Mô hình Random Forest đã huấn luyện
```

#### 3.4 Feature Importance (Tầm Quan Trọng Đặc Trưng)
```
Phiên bản:        34.35% (Quan trọng nhất)
Công tơ mét:      23.60%
Hãng xe:          19.78%
Đời xe:           13.03%
Dòng xe:           9.21%
Màu xe:            0.02% (Ít quan trọng)
```

**Giải thích:**
- Phiên bản quyết định giá nhất (động cơ, trang bị)
- Công tơ mét ảnh hưởng lớn (tình trạng xe)
- Hãng xe ảnh hưởng đáng kể (thương hiệu, độ tin cậy)
- Màu sắc ít ảnh hưởng đến giá

---

### **BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH (train_model.py)**

#### 4.1 Chỉ Số Đánh Giá

**Trên Tập Train (8.000 mẫu):**
```
R² Score = 0.9991
MAE = 5,267,037 VNĐ
```

**Trên Tập Test (2.000 mẫu):**
```
R² Score = 0.9978
MAE = 8,590,610 VNĐ
MAPE = 1.63%
```

#### 4.2 Công Thức Chỉ Số

**R² Score (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)

Ý nghĩa: 0.9978 = Mô hình giải thích 99.78% phương sai giá
Phạm vi: [0, 1], càng gần 1 càng tốt
```

**MAE (Mean Absolute Error):**
```
MAE = (1/n) × Σ|y_true - y_pred|
    = 8,590,610 VNĐ

Ý nghĩa: Trung bình sai số tuyệt đối ~8.6 triệu VNĐ
```

**MAPE (Mean Absolute Percentage Error):**
```
MAPE = (1/n) × Σ|((y_true - y_pred) / y_true) × 100%|
     = 1.63%

Ý nghĩa: Trung bình sai số phần trăm ~1.63%
Phạm vi: [0%, ∞), < 5% là rất tốt
```

#### 4.3 Phân Tích Kết Quả
- **Không overfitting:** R² train (0.9991) ≈ R² test (0.9978)
- **Độ chính xác cao:** MAPE = 1.63% < 5% (tiêu chuẩn tốt)
- **Sai số nhỏ:** MAE = 8.6 triệu VNĐ (< 2% giá trung bình)

---

### **BƯỚC 5: LƯU MÔ HÌ NH VÀ ARTIFACTS**

```
model.pkl                  → Mô hình Random Forest
encoders.pkl              → 4 bộ Label Encoder
feature_columns.pkl       → Danh sách 6 features
processed_data_raw.pkl    → Dữ liệu gốc đã xử lý
training_info.pkl         → Thông tin huấn luyện (R², MAE, MAPE)
```

---

## 🎯 THUẬT TOÁN DÙNG TRONG DỰ ĐOÁN

### **Quy Trình Dự Đoán (app.py)**

```
Input: Thông tin xe từ người dùng
├─ Hãng xe: "Toyota"
├─ Dòng xe: "Vios"
├─ Đời xe: 2020
├─ Phiên bản: "Toyota Vios 1.5E CVT"
├─ Màu xe: "Trắng"
└─ Công tơ mét: 50.000 km

↓ Mã hóa dữ liệu (Label Encoding)

Dữ liệu mã hóa: [1, 1, 2020, 5, 1, 50000]

↓ Dự đoán (Random Forest)

Mô hình chạy 100 cây quyết định:
- Mỗi cây dự đoán giá
- Lấy trung bình 100 dự đoán

↓ Output

Giá dự đoán: 420,000,000 VNĐ
Làm tròn: 420 triệu VNĐ
```

---

## 📈 HIỆU SUẤT MÔ HÌNH

| Chỉ Số | Giá Trị | Đánh Giá |
|--------|--------|---------|
| R² Score (Test) | 0.9978 | Xuất sắc |
| MAPE (Test) | 1.63% | Rất tốt |
| MAE (Test) | 8.6M VNĐ | Tốt |
| Overfitting | Không | Mô hình ổn định |

---

## 🔧 CÔNG NGHỆ SỬ DỤNG

- **Python 3.11**
- **scikit-learn:** Random Forest, Label Encoding, train_test_split
- **pandas:** Xử lý dữ liệu
- **numpy:** Tính toán số học
- **joblib:** Lưu/load mô hình
- **Streamlit:** Giao diện web

---

## 📝 DANH SÁCH FILE

| File | Mục Đích |
|------|----------|
| `generate_xe_cu_data.py` | Sinh 10.000 dòng dữ liệu |
| `train_model.py` | Huấn luyện mô hình |
| `app.py` | Giao diện web dự đoán |
| `xe_cu.csv` | Dữ liệu thô |
| `model.pkl` | Mô hình đã huấn luyện |
| `encoders.pkl` | Bộ encoder |
| `training_info.pkl` | Thông tin huấn luyện |

---

## 🚀 CÁCH CHẠY

```bash
# Bước 1: Sinh dữ liệu
python generate_xe_cu_data.py

# Bước 2: Huấn luyện mô hình
python train_model.py

# Bước 3: Chạy ứng dụng web
streamlit run app.py
```

Truy cập: http://localhost:8501
