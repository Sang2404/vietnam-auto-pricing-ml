# 🚗 Dự Đoán Giá Xe Ô Tô Cũ - Machine Learning

## � Tóm Tắt

Dự án xây dựng mô hình Machine Learning dự đoán giá xe ô tô cũ bằng cách so sánh **3 thuật toán**: Random Forest, Gradient Boosting, XGBoost.

---

## 📊 Dữ Liệu

- **Tổng mẫu:** 9,999 xe
- **Features:** 6 (Hãng xe, Dòng xe, Đời xe, Phiên bản, Màu xe, Công tơ mét)
- **Target:** Giá xe (VNĐ)
- **Chia tập:** 80% Train / 20% Test

---

## 🤖 3 Thuật Toán

### 1. Random Forest Regressor
```python
RandomForestRegressor(n_estimators=50, max_depth=10, 
                      min_samples_split=10, min_samples_leaf=5)
```

### 2. Gradient Boosting Regressor
```python
GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                          max_depth=4, min_samples_split=10)
```

### 3. XGBoost Regressor
```python
XGBRegressor(n_estimators=100, learning_rate=0.05, 
             max_depth=4, min_child_weight=5)
```

---

## � Kết Quả

| Thuật Toán | R² Score | MAE | MAPE |
|---|---|---|---|
| **Random Forest** ⭐ | **0.9763** | **53.1M VNĐ** | **10.22%** |
| Gradient Boosting | 0.9125 | 102.0M VNĐ | 19.67% |
| XGBoost | 0.9080 | 104.1M VNĐ | 20.15% |

**Model tốt nhất: Random Forest** (R² = 0.9763)

---

## 🚀 Chạy Dự Án

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện
```bash
python train_model.py
```

### 3. Chạy web
```bash
streamlit run app.py
```

---

## 📁 Cấu Trúc

```
├── train_model.py          # Huấn luyện 3 model
├── app.py                  # Ứng dụng web
├── xe_cu.csv               # Dữ liệu
├── models.pkl              # 3 mô hình
├── models_performance.pkl  # Kết quả
└── requirements.txt        # Dependencies
```

