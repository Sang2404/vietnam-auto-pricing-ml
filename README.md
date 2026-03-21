# 🚗 Dự Đoán Giá Xe Ô Tô Cũ - Machine Learning

## 📋 Tổng Quan

Dự án so sánh 3 thuật toán ML để dự đoán giá xe cũ: Linear Regression, Random Forest, XGBoost.

## 📊 Dữ Liệu

- 12,164 mẫu xe (lọc bỏ xe "Mới")
- 7 features: Hãng xe, Dòng xe, Đời xe, Tuổi xe, Phiên bản, Màu xe, Công tơ mét, Tình trạng, Nhập khẩu
- Target: Giá (VNĐ)
- Chia tập: 80% Train / 20% Test

## 🤖 Thuật Toán

### 1. Linear Regression
Hồi quy tuyến tính cơ bản, giả định quan hệ tuyến tính giữa features và giá.

### 2. Random Forest
- 200 cây quyết định, max_depth=15
- Ensemble learning: trung bình kết quả từ nhiều cây
- Giảm overfitting, xác định feature quan trọng

### 3. XGBoost ⭐
- Gradient Boosting tối ưu: 200 estimators, learning_rate=0.1
- Xây dựng cây tuần tự, mỗi cây sửa lỗi cây trước
- Regularization (L1=0.1, L2=1.0) chống overfitting

## 📈 Kết Quả

| Model | R² | MAE | MAPE |
|---|---|---|---|
| Linear Regression | 0.0639 | 504M | 72.01% |
| Random Forest | 0.9014 | 150M | 17.72% |
| **XGBoost** | **0.9124** | **146M** | **17.72%** |

**XGBoost thắng:** R²=0.9124 (giải thích 91.24% phương sai), MAE=146M VNĐ.

## 🔧 Xử Lý Dữ Liệu

1. Lọc xe "Mới", giữ "Tốt", "Rất tốt", "Trung bình"
2. Xử lý missing: mode (text), median (số)
3. Feature Engineering: Tuổi xe = 2026 - Đời xe
4. Encoding: LabelEncoder (text), Ordinal (Tình trạng: 1→3)

## 🚀 Chạy

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 📁 Files

- `train_model.py`: Huấn luyện 3 mô hình
- `app.py`: Web app Streamlit
- `model.pkl`: XGBoost model
- `encoders.pkl`: Label encoders
- `training_info.pkl`: Metrics
