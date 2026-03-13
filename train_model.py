"""
Script huấn luyện mô hình dự đoán giá xe ô tô cũ
Sử dụng Random Forest Regressor để dự đoán giá dựa trên các đặc trưng của xe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================================
print("=" * 70)
print("BƯỚC 1: ĐỌC DỮ LIỆU")
print("=" * 70)

df = pd.read_csv('xe_cu.csv')
print(f"✓ Đã đọc file dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
print(f"✓ Các cột: {list(df.columns)}")
print(f"\nThông tin dữ liệu:\n{df.info()}")

# ============================================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 70)

# Kiểm tra giá trị trống
print(f"\nGiá trị trống trước xử lý:")
print(df.isnull().sum())

# Xử lý giá trị trống ở cột 'Màu xe' - điền bằng mode (giá trị xuất hiện nhiều nhất)
if df['Màu xe'].isnull().sum() > 0:
    color_mode = df['Màu xe'].mode()[0]
    df['Màu xe'].fillna(color_mode, inplace=True)
    print(f"✓ Đã xử lý giá trị trống ở 'Màu xe' bằng: {color_mode}")

# Xử lý giá trị trống ở các cột khác (nếu có)
df = df.dropna()
print(f"✓ Đã xóa các dòng có giá trị trống còn lại")
print(f"✓ Dữ liệu sau xử lý: {df.shape[0]} dòng")

# ============================================================================
# 3. CHUYỂN ĐỔI DỮ LIỆU DẠNG CHỮ SANG SỐ (LABEL ENCODING)
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 3: CHUYỂN ĐỔI DỮ LIỆU DẠNG CHỮ SANG SỐ")
print("=" * 70)

# Các cột cần mã hóa
categorical_columns = ['Hãng xe', 'Dòng xe', 'Phiên bản', 'Màu xe']

# Tạo dictionary để lưu các encoder
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"✓ Đã mã hóa '{col}': {len(le.classes_)} giá trị")

print(f"\n✓ Dữ liệu sau mã hóa:\n{df.head()}")

# ============================================================================
# 4. CHIA TẬP DỮ LIỆU (80% TRAIN, 20% TEST)
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 4: CHIA TẬP DỮ LIỆU")
print("=" * 70)

# Tách features (X) và target (y)
X = df.drop('Giá (VNĐ)', axis=1)
y = df['Giá (VNĐ)']

# Chia dữ liệu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Tập Train: {X_train.shape[0]} mẫu ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Tập Test: {X_test.shape[0]} mẫu ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Số features: {X_train.shape[1]}")

# ============================================================================
# 5. HUẤN LUYỆN MÔ HÌNH (RANDOM FOREST REGRESSOR)
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 5: HUẤN LUYỆN MÔ HÌNH")
print("=" * 70)

# Tạo và huấn luyện mô hình Random Forest
model = RandomForestRegressor(
    n_estimators=100,      # Số cây trong rừng
    max_depth=20,          # Độ sâu tối đa của mỗi cây
    min_samples_split=5,   # Số mẫu tối thiểu để chia một node
    min_samples_leaf=2,    # Số mẫu tối thiểu ở leaf node
    random_state=42,
    n_jobs=-1              # Sử dụng tất cả CPU cores
)

print("Đang huấn luyện mô hình...")
model.fit(X_train, y_train)
print("✓ Huấn luyện hoàn tất!")

# ============================================================================
# 6. ĐÁNH GIÁ MÔ HÌNH
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH")
print("=" * 70)

# Dự đoán trên tập Train
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Dự đoán trên tập Test
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n📊 KẾT QUẢ TRÊN TẬP TRAIN:")
print(f"   • R² Score: {train_r2:.4f}")
print(f"   • MAE (Mean Absolute Error): {train_mae:,.0f} VNĐ")

print(f"\n📊 KẾT QUẢ TRÊN TẬP TEST:")
print(f"   • R² Score: {test_r2:.4f}")
print(f"   • MAE (Mean Absolute Error): {test_mae:,.0f} VNĐ")

# Tính MAPE (Mean Absolute Percentage Error) để đánh giá tốt hơn
mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
print(f"   • MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

# Hiển thị feature importance
print(f"\n🎯 TẦM QUAN TRỌNG CỦA CÁC FEATURES:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"   • {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# 7. LƯU MÔ HÌNH VÀ ENCODER
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 7: LƯU MÔ HÌNH VÀ ENCODER")
print("=" * 70)

# Lưu mô hình
joblib.dump(model, 'model.pkl')
print("✓ Đã lưu mô hình: model.pkl")

# Lưu các encoder
joblib.dump(encoders, 'encoders.pkl')
print("✓ Đã lưu encoders: encoders.pkl")

# Lưu thông tin về các cột features
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("✓ Đã lưu danh sách features: feature_columns.pkl")

# Lưu dữ liệu gốc đã xử lý để sử dụng trong web app
# Tạo dataframe với dữ liệu gốc (chưa mã hóa) để lấy danh sách các giá trị
df_raw = pd.read_csv('xe_cu.csv')
if df_raw['Màu xe'].isnull().sum() > 0:
    color_mode = df_raw['Màu xe'].mode()[0]
    df_raw['Màu xe'].fillna(color_mode, inplace=True)
df_raw = df_raw.dropna()
joblib.dump(df_raw, 'processed_data_raw.pkl')
print("✓ Đã lưu dữ liệu gốc đã xử lý: processed_data_raw.pkl")

# Lưu thông tin về dữ liệu huấn luyện
training_info = {
    'total_data': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_percentage': (len(X_train) / len(df)) * 100,
    'test_percentage': (len(X_test) / len(df)) * 100,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'mape': mape
}
joblib.dump(training_info, 'training_info.pkl')
print("✓ Đã lưu thông tin huấn luyện: training_info.pkl")

print("\n" + "=" * 70)
print("✅ HOÀN TẤT HUẤN LUYỆN MÔ HÌNH!")
print("=" * 70)
print("\nCác file đã được lưu:")
print("  1. model.pkl - Mô hình Random Forest")
print("  2. encoders.pkl - Các bộ encoder cho dữ liệu dạng chữ")
print("  3. feature_columns.pkl - Danh sách các features")
print("  4. processed_data_raw.pkl - Dữ liệu gốc đã xử lý")
print("  5. training_info.pkl - Thông tin huấn luyện")
print("\nBạn có thể sử dụng các file này trong ứng dụng web!")
