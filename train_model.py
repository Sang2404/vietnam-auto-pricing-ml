"""
Script huấn luyện mô hình dự đoán giá xe ô tô cũ
So sánh 3 thuật toán: Random Forest, Gradient Boosting, XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
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
# 5. HUẤN LUYỆN 3 MÔ HÌNH
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 5: HUẤN LUYỆN 3 MÔ HÌNH")
print("=" * 70)

models_dict = {}

# Model 1: Random Forest Regressor
print("\n🌲 Huấn luyện Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models_dict['Random Forest'] = rf_model
print("✓ Random Forest hoàn tất!")

# Model 2: Gradient Boosting Regressor
print("\n📈 Huấn luyện Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train, y_train)
models_dict['Gradient Boosting'] = gb_model
print("✓ Gradient Boosting hoàn tất!")

# Model 3: XGBoost Regressor
print("\n⚡ Huấn luyện XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
models_dict['XGBoost'] = xgb_model
print("✓ XGBoost hoàn tất!")

# ============================================================================
# 6. ĐÁNH GIÁ CẢ 3 MÔ HÌNH
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 6: ĐÁNH GIÁ CẢ 3 MÔ HÌNH")
print("=" * 70)

models_performance = {}

for model_name, model in models_dict.items():
    print(f"\n{'='*50}")
    print(f"📊 {model_name}")
    print(f"{'='*50}")
    
    # Dự đoán trên tập Train
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Dự đoán trên tập Test
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Tính MAPE
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"\n📈 KẾT QUẢ TRÊN TẬP TRAIN:")
    print(f"   • R² Score: {train_r2:.4f}")
    print(f"   • MAE: {train_mae:,.0f} VNĐ")
    
    print(f"\n📉 KẾT QUẢ TRÊN TẬP TEST:")
    print(f"   • R² Score: {test_r2:.4f}")
    print(f"   • MAE: {test_mae:,.0f} VNĐ")
    print(f"   • MAPE: {mape:.2f}%")
    
    models_performance[model_name] = {
        'train_r2': train_r2,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'mape': mape
    }

# Tìm model tốt nhất dựa trên R² Score
print(f"\n{'='*70}")
print("🏆 SO SÁNH CÁC MÔ HÌNH")
print(f"{'='*70}\n")

best_model_name = max(models_performance, key=lambda x: models_performance[x]['test_r2'])
best_model = models_dict[best_model_name]
best_performance = models_performance[best_model_name]

comparison_df = pd.DataFrame(models_performance).T
print(comparison_df.to_string())

print(f"\n✨ MÔ HÌNH TỐT NHẤT: {best_model_name}")
print(f"   • R² Score: {best_performance['test_r2']:.4f}")
print(f"   • MAE: {best_performance['test_mae']:,.0f} VNĐ")

# ============================================================================
# 7. LƯU MÔ HÌNH VÀ ENCODER
# ============================================================================
print("\n" + "=" * 70)
print("BƯỚC 7: LƯU MÔ HÌNH VÀ ENCODER")
print("=" * 70)

# Lưu tất cả 3 mô hình
joblib.dump(models_dict, 'models.pkl')
print("✓ Đã lưu cả 3 mô hình: models.pkl")

# Lưu tên mô hình tốt nhất
joblib.dump(best_model_name, 'best_model_name.pkl')
print(f"✓ Đã lưu tên mô hình tốt nhất: best_model_name.pkl ({best_model_name})")

# Lưu hiệu suất của tất cả mô hình
joblib.dump(models_performance, 'models_performance.pkl')
print("✓ Đã lưu hiệu suất các mô hình: models_performance.pkl")

# Lưu các encoder
joblib.dump(encoders, 'encoders.pkl')
print("✓ Đã lưu encoders: encoders.pkl")

# Lưu thông tin về các cột features
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("✓ Đã lưu danh sách features: feature_columns.pkl")

# Lưu dữ liệu gốc đã xử lý để sử dụng trong web app
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
    'best_model': best_model_name,
    'models_performance': models_performance
}
joblib.dump(training_info, 'training_info.pkl')
print("✓ Đã lưu thông tin huấn luyện: training_info.pkl")

print("\n" + "=" * 70)
print("✅ HOÀN TẤT HUẤN LUYỆN 3 MÔ HÌNH!")
print("=" * 70)
print("\nCác file đã được lưu:")
print("  1. models.pkl - Cả 3 mô hình (Random Forest, Gradient Boosting, XGBoost)")
print("  2. best_model_name.pkl - Tên mô hình tốt nhất")
print("  3. models_performance.pkl - Hiệu suất của cả 3 mô hình")
print("  4. encoders.pkl - Các bộ encoder cho dữ liệu dạng chữ")
print("  5. feature_columns.pkl - Danh sách các features")
print("  6. processed_data_raw.pkl - Dữ liệu gốc đã xử lý")
print("  7. training_info.pkl - Thông tin huấn luyện")
print("\nBạn có thể sử dụng các file này trong ứng dụng web!")
