"""
Script huan luyen mo hinh du doan gia xe o to cu
Su dung XGBoost Regressor voi cac dac trung phu hop
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DOC DU LIEU
# ============================================================================
print("=" * 70)
print("BUOC 1: DOC DU LIEU")
print("=" * 70)

df = pd.read_csv('xe_cu.csv', encoding='utf-8')
print(f"Da doc file du lieu: {df.shape[0]} dong, {df.shape[1]} cot")
print(f"Cac cot: {list(df.columns)}")

# Lọc bỏ "Mới", giữ lại "Tốt", "Rất tốt", "Trung bình"
print(f"\nTrước lọc: {df.shape[0]} dòng")
df = df[df['Tình trạng'] != 'Mới']
print(f"Sau lọc (bỏ 'Mới', giữ 'Tốt', 'Rất tốt', 'Trung bình'): {df.shape[0]} dòng")

# ============================================================================
# 2. TIEN XU LY DU LIEU
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 2: TIEN XU LY DU LIEU")
print("=" * 70)

print(f"\nGia tri trong truoc xu ly:")
print(df.isnull().sum())

# Xử lý giá trị trống
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Da xu ly gia tri trong o '{col}' bang: {mode_val}")
        else:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Da xu ly gia tri trong o '{col}' bang: {median_val}")

df = df.dropna()
print(f"Du lieu sau xu ly: {df.shape[0]} dong")

# ============================================================================
# 3. TAO FEATURE MOI - TUOI XE
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 3: TAO FEATURE MOI")
print("=" * 70)

# Tinh tuoi xe (nam hien tai - doi xe)
current_year = 2026
df['Tuổi xe'] = current_year - df['Đời xe']
print(f"Da tao feature 'Tuổi xe': min={df['Tuổi xe'].min()}, max={df['Tuổi xe'].max()}")

# ============================================================================
# 4. CHUYEN DOI DU LIEU DANG CHU SANG SO
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 4: MA HOA DU LIEU")
print("=" * 70)

# Cac cot phan loai can ma hoa (KHÔNG bao gồm 'Tình trạng')
categorical_columns = ['Hãng xe', 'Dòng xe', 'Phiên bản', 'Màu xe', 'Nhập khẩu']

encoders = {}

# Ma hoa cac cot thong thuong bang LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"Da ma hoa '{col}': {len(le.classes_)} gia tri")

# Ma hoa 'Tình trạng' bang Ordinal Encoding (thủ công theo logic)
# Số càng lớn = xe càng đắt
mapping_tinh_trang = {'Trung bình': 1, 'Tốt': 2, 'Rất tốt': 3, 'Mới': 4}
df['Tình trạng'] = df['Tình trạng'].map(mapping_tinh_trang)
print(f"Da ma hoa 'Tình trạng' bang Ordinal Encoding:")
print(f"   Trung bình=1, Tốt=2, Rất tốt=3, Mới=4")

# Lưu mapping để dùng trong app.py
encoders['mapping_tinh_trang'] = mapping_tinh_trang

# ============================================================================
# 5. CHIA TAP DU LIEU
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 5: CHIA TAP DU LIEU")
print("=" * 70)

# Features: tat ca cac cot truoc 'Gia (VND)'
X = df.drop('Giá (VNĐ)', axis=1)
y = df['Giá (VNĐ)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Tap Train: {X_train.shape[0]} mau ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Tap Test: {X_test.shape[0]} mau ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"So features: {X_train.shape[1]}")
print(f"Features: {list(X.columns)}")

# ============================================================================
# 6. HUAN LUYEN 3 MO HINH
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 6: HUAN LUYEN 3 MO HINH (Linear Regression, Random Forest, XGBoost)")
print("=" * 70)

# 1. Linear Regression
print("\n[1/3] Dang huan luyen Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred_test = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred_test)
lr_mae = mean_absolute_error(y_test, lr_pred_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred_test))
print("✓ Huan luyen Linear Regression hoan tat!")

# 2. Random Forest
print("\n[2/3] Dang huan luyen Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred_test = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred_test)
rf_mae = mean_absolute_error(y_test, rf_pred_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))
print("✓ Huan luyen Random Forest hoan tat!")

# 3. XGBoost
print("\n[3/3] Dang huan luyen XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred_test = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred_test))
print("✓ Huan luyen XGBoost hoan tat!")

# ============================================================================
# 7. SO SANH 3 MO HINH
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 7: SO SANH 3 MO HINH")
print("=" * 70)

# Tao bang so sanh
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R² Score': [lr_r2, rf_r2, xgb_r2],
    'MAE': [lr_mae, rf_mae, xgb_mae],
    'RMSE': [lr_rmse, rf_rmse, xgb_rmse]
})

print("\n" + "=" * 70)
print("BANG SO SANH TREN TAP TEST:")
print("=" * 70)
print(comparison_df.to_string(index=False))

# Tim mo hinh tot nhat
best_idx = comparison_df['R² Score'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']
best_r2 = comparison_df.loc[best_idx, 'R² Score']

print("\n" + "=" * 70)
print(f"🏆 MO HINH CHIEN THANG: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print("=" * 70)

# Chon mo hinh tot nhat
if best_model_name == 'Linear Regression':
    best_model = lr_model
    best_model_pred = lr_pred_test
elif best_model_name == 'Random Forest':
    best_model = rf_model
    best_model_pred = rf_pred_test
else:  # XGBoost
    best_model = xgb_model
    best_model_pred = xgb_pred_test

# ============================================================================
# 8. DANH GIA MO HINH CHIEN THANG
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 8: DANH GIA MO HINH CHIEN THANG")
print("=" * 70)

# Du doan tren tap Train
y_train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Du doan tren tap Test
test_r2 = best_r2
test_mae = comparison_df.loc[best_idx, 'MAE']
test_rmse = comparison_df.loc[best_idx, 'RMSE']

print(f"\nKET QUA TREN TAP TRAIN:")
print(f"   R-squared: {train_r2:.4f}")
print(f"   MAE: {train_mae:,.0f} VND")
print(f"   RMSE: {train_rmse:,.0f} VND")

print(f"\nKET QUA TREN TAP TEST:")
print(f"   R-squared: {test_r2:.4f}")
print(f"   MAE: {test_mae:,.0f} VND")
print(f"   RMSE: {test_rmse:,.0f} VND")

# Tinh MAPE (bo qua gia 0)
mask = y_test > 0
mape = np.mean(np.abs((y_test[mask] - best_model_pred[mask]) / y_test[mask])) * 100
print(f"   MAPE: {mape:.2f}%")

# Kiem tra overfitting
print(f"\n--- KIEM TRA OVERFITTING ---")
print(f"Chenhlech R2 (Train - Test): {train_r2 - test_r2:.4f}")
if train_r2 - test_r2 > 0.1:
    print("CANH BAO: Co dau hieu overfitting!")
else:
    print("OK: Khong co dau hieu overfitting nghiem trong.")

# Hien thi feature importance (neu co)
if hasattr(best_model, 'feature_importances_'):
    print(f"\nTAM QUAN TRONG CUA CAC FEATURES:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    for idx, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
else:
    print(f"\nMo hinh {best_model_name} khong co feature importance.")

# ============================================================================
# 9. LUU MO HINH CHIEN THANG VA ENCODER
# ============================================================================
print("\n" + "=" * 70)
print("BUOC 9: LUU MO HINH CHIEN THANG VA ENCODER")
print("=" * 70)

joblib.dump(best_model, 'model.pkl')
print(f"Da luu mo hinh chiến thắng ({best_model_name}): model.pkl")

joblib.dump(encoders, 'encoders.pkl')
print("Da luu encoders: encoders.pkl")

joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("Da luu feature_columns.pkl")

# Luu du lieu goc
df_raw = pd.read_csv('xe_cu.csv', encoding='utf-8')
for col in df_raw.columns:
    if df_raw[col].isnull().sum() > 0:
        if df_raw[col].dtype == 'object':
            df_raw[col].fillna(df_raw[col].mode()[0], inplace=True)
        else:
            df_raw[col].fillna(df_raw[col].median(), inplace=True)
df_raw = df_raw.dropna()

# Them tuoi xe vao du lieu goc
df_raw['Tuổi xe'] = current_year - df_raw['Đời xe']
joblib.dump(df_raw, 'processed_data_raw.pkl')
print("Da luu processed_data_raw.pkl")

# Luu thong tin huan luyen
training_info = {
    'total_data': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'mape': mape,
    'overfitting_check': train_r2 - test_r2,
    'features': list(X.columns),
    'best_model': best_model_name
}
joblib.dump(training_info, 'training_info.pkl')
print("Da luu training_info.pkl")

print("\n" + "=" * 70)
print("HOAN TAT!")
print("=" * 70)
