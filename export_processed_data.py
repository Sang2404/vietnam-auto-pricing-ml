"""
Script xuất dữ liệu đã xử lý ra file CSV
"""
import pandas as pd
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đọc dữ liệu gốc
df = pd.read_csv('xe_cu.csv', encoding='utf-8')
print(f"Dữ liệu gốc: {df.shape[0]} dòng")

# Lọc bỏ xe "Mới"
df = df[df['Tình trạng'] != 'Mới']
print(f"Sau lọc bỏ 'Mới': {df.shape[0]} dòng")

# Xử lý missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

df = df.dropna()
print(f"Sau xử lý missing: {df.shape[0]} dòng")

# Thêm feature "Tuổi xe"
current_year = 2026
df['Tuổi xe'] = current_year - df['Đời xe']

# Xuất ra file CSV
output_file = 'xe_cu_processed.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nĐã xuất dữ liệu đã xử lý ra: {output_file}")
print(f"Tổng số dòng: {len(df)}")
print(f"Các cột: {list(df.columns)}")
