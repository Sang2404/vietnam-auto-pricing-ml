"""
Script sinh dữ liệu xe ô tô cũ Việt Nam 2026 - BẢN MỞ RỘNG
Bao phủ 50+ dòng xe phổ biến tại Việt Nam
"""

import pandas as pd
import numpy as np
import random

# ============================================================================
# PHẦN 1: DICTIONARY CAR_DB - DANH SÁCH HÃ XE VÀ GIÁ GỐC (MỞ RỘNG)
# ============================================================================

car_db = {
    # ===== TOYOTA (14 dòng) =====
    'Toyota': {
        'Wigo': {
            'Toyota Wigo 1.2 MT': 280_000_000,
            'Toyota Wigo 1.2 AT': 330_000_000,
        },
        'Vios': {
            'Toyota Vios 1.5E MT': 360_000_000,
            'Toyota Vios 1.5E CVT': 420_000_000,
            'Toyota Vios 1.5GR-S CVT': 520_000_000,
        },
        'Yaris': {
            'Toyota Yaris 1.5 MT': 420_000_000,
            'Toyota Yaris 1.5 CVT': 500_000_000,
        },
        'Raize': {
            'Toyota Raize 1.0 Turbo MT': 480_000_000,
            'Toyota Raize 1.0 Turbo CVT': 550_000_000,
        },
        'Corolla Altis': {
            'Toyota Corolla Altis 1.6 MT': 650_000_000,
            'Toyota Corolla Altis 1.8 CVT': 750_000_000,
        },
        'Corolla Cross': {
            'Toyota Corolla Cross 1.8 CVT': 750_000_000,
            'Toyota Corolla Cross 1.8 Hybrid': 850_000_000,
        },
        'Yaris Cross': {
            'Toyota Yaris Cross 1.5 CVT': 650_000_000,
            'Toyota Yaris Cross 1.5 Hybrid': 750_000_000,
        },
        'Camry': {
            'Toyota Camry 2.5 AT': 1_200_000_000,
            'Toyota Camry 2.5 Hybrid': 1_350_000_000,
        },
        'Innova': {
            'Toyota Innova 2.0 MT': 750_000_000,
            'Toyota Innova 2.0 AT': 850_000_000,
        },
        'Veloz Cross': {
            'Toyota Veloz Cross 1.5 MT': 550_000_000,
            'Toyota Veloz Cross 1.5 AT': 650_000_000,
        },
        'Avanza': {
            'Toyota Avanza 1.3 MT': 420_000_000,
            'Toyota Avanza 1.3 AT': 500_000_000,
        },
        'Fortuner': {
            'Toyota Fortuner 2.4 MT': 1_050_000_000,
            'Toyota Fortuner 2.8 AT': 1_250_000_000,
        },
        'Hilux': {
            'Toyota Hilux 2.4 MT': 650_000_000,
            'Toyota Hilux 2.8 AT': 850_000_000,
        },
        'Land Cruiser Prado': {
            'Toyota Land Cruiser Prado 2.7 AT': 1_650_000_000,
            'Toyota Land Cruiser Prado 2.8 Diesel': 1_850_000_000,
        },
    },
    
    # ===== HYUNDAI (9 dòng) =====
    'Hyundai': {
        'Grand i10': {
            'Hyundai Grand i10 1.2 MT': 320_000_000,
            'Hyundai Grand i10 1.2 AT': 380_000_000,
        },
        'Accent': {
            'Hyundai Accent 1.4 MT': 380_000_000,
            'Hyundai Accent 1.4 AT': 450_000_000,
            'Hyundai Accent 1.4 AT Đặc biệt': 520_000_000,
        },
        'Elantra': {
            'Hyundai Elantra 1.6 MT': 580_000_000,
            'Hyundai Elantra 1.6 AT': 680_000_000,
        },
        'Creta': {
            'Hyundai Creta 1.6 MT': 650_000_000,
            'Hyundai Creta 1.6 AT': 750_000_000,
        },
        'Tucson': {
            'Hyundai Tucson 2.0 MT': 750_000_000,
            'Hyundai Tucson 2.0 AT': 850_000_000,
        },
        'Santa Fe': {
            'Hyundai Santa Fe 2.4 AT': 1_050_000_000,
            'Hyundai Santa Fe 2.4 Turbo': 1_200_000_000,
        },
        'Stargazer': {
            'Hyundai Stargazer 1.5 MT': 480_000_000,
            'Hyundai Stargazer 1.5 AT': 580_000_000,
        },
        'Custin': {
            'Hyundai Custin 1.5 Turbo MT': 650_000_000,
            'Hyundai Custin 1.5 Turbo AT': 750_000_000,
        },
        'Palisade': {
            'Hyundai Palisade 2.2 Diesel': 1_350_000_000,
            'Hyundai Palisade 2.2 Diesel AWD': 1_500_000_000,
        },
    },
    
    # ===== KIA (10 dòng) =====
    'Kia': {
        'Morning': {
            'Kia Morning 1.0 MT': 280_000_000,
            'Kia Morning 1.0 AT': 330_000_000,
        },
        'Soluto': {
            'Kia Soluto 1.4 MT': 350_000_000,
            'Kia Soluto 1.4 AT': 420_000_000,
        },
        'K3/Cerato': {
            'Kia K3 1.6 MT': 520_000_000,
            'Kia K3 1.6 AT': 620_000_000,
            'Kia K3 1.6 AT Luxury': 750_000_000,
        },
        'K5/Optima': {
            'Kia K5 1.6 Turbo': 850_000_000,
            'Kia K5 2.0 Turbo': 950_000_000,
        },
        'Sonet': {
            'Kia Sonet 1.5 MT': 480_000_000,
            'Kia Sonet 1.5 AT': 580_000_000,
        },
        'Seltos': {
            'Kia Seltos 1.4 Turbo': 650_000_000,
            'Kia Seltos 1.6 AT': 750_000_000,
        },
        'Sportage': {
            'Kia Sportage 1.6 Turbo': 850_000_000,
            'Kia Sportage 2.0 AT': 950_000_000,
        },
        'Sorento': {
            'Kia Sorento 2.4 AT': 950_000_000,
            'Kia Sorento 2.4 Turbo': 1_100_000_000,
        },
        'Carnival/Sedona': {
            'Kia Carnival 3.5 V6': 1_350_000_000,
            'Kia Carnival 3.5 V6 Luxury': 1_550_000_000,
        },
        'Carens': {
            'Kia Carens 1.5 MT': 550_000_000,
            'Kia Carens 1.5 AT': 650_000_000,
        },
    },
    
    # ===== HONDA (6 dòng) =====
    'Honda': {
        'Brio': {
            'Honda Brio 1.2 MT': 350_000_000,
            'Honda Brio 1.2 AT': 420_000_000,
        },
        'City': {
            'Honda City 1.5 MT': 450_000_000,
            'Honda City 1.5 CVT': 520_000_000,
        },
        'Civic': {
            'Honda Civic 1.5 Turbo': 750_000_000,
            'Honda Civic 1.8 CVT': 850_000_000,
        },
        'HR-V': {
            'Honda HR-V 1.8 MT': 650_000_000,
            'Honda HR-V 1.8 CVT': 750_000_000,
        },
        'CR-V': {
            'Honda CR-V 1.5L G': 850_000_000,
            'Honda CR-V 1.5L L': 950_000_000,
            'Honda CR-V 1.5L L AWD': 1_100_000_000,
        },
        'Accord': {
            'Honda Accord 1.5 Turbo': 900_000_000,
            'Honda Accord 2.0 Hybrid': 1_050_000_000,
        },
    },
    
    # ===== MAZDA (8 dòng) =====
    'Mazda': {
        'Mazda 2': {
            'Mazda 2 1.5 MT': 480_000_000,
            'Mazda 2 1.5 AT': 550_000_000,
        },
        'Mazda 3': {
            'Mazda 3 1.5 MT': 650_000_000,
            'Mazda 3 1.5 AT': 750_000_000,
        },
        'Mazda 6': {
            'Mazda 6 2.0 AT': 850_000_000,
            'Mazda 6 2.5 AT': 950_000_000,
        },
        'CX-3': {
            'Mazda CX-3 1.5 MT': 580_000_000,
            'Mazda CX-3 1.5 AT': 680_000_000,
        },
        'CX-30': {
            'Mazda CX-30 1.5 MT': 650_000_000,
            'Mazda CX-30 1.5 AT': 750_000_000,
        },
        'CX-5': {
            'Mazda CX-5 2.0L 2WD': 850_000_000,
            'Mazda CX-5 2.5L AWD': 1_050_000_000,
            'Mazda CX-5 2.5L Turbo AWD': 1_200_000_000,
        },
        'CX-8': {
            'Mazda CX-8 2.5 AT': 1_050_000_000,
            'Mazda CX-8 2.5 Turbo': 1_200_000_000,
        },
        'BT-50': {
            'Mazda BT-50 2.2 MT': 650_000_000,
            'Mazda BT-50 2.2 AT': 750_000_000,
        },
    },
    
    # ===== FORD (5 dòng) =====
    'Ford': {
        'EcoSport': {
            'Ford EcoSport 1.5 MT': 520_000_000,
            'Ford EcoSport 1.5 AT': 620_000_000,
        },
        'Ranger': {
            'Ford Ranger XL 2.2L MT 4x4': 650_000_000,
            'Ford Ranger XLS 2.2L AT 4x2': 750_000_000,
            'Ford Ranger Wildtrak 2.0L AT 4x4': 950_000_000,
            'Ford Ranger Raptor 2.0L AT 4x4': 1_100_000_000,
        },
        'Everest': {
            'Ford Everest 2.0 MT': 850_000_000,
            'Ford Everest 2.0 AT': 950_000_000,
        },
        'Explorer': {
            'Ford Explorer 2.3 Turbo': 1_350_000_000,
            'Ford Explorer 3.0 Turbo': 1_550_000_000,
        },
        'Territory': {
            'Ford Territory 1.5 Turbo': 850_000_000,
            'Ford Territory 1.5 Turbo AWD': 950_000_000,
        },
    },
    
    # ===== MITSUBISHI (6 dòng) =====
    'Mitsubishi': {
        'Attrage': {
            'Mitsubishi Attrage 1.2 MT': 320_000_000,
            'Mitsubishi Attrage 1.2 AT': 380_000_000,
        },
        'Xpander': {
            'Mitsubishi Xpander 1.5 MT': 550_000_000,
            'Mitsubishi Xpander 1.5 AT': 650_000_000,
        },
        'Xpander Cross': {
            'Mitsubishi Xpander Cross 1.5 MT': 650_000_000,
            'Mitsubishi Xpander Cross 1.5 AT': 750_000_000,
        },
        'Outlander': {
            'Mitsubishi Outlander 2.0 MT': 750_000_000,
            'Mitsubishi Outlander 2.4 AT': 900_000_000,
        },
        'Pajero Sport': {
            'Mitsubishi Pajero Sport 2.4 AT': 950_000_000,
            'Mitsubishi Pajero Sport 2.4 Diesel': 1_050_000_000,
        },
        'Triton': {
            'Mitsubishi Triton 2.4 MT': 650_000_000,
            'Mitsubishi Triton 2.4 AT': 750_000_000,
        },
    },
    
    # ===== VINFAST (9 dòng) =====
    'VinFast': {
        'Fadil': {
            'VinFast Fadil 1.2 MT': 380_000_000,
            'VinFast Fadil 1.2 AT': 450_000_000,
        },
        'Lux A2.0': {
            'VinFast Lux A2.0 2.0 AT': 750_000_000,
            'VinFast Lux A2.0 2.0 Turbo': 850_000_000,
        },
        'Lux SA2.0': {
            'VinFast Lux SA2.0 2.0 AT': 850_000_000,
            'VinFast Lux SA2.0 2.0 Turbo': 950_000_000,
        },
        'VF3': {
            'VinFast VF3 Plus': 450_000_000,
            'VinFast VF3 Plus EV': 550_000_000,
        },
        'VF5': {
            'VinFast VF5 Plus': 550_000_000,
            'VinFast VF5 Plus EV': 650_000_000,
        },
        'VF e34': {
            'VinFast VF e34 Standard': 650_000_000,
            'VinFast VF e34 Premium': 750_000_000,
        },
        'VF6': {
            'VinFast VF6 Plus': 750_000_000,
            'VinFast VF6 Plus EV': 850_000_000,
        },
        'VF8': {
            'VinFast VF8 Plus': 850_000_000,
            'VinFast VF8 Plus EV': 950_000_000,
        },
        'VF9': {
            'VinFast VF9 Plus': 1_200_000_000,
            'VinFast VF9 Plus EV': 1_350_000_000,
        },
    },
    
    # ===== MERCEDES-BENZ (3 dòng) =====
    'Mercedes-Benz': {
        'C-Class': {
            'Mercedes-Benz C200 1.5 Turbo': 1_500_000_000,
            'Mercedes-Benz C300 2.0 Turbo': 1_750_000_000,
        },
        'E-Class': {
            'Mercedes-Benz E180 1.5 Turbo': 1_950_000_000,
            'Mercedes-Benz E200 2.0 Turbo': 2_150_000_000,
            'Mercedes-Benz E300 2.0 Turbo': 2_450_000_000,
        },
        'GLC': {
            'Mercedes-Benz GLC200 1.5 Turbo': 1_750_000_000,
            'Mercedes-Benz GLC300 2.0 Turbo': 2_100_000_000,
        },
    },
    
    # ===== BMW (3 dòng) =====
    'BMW': {
        '3 Series': {
            'BMW 320i 2.0 Turbo': 1_450_000_000,
            'BMW 330i 2.0 Turbo': 1_750_000_000,
        },
        '5 Series': {
            'BMW 520i 2.0 Turbo': 1_950_000_000,
            'BMW 530i 3.0 Turbo': 2_350_000_000,
        },
        'X3': {
            'BMW X3 20i 2.0 Turbo': 1_850_000_000,
            'BMW X3 30i 2.0 Turbo': 2_150_000_000,
        },
    },
    
    # ===== LEXUS (5 dòng) =====
    'Lexus': {
        'IS': {
            'Lexus IS 250 2.5 AT': 1_650_000_000,
            'Lexus IS 300 3.0 V6': 1_950_000_000,
        },
        'ES': {
            'Lexus ES 250 2.5 Hybrid': 1_850_000_000,
            'Lexus ES 350 3.5 V6': 2_150_000_000,
        },
        'NX': {
            'Lexus NX 300 2.0 Turbo': 1_750_000_000,
            'Lexus NX 350 3.5 V6': 2_050_000_000,
        },
        'RX': {
            'Lexus RX 350 3.5 V6': 2_250_000_000,
            'Lexus RX 450h 3.5 Hybrid': 2_550_000_000,
        },
        'GX': {
            'Lexus GX 460 4.6 V8': 2_850_000_000,
        },
    },
}

# ============================================================================
# PHẦN 2: DANH SÁCH MÀU XE
# ============================================================================

colors = ['Trắng', 'Đen', 'Bạc', 'Xanh', 'Đỏ', 'Ghi', 'Nâu', 'Vàng']

# ============================================================================
# PHẦN 3: HÀM TÍNH HỆ SỐ KHẤU HAO
# ============================================================================

def get_depreciation_rate(brand):
    """Tính hệ số khấu hao hàng năm theo hãng xe"""
    if brand in ['Toyota', 'Honda', 'Lexus']:
        return np.random.uniform(0.06, 0.07)
    elif brand in ['Mazda', 'Mitsubishi', 'Suzuki', 'Nissan', 'Subaru', 'Isuzu']:
        return np.random.uniform(0.07, 0.08)
    elif brand in ['Hyundai', 'Kia', 'Ford', 'Peugeot', 'Volkswagen', 'MG']:
        return np.random.uniform(0.08, 0.10)
    elif brand == 'VinFast':
        return np.random.uniform(0.10, 0.13)
    elif brand in ['Mercedes-Benz', 'BMW', 'Audi', 'Volvo']:
        return np.random.uniform(0.10, 0.12)
    elif brand in ['Porsche']:
        return np.random.uniform(0.12, 0.15)
    else:
        return 0.09

# ============================================================================
# PHẦN 4: HÀM TÍNH GIÁ XE CŨ
# ============================================================================

def calculate_used_car_price(brand, base_price, year, mileage):
    """Tính giá xe cũ dựa trên giá gốc, năm, và số km"""
    current_year = 2026
    age = current_year - year
    depreciation_rate = get_depreciation_rate(brand)
    price_after_age = base_price * ((1 - depreciation_rate) ** age)
    standard_mileage = age * 15_000
    mileage_diff = mileage - standard_mileage
    mileage_adjustment = (mileage_diff / 1000) * 0.005 * price_after_age
    final_price = price_after_age - mileage_adjustment
    random_factor = np.random.uniform(0.97, 1.03)
    final_price = final_price * random_factor
    final_price = round(final_price / 1_000_000) * 1_000_000
    return max(final_price, base_price * 0.3)

# ============================================================================
# PHẦN 5: SINH DỮ LIỆU
# ============================================================================

print("=" * 70)
print("SINH DỮ LIỆU XE Ô TÔ CŨ VIỆT NAM 2026 - BẢN MỞ RỘNG")
print("=" * 70)

data = []
brands_list = list(car_db.keys())
num_brands = len(brands_list)
rows_per_brand = 10_000 // num_brands

print(f"\nTổng hãng xe: {num_brands}")
print(f"Dòng dữ liệu/hãng: ~{rows_per_brand}")

for brand_idx, brand in enumerate(brands_list):
    print(f"\n[{brand_idx+1}/{num_brands}] Sinh dữ liệu cho {brand}...")
    models_dict = car_db[brand]
    models_list = list(models_dict.keys())
    
    for i in range(rows_per_brand):
        model = random.choice(models_list)
        versions_dict = models_dict[model]
        version = random.choice(list(versions_dict.keys()))
        base_price = versions_dict[version]
        year = random.randint(2016, 2026)
        age = 2026 - year
        if age == 0:
            mileage = random.randint(100, 20_000)
        else:
            standard = age * 15_000
            mileage = int(standard * random.uniform(0.7, 1.3))
        color = random.choice(colors)
        price = calculate_used_car_price(brand, base_price, year, mileage)
        
        data.append({
            'Hãng xe': brand,
            'Dòng xe': model,
            'Đời xe': year,
            'Phiên bản': version,
            'Màu xe': color,
            'Công tơ mét (km)': mileage,
            'Giá (VNĐ)': int(price)
        })

df = pd.DataFrame(data)
df.to_csv('xe_cu.csv', index=False, encoding='utf-8')

print("\n" + "=" * 70)
print("✅ HOÀN TẤT SINH DỮ LIỆU!")
print("=" * 70)
print(f"\n📊 Thống kê dữ liệu:")
print(f"   • Tổng dòng: {len(df):,}")
print(f"   • Tổng hãng xe: {df['Hãng xe'].nunique()}")
print(f"   • Tổng dòng xe: {df['Dòng xe'].nunique()}")
print(f"   • Tổng phiên bản: {df['Phiên bản'].nunique()}")
print(f"   • Giá trung bình: {df['Giá (VNĐ)'].mean():,.0f} VNĐ")

print(f"\n📋 Danh sách hãng xe:")
for brand in sorted(df['Hãng xe'].unique()):
    count = len(df[df['Hãng xe'] == brand])
    print(f"   • {brand}: {count} xe")

print(f"\n✓ File 'xe_cu.csv' đã được lưu!")
