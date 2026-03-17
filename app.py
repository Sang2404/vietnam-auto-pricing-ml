"""
Ứng dụng Web dự đoán giá xe ô tô cũ sử dụng Streamlit
Giao diện hiện đại với phong cách premium dark theme
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================================================
# CẤU HÌNH TRANG
# ============================================================================
st.set_page_config(
    page_title="Dự đoán Giá Xe Ô Tô Cũ",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS TÙYCHỈNH - LIGHT THEME VỚI XANH BIỂN VÀ XANH LÁ CÂY
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Main background - Light Theme */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 50%, #e8f2f7 100%);
        min-height: 100vh;
    }
    
    /* Header section */
    .hero-section {
        text-align: center;
        padding: 40px 20px 24px;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(2, 132, 199, 0.08) 0%, transparent 50%);
        animation: float 15s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(20px, 20px) rotate(5deg); }
    }
    
    .hero-title {
        font-size: 3.2em;
        font-weight: 900;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #16a34a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
        position: relative;
        animation: slideUp 0.8s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-subtitle {
        font-size: 1.05em;
        color: #1e293b;
        margin-top: 8px;
        font-weight: 400;
        letter-spacing: 0.5px;
        position: relative;
        animation: slideUp 0.8s ease-out 0.1s both;
    }
    
    /* Glass card - Light Theme */
    .glass-card {
        background: #ffffff;
        border: 2px solid #e0f2fe;
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.08);
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0284c7, transparent);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(2, 132, 199, 0.15);
        border-color: #0284c7;
    }
    
    /* Form title */
    .section-title {
        font-size: 1.6em;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Custom HTML title inside glass-card */
    .glass-card h3 {
        font-size: 1.4em;
        font-weight: 700;
        color: #0284c7;
        margin: 0 0 16px 0 !important;
        padding: 0 !important;
    }
    
    .section-title .icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #0284c7, #0369a1);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2em;
        color: white;
    }
    
    /* Input styling - Light Theme */
    .stSelectbox > div > div {
        background: #f8fafc !important;
        border: 2px solid #e0f2fe !important;
        border-radius: 12px !important;
        color: #0f172a !important;
        transition: all 0.3s ease;
        min-height: 38px !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #0284c7 !important;
        box-shadow: 0 0 12px rgba(2, 132, 199, 0.2);
    }
    
    .stSelectbox > div > div > div {
        color: #0f172a !important;
    }
    
    .stNumberInput > div > div {
        background: #f8fafc !important;
        border: 2px solid #e0f2fe !important;
        border-radius: 12px !important;
        min-height: 38px !important;
    }
    
    .stNumberInput input {
        color: #0f172a !important;
        background: #f8fafc !important;
    }
    
    /* Label styling - Light Theme */
    .input-label {
        font-size: 0.95em;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .input-label .step {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #0284c7, #0369a1);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75em;
        font-weight: 700;
        color: white;
    }
    
    /* Button styling - Ocean Blue */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 32px !important;
        font-size: 1.05em !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 16px rgba(2, 132, 199, 0.3) !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 24px rgba(2, 132, 199, 0.4) !important;
    }
    
    /* Result card - Light Theme with Green */
    .result-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 2px solid #22c55e;
        border-radius: 24px;
        padding: 32px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: glow 3s ease-in-out infinite;
        margin-bottom: 16px;
        box-shadow: 0 8px 16px rgba(34, 197, 94, 0.15);
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 8px 16px rgba(34, 197, 94, 0.15); }
        50% { box-shadow: 0 12px 24px rgba(34, 197, 94, 0.25); }
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(34, 197, 94, 0.1) 0%, transparent 40%);
    }
    
    .result-label {
        font-size: 0.95em;
        color: #1e293b;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .result-price {
        font-size: 2.8em;
        font-weight: 900;
        color: #16a34a;
        margin: 12px 0;
        position: relative;
    }
    
    .result-subtext {
        font-size: 0.95em;
        color: #1e293b;
        font-weight: 500;
    }
    
    /* Stats cards - Light Theme */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    
    .stat-card {
        background: #f8fafc;
        border: 2px solid #e0f2fe;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: #f0f9ff;
        border-color: #0284c7;
    }
    
    .stat-number {
        font-size: 1.5em;
        font-weight: 800;
        color: #0284c7;
    }
    
    .stat-label {
        font-size: 0.75em;
        color: #1e293b;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* Info cards - Light Theme */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }
    
    .info-card {
        background: #f8fafc;
        border: 2px solid #e0f2fe;
        border-radius: 16px;
        padding: 14px;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: #f0f9ff;
        border-color: #0284c7;
        transform: translateY(-2px);
    }
    
    .info-label {
        font-size: 0.7em;
        color: #1e293b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .info-value {
        font-size: 0.95em;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Model info box - Light Theme */
    .model-info {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    
    .model-info-title {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        font-size: 1.05em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .model-info-text {
        font-size: 0.9em;
        color: #1e293b;
        line-height: 1.8;
    }
    
    .highlight {
        color: #0284c7;
        font-weight: 600;
    }
    
    /* Divider - Light Theme */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0f2fe, transparent);
        margin: 16px 0;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stats-grid, .info-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.5em;
        }
        .result-price {
            font-size: 2.2em;
        }
    }
    
    /* Animation delays */
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }
    
    /* Căn chỉnh chiều dọc 2 cột */
    [data-testid="column"] {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    
    [data-testid="column"] > div {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    
    h3 {
        margin-top: 0px !important;
        padding-top: 0px !important;
        margin-bottom: 16px !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    .glass-card {
        margin-top: 0px !important;
        padding-top: 24px !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    button, input, select, textarea {
        font-family: 'Outfit', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HÀM LOAD MÔ HÌNH VÀ ENCODER
# ============================================================================
@st.cache_resource(ttl=3600)  # Cache 1 giờ
def load_model_and_encoders():
    """Load mô hình và encoders từ file .pkl"""
    try:
        model = joblib.load('model.pkl')
        encoders = joblib.load('encoders.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        training_info = joblib.load('training_info.pkl')
        df_processed = joblib.load('processed_data_raw.pkl')
        return model, encoders, feature_columns, training_info, df_processed
    except FileNotFoundError as e:
        st.error(f"❌ Lỗi: Không tìm thấy file mô hình! ({str(e)})")
        st.error("Vui lòng chạy 'python train_model.py' trước tiên.")
        st.stop()

# ============================================================================
# HÀM LẤY DANH SÁCH
# ============================================================================
def filter_dataframe(df, conditions):
    """
    Lọc dataframe theo các điều kiện đã chọn
    
    Args:
        df: DataFrame gốc
        conditions: Dictionary chứa các điều kiện lọc
                   VD: {'Hãng xe': 'Toyota', 'Dòng xe': 'Camry'}
    
    Returns:
        DataFrame đã được lọc
    """
    filtered_df = df.copy()
    for col, value in conditions.items():
        if value is not None and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] == value]
    return filtered_df

def get_unique_values_filtered(df, col, conditions):
    """
    Lấy danh sách giá trị duy nhất của một cột sau khi lọc theo các điều kiện
    
    Args:
        df: DataFrame gốc
        col: Tên cột cần lấy giá trị
        conditions: Dictionary chứa các điều kiện lọc
    
    Returns:
        List các giá trị duy nhất đã sắp xếp
    """
    filtered_df = filter_dataframe(df, conditions)
    values = [c for c in filtered_df[col].unique().tolist() if pd.notna(c)]
    return sorted(values)

def get_models_by_brand(df, brand):
    models = df[df['Hãng xe'] == brand]['Dòng xe'].unique().tolist()
    models = [m for m in models if pd.notna(m)]
    return sorted(models)

def get_unique_values(df, col):
    values = [c for c in df[col].unique().tolist() if pd.notna(c)]
    return sorted(values)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Load dữ liệu
    model, encoders, feature_columns, training_info, df_processed = load_model_and_encoders()
    
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🚗 Dự Đoán Giá Xe Ô Tô Cũ</h1>
            <p class="hero-subtitle">Machine Learning dự đoán giá xe chính xác đến 90%</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout chính
    col_left, col_right = st.columns([1.1, 0.9], gap="large")
    
    # ========== CỘT TRÁI: FORM NHẬP LIỆU ==========
    with col_left:
        st.markdown("""
            <div class="glass-card">
                <h3>📝 Nhập thông tin xe</h3>
        """, unsafe_allow_html=True)
        
        # 1. Hãng xe
        st.markdown('<div class="input-label"><span class="step">1</span>Hãng xe</div>', unsafe_allow_html=True)
        brands = get_unique_values(df_processed, 'Hãng xe')
        selected_brand = st.selectbox(
            "Chọn hãng xe",
            brands,
            key="brand",
            label_visibility="collapsed"
        )
        
        # 2. Dòng xe (lọc theo Hãng xe)
        st.markdown('<div class="input-label"><span class="step">2</span>Dòng xe</div>', unsafe_allow_html=True)
        models = get_unique_values_filtered(
            df_processed, 
            'Dòng xe', 
            {'Hãng xe': selected_brand}
        )
        selected_model = st.selectbox(
            "Chọn dòng xe",
            models,
            key="model",
            label_visibility="collapsed"
        )
        
        # 3. Đời xe (lọc theo Hãng xe + Dòng xe)
        st.markdown('<div class="input-label"><span class="step">3</span>Đời xe (Năm sản xuất)</div>', unsafe_allow_html=True)
        years = get_unique_values_filtered(
            df_processed,
            'Đời xe',
            {'Hãng xe': selected_brand, 'Dòng xe': selected_model}
        )
        selected_year = st.selectbox(
            "Chọn đời xe",
            years,
            key="year",
            label_visibility="collapsed"
        )
        
        # 4. Phiên bản (lọc theo Hãng xe + Dòng xe + Đời xe)
        st.markdown('<div class="input-label"><span class="step">4</span>Phiên bản</div>', unsafe_allow_html=True)
        versions = get_unique_values_filtered(
            df_processed,
            'Phiên bản',
            {'Hãng xe': selected_brand, 'Dòng xe': selected_model, 'Đời xe': selected_year}
        )
        selected_version = st.selectbox(
            "Chọn phiên bản",
            versions,
            key="version",
            label_visibility="collapsed"
        )
        
        # 5. Màu xe (lọc theo Hãng xe + Dòng xe + Đời xe + Phiên bản)
        st.markdown('<div class="input-label"><span class="step">5</span>Màu xe</div>', unsafe_allow_html=True)
        colors = get_unique_values_filtered(
            df_processed,
            'Màu xe',
            {
                'Hãng xe': selected_brand,
                'Dòng xe': selected_model,
                'Đời xe': selected_year,
                'Phiên bản': selected_version
            }
        )
        selected_color = st.selectbox(
            "Chọn màu xe",
            colors,
            key="color",
            label_visibility="collapsed"
        )
        
        # 6. Nhập khẩu
        st.markdown('<div class="input-label"><span class="step">6</span>Xuất xứ / Nhập khẩu</div>', unsafe_allow_html=True)
        imports = get_unique_values(df_processed, 'Nhập khẩu')
        selected_import = st.selectbox(
            "Chọn xuất xứ",
            imports,
            key="import",
            label_visibility="collapsed"
        )
        
        # 7. Tình trạng
        st.markdown('<div class="input-label"><span class="step">7</span>Tình trạng</div>', unsafe_allow_html=True)
        conditions = get_unique_values(df_processed, 'Tình trạng')
        
        # LỌC BỎ "Mới": Chỉ giữ "Tốt", "Rất tốt", "Trung bình"
        if 'Mới' in conditions:
            conditions.remove('Mới')
            
        selected_condition = st.selectbox(
            "Chọn tình trạng",
            conditions,
            key="condition",
            label_visibility="collapsed"
        )
        
        # 8. Công tơ mét
        st.markdown('<div class="input-label"><span class="step">8</span>Công tơ mét (km)</div>', unsafe_allow_html=True)
        mileage = st.number_input(
            "Nhập số km đã chạy",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            label_visibility="collapsed"
        )
        
        # Tính tuổi xe tự động
        current_year = 2026
        car_age = current_year - selected_year
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== CỘT PHẢI: NÚT DỰ ĐOÁN + KẾT QUẢ ==========
    with col_right:
        st.markdown("""
            <div class="glass-card">
                <h3>🎯 Kết quả dự đoán</h3>
        """, unsafe_allow_html=True)
        
        # NÚT DỰ ĐOÁN
        if st.button("Dự đoán ngay", use_container_width=True, type="primary"):
            # Chuẩn bị dữ liệu
            input_data = {
                'Hãng xe': selected_brand,
                'Dòng xe': selected_model,
                'Đời xe': selected_year,
                'Phiên bản': selected_version,
                'Màu xe': selected_color,
                'Nhập khẩu': selected_import,
                'Tình trạng': selected_condition,
                'Công tơ mét (km)': mileage,
                'Tuổi xe': car_age
            }
            
            # Mã hóa
            encoded_data = {}
            
            # Mapping thủ công cho Tình trạng
            mapping_tinh_trang = {'Trung bình': 1, 'Tốt': 2, 'Rất tốt': 3, 'Mới': 4}
            
            for col, value in input_data.items():
                if col == 'Tình trạng':
                    # Sử dụng mapping thủ công cho Tình trạng
                    encoded_data[col] = mapping_tinh_trang.get(value, 1)
                elif col in encoders and col != 'mapping_tinh_trang':
                    # Sử dụng LabelEncoder cho các cột khác
                    encoded_data[col] = encoders[col].transform([value])[0]
                else:
                    # Giữ nguyên giá trị số
                    encoded_data[col] = value
            
            X_input = pd.DataFrame([encoded_data])[feature_columns]
            predicted_price = model.predict(X_input)[0]
            predicted_price_rounded = round(predicted_price / 1_000_000) * 1_000_000
            
            # Hiển thị kết quả
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-price">{predicted_price_rounded:,.0f} VNĐ</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Chi tiết
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Thông tin xe dạng text để dễ copy
            car_info_text = f"{selected_brand} {selected_model} {selected_version} {selected_year}"
            st.markdown(f"""
                <div style="background: #f8fafc; border: 2px solid #e0f2fe; border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; font-family: 'Courier New', monospace; font-size: 0.95em; color: #0f172a; word-break: break-all; cursor: pointer;" title="Click để copy">
                    {car_info_text}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                <div style="font-size: 1.1em; font-weight: 700; color: #0f172a; margin-bottom: 12px;">
                    📋 Chi tiết xe
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">Hãng xe</div>
                        <div class="info-value">{selected_brand}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 8px;">
                        <div class="info-label">Dòng xe</div>
                        <div class="info-value">{selected_model}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 8px;">
                        <div class="info-label">Đời xe</div>
                        <div class="info-value">{selected_year}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">Phiên bản</div>
                        <div class="info-value">{selected_version}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 8px;">
                        <div class="info-label">Màu xe</div>
                        <div class="info-value">{selected_color}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 8px;">
                        <div class="info-label">Xuất xứ</div>
                        <div class="info-value">{selected_import}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">Tình trạng</div>
                        <div class="info-value">{selected_condition}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 8px;">
                        <div class="info-label">Công tơ mét</div>
                        <div class="info-value">{mileage:,} km</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 8px;">
                        <div class="info-label">Tuổi xe</div>
                        <div class="info-value">{car_age} năm</div>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== FOOTER: THÔNG TIN DỮ LIỆU ==========
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_data1, col_data2, col_data3 = st.columns(3)
    with col_data1:
        st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Tổng dữ liệu</div>
                <div class="info-value">{training_info['total_data']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_data2:
        st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Dữ liệu Train</div>
                <div class="info-value">{training_info['train_size']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_data3:
        st.markdown(f"""
            <div class="info-card">
                <div class="info-label">Dữ liệu Test</div>
                <div class="info-value">{training_info['test_size']:,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # So sánh 3 mô hình
    st.markdown("""
        <div style="font-size: 0.95em; color: #0f172a; font-weight: 600; margin-top: 20px; margin-bottom: 12px;">⚖️ So sánh 3 mô hình:</div>
    """, unsafe_allow_html=True)
    
    all_models = training_info.get('all_models', {})
    
    # Hiển thị bảng so sánh
    col_m1, col_m2, col_m3 = st.columns(3)
    
    # Linear Regression
    with col_m1:
        lr_data = all_models.get('Linear Regression', {})
        st.markdown(f"""
            <div style="background: #f8fafc; border: 2px solid #e0f2fe; border-radius: 12px; padding: 12px;">
                <div style="font-size: 0.85em; color: #0284c7; font-weight: 600; margin-bottom: 8px; text-align: center;">Linear Regression</div>
                <div style="font-size: 0.8em; color: #1e293b; line-height: 1.8;">
                    <div>R²: <span style="color: #0f172a; font-weight: 600;">{lr_data.get('r2', 0):.4f}</span></div>
                    <div>MAE: <span style="color: #0f172a; font-weight: 600;">{lr_data.get('mae', 0)/1_000_000:.0f}M</span></div>
                    <div>MAPE: <span style="color: #0f172a; font-weight: 600;">{lr_data.get('mape', 0):.2f}%</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Random Forest
    with col_m2:
        rf_data = all_models.get('Random Forest', {})
        st.markdown(f"""
            <div style="background: #f8fafc; border: 2px solid #e0f2fe; border-radius: 12px; padding: 12px;">
                <div style="font-size: 0.85em; color: #0284c7; font-weight: 600; margin-bottom: 8px; text-align: center;">Random Forest</div>
                <div style="font-size: 0.8em; color: #1e293b; line-height: 1.8;">
                    <div>R²: <span style="color: #0f172a; font-weight: 600;">{rf_data.get('r2', 0):.4f}</span></div>
                    <div>MAE: <span style="color: #0f172a; font-weight: 600;">{rf_data.get('mae', 0)/1_000_000:.0f}M</span></div>
                    <div>MAPE: <span style="color: #0f172a; font-weight: 600;">{rf_data.get('mape', 0):.2f}%</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # XGBoost
    with col_m3:
        xgb_data = all_models.get('XGBoost', {})
        st.markdown(f"""
            <div style="background: #f8fafc; border: 2px solid #e0f2fe; border-radius: 12px; padding: 12px;">
                <div style="font-size: 0.85em; color: #0284c7; font-weight: 600; margin-bottom: 8px; text-align: center;">🏆 XGBoost</div>
                <div style="font-size: 0.8em; color: #1e293b; line-height: 1.8;">
                    <div>R²: <span style="color: #0f172a; font-weight: 600;">{xgb_data.get('r2', 0):.4f}</span></div>
                    <div>MAE: <span style="color: #0f172a; font-weight: 600;">{xgb_data.get('mae', 0)/1_000_000:.0f}M</span></div>
                    <div>MAPE: <span style="color: #0f172a; font-weight: 600;">{xgb_data.get('mape', 0):.2f}%</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
