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
# CSS TÙYCHỈNH - PHONG CÁCH HIỆN ĐẠI PREMIUM
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    /* Header section */
    .hero-section {
        text-align: center;
        padding: 60px 20px 40px;
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
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 50%);
        animation: float 15s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(20px, 20px) rotate(5deg); }
    }
    
    .hero-title {
        font-size: 3.8em;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
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
        font-size: 1.25em;
        color: #94a3b8;
        margin-top: 16px;
        font-weight: 400;
        letter-spacing: 0.5px;
        position: relative;
        animation: slideUp 0.8s ease-out 0.1s both;
    }
    
    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(129, 140, 248, 0.3);
    }
    
    /* Form title */
    .section-title {
        font-size: 1.6em;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .section-title .icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2em;
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #818cf8 !important;
        box-shadow: 0 0 20px rgba(129, 140, 248, 0.2);
    }
    
    .stSelectbox > div > div > div {
        color: #f1f5f9 !important;
    }
    
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    .stNumberInput input {
        color: #f1f5f9 !important;
    }
    
    /* Label styling */
    .input-label {
        font-size: 0.95em;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .input-label .step {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8em;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 18px 40px !important;
        font-size: 1.15em !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4) !important;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 16px 48px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 2px solid rgba(129, 140, 248, 0.4);
        backdrop-filter: blur(20px);
        padding: 50px 40px;
        border-radius: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 30px rgba(129, 140, 248, 0.2); }
        50% { box-shadow: 0 0 50px rgba(129, 140, 248, 0.4); }
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(129, 140, 248, 0.1) 0%, transparent 40%);
    }
    
    .result-label {
        font-size: 1.1em;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 16px;
    }
    
    .result-price {
        font-size: 4em;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 20px 0;
        position: relative;
    }
    
    .result-subtext {
        font-size: 1.1em;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(129, 140, 248, 0.3);
    }
    
    .stat-number {
        font-size: 1.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.85em;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Info cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(129, 140, 248, 0.2);
        transform: translateY(-2px);
    }
    
    .info-label {
        font-size: 0.8em;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .info-value {
        font-size: 1.15em;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    /* Model info box */
    .model-info {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        border-left: 4px solid #818cf8;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    
    .model-info-title {
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 12px;
        font-size: 1.05em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .model-info-text {
        font-size: 0.9em;
        color: #94a3b8;
        line-height: 1.8;
    }
    
    .highlight {
        color: #818cf8;
        font-weight: 600;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 30px 0;
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
            font-size: 2.8em;
        }
    }
    
    /* Animation delays */
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HÀM LOAD MÔ HÌNH VÀ ENCODER
# ============================================================================
@st.cache_resource
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
                <div class="section-title">
                    <span class="icon">📝</span>
                    Thông tin xe của bạn
                </div>
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
    
    # ========== CỘT PHẢI: THÔNG TIN MÔ HÌNH ==========
    with col_right:
        st.markdown("""
            <div class="glass-card">
                <div class="section-title">
                    <span class="icon">🤖</span>
                    Thông tin mô hình
                </div>
        """, unsafe_allow_html=True)
        
        # Stats
        st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{training_info['total_data']:,}</div>
                    <div class="stat-label">Tổng dữ liệu</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{training_info['train_size']:,}</div>
                    <div class="stat-label">Training</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{training_info['test_size']:,}</div>
                    <div class="stat-label">Testing</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        

        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== NÚT DỰ ĐOÁN ==========
    st.markdown('<br>', unsafe_allow_html=True)
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        if st.button("🔮 Dự đoán ngay", use_container_width=True, type="primary"):
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
                    <div class="result-label">💰 Giá dự đoán của xe bạn:</div>
                    <div class="result-price">{predicted_price_rounded:,.0f} ₫</div>
                    <div class="result-subtext">{predicted_price_rounded/1_000_000:.0f} triệu đồng</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Chi tiết
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            st.markdown("""
                <div style="font-size: 1.3em; font-weight: 700; color: #f1f5f9; margin-bottom: 20px;">
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
                    <div class="info-card" style="margin-top: 12px;">
                        <div class="info-label">Dòng xe</div>
                        <div class="info-value">{selected_model}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 12px;">
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
                    <div class="info-card" style="margin-top: 12px;">
                        <div class="info-label">Màu xe</div>
                        <div class="info-value">{selected_color}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 12px;">
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
                    <div class="info-card" style="margin-top: 12px;">
                        <div class="info-label">Công tơ mét</div>
                        <div class="info-value">{mileage:,} km</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 12px;">
                        <div class="info-label">Tuổi xe</div>
                        <div class="info-value">{car_age} năm</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<br>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
