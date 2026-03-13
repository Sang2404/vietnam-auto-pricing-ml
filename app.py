"""
Ứng dụng Web dự đoán giá xe ô tô cũ sử dụng Streamlit
Giao diện sang trọng, sáng sửa với phong cách premium
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
# CSS TÙYCHỈNH - PHONG CÁCH SANG TRỌNG, SÁNG SỬA
# ============================================================================
st.markdown("""
    <style>
    /* Nền chính - Trắng sạch */
    .stApp {
        background: #ffffff;
    }
    
    /* Tiêu đề chính */
    .main-header {
        text-align: center;
        padding: 50px 20px;
        background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%);
        border-bottom: 3px solid #0066cc;
        margin-bottom: 40px;
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: 900;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .subtitle {
        font-size: 1.15em;
        color: #666;
        margin-top: 15px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Container chính */
    .main-container {
        max-width: 1300px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Card form */
    .form-card {
        background: white;
        padding: 45px;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 30px;
    }
    
    /* Tiêu đề form */
    .form-title {
        font-size: 1.8em;
        font-weight: 800;
        color: #333;
        margin-bottom: 35px;
        display: flex;
        align-items: center;
        gap: 12px;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 15px;
    }
    
    /* Nhóm input */
    .input-group {
        display: flex;
        flex-direction: column;
        margin-bottom: 20px;
    }
    
    .input-label {
        font-size: 1.05em;
        font-weight: 700;
        color: #333;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: 0.3px;
    }
    
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border-radius: 50%;
        font-weight: bold;
        font-size: 0.95em;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.3);
    }
    
    /* Grid form */
    .form-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 30px;
        margin-bottom: 30px;
    }
    
    @media (max-width: 768px) {
        .form-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Nút dự đoán */
    .predict-btn {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        padding: 20px 50px;
        border: none;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: 700;
        cursor: pointer;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3);
        letter-spacing: 0.5px;
    }
    
    .predict-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 102, 204, 0.4);
    }
    
    /* Container kết quả */
    .result-card {
        background: linear-gradient(135deg, #f0f7ff 0%, #ffffff 100%);
        border: 2px solid #0066cc;
        color: #333;
        padding: 60px 40px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 102, 204, 0.15);
        margin-bottom: 30px;
    }
    
    .result-label {
        font-size: 1.2em;
        color: #666;
        font-weight: 600;
        margin-bottom: 20px;
        letter-spacing: 0.5px;
    }
    
    .result-price {
        font-size: 4.5em;
        font-weight: 900;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 20px 0;
        letter-spacing: 1px;
    }
    
    .result-subtext {
        font-size: 1.15em;
        color: #999;
        font-weight: 600;
    }
    
    /* Info cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 30px;
    }
    
    @media (max-width: 768px) {
        .info-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.15);
        transform: translateY(-2px);
    }
    
    .info-label {
        font-size: 0.9em;
        color: #999;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    
    .info-value {
        font-size: 1.35em;
        font-weight: 800;
        color: #333;
    }
    
    /* Stats section */
    .stats-section {
        background: white;
        padding: 35px;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 30px;
    }
    
    .stats-title {
        font-size: 1.5em;
        font-weight: 800;
        color: #333;
        margin-bottom: 25px;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 12px;
    }
    
    .stat-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-bottom: 20px;
    }
    
    @media (max-width: 768px) {
        .stat-row {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    .stat-item {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #fafafa 0%, #ffffff 100%);
        border-radius: 10px;
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        border-color: #0066cc;
        box-shadow: 0 2px 10px rgba(0, 102, 204, 0.1);
    }
    
    .stat-number {
        font-size: 2em;
        font-weight: 900;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.9em;
        color: #999;
        margin-top: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    /* Config box */
    .config-box {
        background: linear-gradient(135deg, #fafafa 0%, #ffffff 100%);
        border-left: 4px solid #0066cc;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    .config-title {
        font-weight: 700;
        color: #333;
        margin-bottom: 12px;
        font-size: 1.05em;
    }
    
    .config-text {
        font-size: 0.95em;
        color: #666;
        line-height: 1.8;
    }
    
    .config-highlight {
        color: #0066cc;
        font-weight: 700;
    }
    
    /* Streamlit overrides */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    .stNumberInput > div > div {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
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
def get_models_by_brand(df, brand):
    models = df[df['Hãng xe'] == brand]['Dòng xe'].unique().tolist()
    models = [m for m in models if pd.notna(m)]
    return sorted(models)

def get_versions_by_model(df, brand, model):
    versions = df[(df['Hãng xe'] == brand) & (df['Dòng xe'] == model)]['Phiên bản'].unique().tolist()
    versions = [v for v in versions if pd.notna(v)]
    return sorted(versions)

def get_years(df):
    years = df['Đời xe'].unique().tolist()
    years = [y for y in years if pd.notna(y)]
    return sorted(years)

def get_colors(df):
    colors = [c for c in df['Màu xe'].unique().tolist() if pd.notna(c)]
    return sorted(colors)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Load dữ liệu
    model, encoders, feature_columns, training_info, df_processed = load_model_and_encoders()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🚗 Dự Đoán Giá Xe Ô Tô Cũ</h1>
            <p class="subtitle">Ước tính giá trị hiện tại của xe của bạn một cách chính xác</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout chính
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # ========== CỘT TRÁI: FORM NHẬP LIỆU ==========
    with col_left:
        st.markdown("""
            <div class="form-card">
                <div class="form-title">📝 Thông tin xe của bạn</div>
        """, unsafe_allow_html=True)
        
        # Hãng xe
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown('<div class="input-label"><span class="step-badge">1</span>Hãng xe</div>', unsafe_allow_html=True)
        brands = sorted(df_processed['Hãng xe'].unique().tolist())
        selected_brand = st.selectbox(
            "Chọn hãng xe",
            brands,
            key="brand",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dòng xe
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown('<div class="input-label"><span class="step-badge">2</span>Dòng xe</div>', unsafe_allow_html=True)
        models = get_models_by_brand(df_processed, selected_brand)
        selected_model = st.selectbox(
            "Chọn dòng xe",
            models,
            key="model",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Đời xe
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown('<div class="input-label"><span class="step-badge">3</span>Đời xe</div>', unsafe_allow_html=True)
        years = get_years(df_processed)
        selected_year = st.selectbox(
            "Chọn đời xe",
            years,
            key="year",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Phiên bản
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown('<div class="input-label"><span class="step-badge">4</span>Phiên bản</div>', unsafe_allow_html=True)
        versions = get_versions_by_model(df_processed, selected_brand, selected_model)
        selected_version = st.selectbox(
            "Chọn phiên bản",
            versions,
            key="version",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Màu xe
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown('<div class="input-label"><span class="step-badge">5</span>Màu xe</div>', unsafe_allow_html=True)
        colors = get_colors(df_processed)
        selected_color = st.selectbox(
            "Chọn màu xe",
            colors,
            key="color",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Công tơ mét
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        st.markdown('<div class="input-label"><span class="step-badge">6</span>Công tơ mét (km)</div>', unsafe_allow_html=True)
        mileage = st.number_input(
            "Nhập số km",
            min_value=0,
            max_value=500000,
            value=50000,
            step=1000,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== CỘT PHẢI: THÔNG TIN & KẾT QUẢ ==========
    with col_right:
        # Thông tin mô hình
        st.markdown("""
            <div class="stats-section">
                <div class="stats-title">📊 Thông tin mô hình</div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{training_info['total_data']:,}</div>
                    <div class="stat-label">Tổng dữ liệu</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{training_info['train_size']:,}</div>
                    <div class="stat-label">Train (80%)</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-number">{training_info['test_size']:,}</div>
                    <div class="stat-label">Test (20%)</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="config-box">
                <div class="config-title">🤖 Cấu hình mô hình</div>
                <div class="config-text">
                    • Thuật toán: <span class="config-highlight">Random Forest Regressor</span><br>
                    • Số cây: <span class="config-highlight">100</span> | Độ sâu: <span class="config-highlight">20</span><br>
                    • Test R²: <span class="config-highlight">{training_info['test_r2']:.4f}</span><br>
                    • Test MAPE: <span class="config-highlight">{training_info['mape']:.2f}%</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== NÚT DỰ ĐOÁN (FULL WIDTH) ==========
    st.markdown('<br>', unsafe_allow_html=True)
    
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        if st.button("🔮 Dự đoán giá", use_container_width=True, type="primary"):
            # Chuẩn bị dữ liệu
            input_data = {
                'Hãng xe': selected_brand,
                'Dòng xe': selected_model,
                'Đời xe': selected_year,
                'Phiên bản': selected_version,
                'Màu xe': selected_color,
                'Công tơ mét (km)': mileage
            }
            
            # Mã hóa
            encoded_data = {}
            for col, value in input_data.items():
                if col in encoders:
                    encoded_data[col] = encoders[col].transform([value])[0]
                else:
                    encoded_data[col] = value
            
            X_input = pd.DataFrame([encoded_data])[feature_columns]
            predicted_price = model.predict(X_input)[0]
            predicted_price_rounded = round(predicted_price / 1_000_000) * 1_000_000
            
            # Hiển thị kết quả
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">💰 Giá dự đoán của xe bạn:</div>
                    <div class="result-price">{predicted_price_rounded:,.0f}</div>
                    <div class="result-subtext">({predicted_price_rounded/1_000_000:.0f} triệu đồng)</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Chi tiết dự đoán
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("""
                <div style="font-size: 1.3em; font-weight: 800; color: #333; margin-bottom: 25px;">
                    📋 Chi tiết dự đoán
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
                    <div class="info-card" style="margin-top: 15px;">
                        <div class="info-label">Dòng xe</div>
                        <div class="info-value">{selected_model}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">Đời xe</div>
                        <div class="info-value">{selected_year}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 15px;">
                        <div class="info-label">Màu xe</div>
                        <div class="info-value">{selected_color}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">Phiên bản</div>
                        <div class="info-value" style="font-size: 1em;">{selected_version}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="info-card" style="margin-top: 15px;">
                        <div class="info-label">Công tơ mét</div>
                        <div class="info-value">{mileage:,} km</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<br>', unsafe_allow_html=True)
            st.success(f"✅ Giá dự đoán (chính xác): **{predicted_price:,.0f} VNĐ**")

if __name__ == "__main__":
    main()
