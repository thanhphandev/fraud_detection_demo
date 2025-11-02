"""
Ứng dụng Demo Phát hiện Gian lận Thẻ Tín dụng bằng Machine Learning

Ứng dụng này sử dụng Streamlit để tạo giao diện tương tác cho phép người dùng:
- Chọn phương pháp xử lý dữ liệu mất cân bằng (Original, Oversampling, SMOTE)
- Chọn một hoặc nhiều mô hình ML để huấn luyện và so sánh
- Xem kết quả chi tiết và so sánh hiệu suất các mô hình

Phiên bản Python: 3.11
"""

import sys
import os

# Thêm thư mục src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import numpy as np

# Import các module tự tạo
from src.data_processing import (
    load_data, 
    get_data_info, 
    prepare_data, 
    process_data_by_method
)
from src.models import train_and_evaluate_models, get_best_model
from src.visualization import (
    plot_confusion_matrix,
    create_metrics_dataframe,
    display_metrics_summary,
    plot_comparison_chart,
    display_data_info,
    get_recommendation,
    create_detailed_metrics_table
)


# ==================== Cấu hình trang ====================
st.set_page_config(
    page_title="Phát hiện Gian lận Thẻ Tín dụng",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CSS tùy chỉnh ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== Header ====================
st.markdown('<div class="main-header">Phát hiện Gian lận Thẻ Tín dụng bằng Machine Learning</div>', 
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Demo ứng dụng học máy để phát hiện gian lận và so sánh hiệu quả các mô hình</div>', 
            unsafe_allow_html=True)


# ==================== Sidebar - Bảng điều khiển ====================
st.sidebar.title("Bảng điều khiển")
st.sidebar.markdown("---")

# Bước 1: Chọn phương pháp xử lý dữ liệu
st.sidebar.subheader("Bước 1: Chọn phương pháp xử lý dữ liệu")
data_processing_method = st.sidebar.selectbox(
    "Phương pháp xử lý dữ liệu mất cân bằng:",
    [
        'Dữ liệu gốc (Mất cân bằng)',
        'Xử lý bằng Oversampling',
        'Xử lý bằng SMOTE'
    ],
    help="Chọn phương pháp để xử lý dữ liệu mất cân bằng trong tập huấn luyện"
)

# Hiển thị mô tả phương pháp
method_descriptions = {
    'Dữ liệu gốc (Mất cân bằng)': """
    **Dữ liệu gốc (Imbalanced Data)**
    
    Sử dụng dữ liệu gốc không qua xử lý. Tập dữ liệu có tỷ lệ gian lận chỉ 0.172%, 
    rất mất cân bằng giữa lớp gian lận và lớp hợp pháp.
    
    ⚠️ Có thể dẫn đến mô hình thiên vị về lớp đa số (hợp pháp).
    """,
    'Xử lý bằng Oversampling': """
    **Random Oversampling**
    
    Lặp lại ngẫu nhiên các mẫu gian lận (minority class) để cân bằng với lớp đa số.
    
    ✅ Đơn giản và nhanh chóng.
    ⚠️ Có thể gây overfitting do lặp lại chính xác các mẫu.
    """,
    'Xử lý bằng SMOTE': """
    **SMOTE (Synthetic Minority Oversampling Technique)**
    
    Tạo ra các mẫu gian lận tổng hợp bằng cách nội suy giữa các mẫu gian lận hiện có.
    
    ✅ Tạo ra dữ liệu đa dạng hơn, giảm overfitting.
    ✅ Thường cho kết quả tốt hơn Random Oversampling.
    """
}

st.sidebar.info(method_descriptions[data_processing_method])

st.sidebar.markdown("---")

# Bước 2: Chọn mô hình
st.sidebar.subheader("Bước 2: Chọn mô hình học máy")
selected_models = st.sidebar.multiselect(
    "Chọn một hoặc nhiều mô hình để huấn luyện:",
    [
        'Hồi quy Logistic',
        'Cây quyết định',
        'Mạng Bayesian',
    ],
    default=['Hồi quy Logistic', 'Cây quyết định'],
    help="Chọn các mô hình bạn muốn huấn luyện và so sánh"
)

# Hiển thị thông tin về mô hình
if selected_models:
    with st.sidebar.expander("Thông tin về các mô hình đã chọn"):
        for model in selected_models:
            if model == 'Hồi quy Logistic':
                st.markdown("**Hồi quy Logistic:** Mô hình tuyến tính sử dụng hàm sigmoid.")
            elif model == 'Cây quyết định':
                st.markdown("**Cây quyết định:** Mô hình phi tuyến tạo cấu trúc cây.")
            elif model == 'Mạng Bayesian':
                st.markdown("**Mạng Bayesian:** Sử dụng định lý Bayes (Gaussian NB).")

st.sidebar.markdown("---")

# Bước 3: Nút thực thi
st.sidebar.subheader("Bước 3: Thực thi")
run_button = st.sidebar.button("Huấn luyện và Đánh giá", type="primary")

# Thông tin bổ sung
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Hướng dẫn sử dụng:
1. Chọn phương pháp xử lý dữ liệu
2. Chọn một hoặc nhiều mô hình
3. Nhấn nút "Huấn luyện và Đánh giá"
4. Xem kết quả và so sánh

### Dataset:
- **Nguồn:** Kaggle Credit Card Fraud Detection
- **Tổng giao dịch:** 284,807
- **Giao dịch gian lận:** 492 (0.172%)
""")


# ==================== Main Area ====================

# Hiển thị thông báo nếu chưa chọn mô hình
if not selected_models:
    st.warning("⚠️ Vui lòng chọn ít nhất một mô hình ở thanh bên để bắt đầu!")
    st.stop()

# Khi người dùng nhấn nút
if run_button:
    with st.spinner("🔄 Đang tải và xử lý dữ liệu..."):
        # Tải dữ liệu
        df = load_data()
        
        if df is None:
            st.error("❌ Không thể tải dữ liệu. Vui lòng kiểm tra kết nối internet hoặc tải dữ liệu thủ công.")
            st.stop()
        
        # Lấy thông tin dữ liệu gốc
        data_info = get_data_info(df)
        
        # Chuẩn bị dữ liệu (chia train/test) (80/20)
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Xử lý dữ liệu theo phương pháp đã chọn
        X_train_processed, y_train_processed, method_info = process_data_by_method(
            data_processing_method, X_train, y_train
        )
    
    # Hiển thị thông tin dữ liệu
    st.success("Dữ liệu đã được tải và xử lý thành công!")
    display_data_info(data_info, method_info)
    
    st.markdown("---")
    
    # Huấn luyện các mô hình
    st.subheader("Huấn luyện và Đánh giá các Mô hình")
    
    trained_models = train_and_evaluate_models(
        selected_models,
        X_train_processed,
        y_train_processed,
        X_test,
        y_test
    )
    
    st.success(f"Đã hoàn thành huấn luyện {len(trained_models)} mô hình!")
    
    st.markdown("---")
    
    # Hiển thị kết quả chi tiết cho từng mô hình
    st.subheader("Kết quả Chi tiết từng Mô hình")
    
    for i, model in enumerate(trained_models):
        with st.expander(f"{model.model_name}", expanded=(i == 0)):
            # Hiển thị metrics summary
            display_metrics_summary(model)
            
            st.markdown("---")
            
            # Tạo 2 cột: confusion matrix và chi tiết
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Confusion Matrix:**")
                fig_cm = plot_confusion_matrix(
                    model.metrics['confusion_matrix'],
                    model.model_name
                )
                st.pyplot(fig_cm)
            
            with col2:
                st.markdown("**Chi tiết Metrics:**")
                metrics = model.metrics
                
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'True Negative (TN)',
                        'False Positive (FP)',
                        'False Negative (FN)',
                        'True Positive (TP)',
                        'Accuracy',
                        'Precision',
                        'Recall',
                        'F1-Score'
                    ],
                    'Value': [
                        f"{metrics['true_negative']:,}",
                        f"{metrics['false_positive']:,}",
                        f"{metrics['false_negative']:,}",
                        f"{metrics['true_positive']:,}",
                        f"{metrics['accuracy']*100:.2f}%",
                        f"{metrics['precision']*100:.2f}%",
                        f"{metrics['recall']*100:.2f}%",
                        f"{metrics['f1_score']*100:.2f}%"
                    ]
                })
                
                st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Bảng so sánh tổng hợp
    st.subheader("Bảng So sánh Tổng hợp")
    
    comparison_df = create_detailed_metrics_table(trained_models)
    
    # Hiển thị bảng với styling
    st.dataframe(
        comparison_df,
        width='stretch',
        hide_index=True
    )
    
    # Thêm ghi chú
    st.caption("""
    **📌 Giải thích Metrics:**
    
    **Confusion Matrix:**
    - ✅ **TP**: Bắt đúng gian lận | ❌ **FP**: Báo nhầm (oan) | ✅ **TN**: Nhận đúng hợp pháp | ❌ **FN**: Bỏ sót gian lận
    
    **Chỉ số đánh giá:**
    - **Precision** = TP/(TP+FP) → Khi báo "gian lận", đúng bao nhiêu %? (↓ FP)
    - **Recall** = TP/(TP+FN) → Bắt được bao nhiêu % gian lận thực tế? (↓ FN)  
    - **F1-Score** → Cân bằng Precision & Recall (quan trọng nhất với dữ liệu mất cân bằng)
    - **Accuracy** → % dự đoán đúng tổng thể (⚠️ không tin cậy khi dữ liệu mất cân bằng)
    """)
    
    st.markdown("---")
    
    # Biểu đồ so sánh
    if len(trained_models) > 1:
        st.subheader("Biểu đồ So sánh")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_f1 = plot_comparison_chart(trained_models, 'f1_score')
            st.pyplot(fig_f1)
        
        with col2:
            fig_recall = plot_comparison_chart(trained_models, 'recall')
            st.pyplot(fig_recall)
        
        st.markdown("---")
    
    # Khuyến nghị
    get_recommendation(trained_models, data_processing_method)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p>Đồ án: Phát hiện Gian lận Thẻ Tín dụng bằng Machine Learning</p>
        <p>Dataset: Credit Card Fraud Detection - Kaggle</p>
        <p>Công nghệ: Python 3.11 | Streamlit | Scikit-learn | Imbalanced-learn</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Hiển thị trang chào mừng khi chưa nhấn nút
    st.info("Vui lòng cấu hình các tùy chọn ở thanh bên và nhấn nút **'Huấn luyện và Đánh giá'** để bắt đầu!")
    
    # Hiển thị thông tin về đồ án
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Mục tiêu
        Ứng dụng học máy để phát hiện gian lận thẻ tín dụng và so sánh hiệu quả 
        của các mô hình khác nhau khi xử lý dữ liệu mất cân bằng.
        """)
    
    with col2:
        st.markdown("""
        ### Phương pháp
        - **Xử lý dữ liệu:** Oversampling, SMOTE
        - **Mô hình:** Logistic Regression, Decision Tree, Bayesian Network
        - **Đánh giá:** Confusion Matrix, Accuracy, TP, FP
        """)
    
    with col3:
        st.markdown("""
        ### Dữ liệu
        - **Nguồn:** Kaggle
        - **Giao dịch:** 284,807
        - **Gian lận:** 492 (0.172%)
        - **Đặc trưng:** 30 features (PCA transformed)
        """)
    
    st.markdown("---")
    
    # Hình minh họa (placeholder)
    st.image("https://cdn.prod.website-files.com/65e3b115e37b4719332ddd06/65e3b115e37b4719332de329_Fraud%20Detection%20with%20Machine%20Learning%20and%20AI.png", 
             width='stretch',
             caption="Demo ứng dụng phát hiện gian lận thẻ tín dụng")
