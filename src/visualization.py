"""
Module trực quan hóa kết quả cho dự án phát hiện gian lận thẻ tín dụng.

Module này cung cấp các hàm để:
- Vẽ confusion matrix
- Tạo bảng so sánh các mô hình
- Trực quan hóa các metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st


def plot_confusion_matrix(cm, model_name, figsize=(8, 6)):
    """
    Vẽ confusion matrix dưới dạng heatmap.
    
    Args:
        cm (np.array): Confusion matrix (2x2)
        model_name (str): Tên mô hình
        figsize (tuple): Kích thước figure
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tạo heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar=True,
        square=True,
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    # Thiết lập labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    # Thiết lập tick labels
    ax.set_xticklabels(['Normal (0)', 'Fraud (1)'], fontsize=11)
    ax.set_yticklabels(['Normal (0)', 'Fraud (1)'], fontsize=11, rotation=0)
    
    # Thêm chú thích
    tn, fp, fn, tp = cm.ravel()
    textstr = f'TN: {tn}  |  FP: {fp}\nFN: {fn}  |  TP: {tp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig


def create_metrics_dataframe(models):
    """
    Tạo DataFrame chứa các metrics của tất cả các mô hình.
    
    Args:
        models (list): Danh sách các FraudDetectionModel đã được đánh giá
        
    Returns:
        pd.DataFrame: DataFrame chứa metrics
    """
    data = []
    
    for model in models:
        metrics = model.metrics
        data.append({
            'Tên mô hình': metrics['model_name'],
            'True Positive (TP)': metrics['true_positive'],
            'False Positive (FP)': metrics['false_positive'],
            'True Negative (TN)': metrics['true_negative'],
            'False Negative (FN)': metrics['false_negative'],
            'Accuracy (%)': f"{metrics['accuracy'] * 100:.2f}",
            'Precision (%)': f"{metrics['precision'] * 100:.2f}",
            'Recall (%)': f"{metrics['recall'] * 100:.2f}",
            'F1-Score (%)': f"{metrics['f1_score'] * 100:.2f}"
        })
    
    df = pd.DataFrame(data)
    return df


def display_metrics_summary(model):
    """
    Hiển thị tóm tắt metrics của một mô hình dưới dạng cards.
    
    Args:
        model: FraudDetectionModel đã được đánh giá
    """
    metrics = model.metrics
    
    # Tạo 4 cột cho các metrics chính
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="✅ True Positive (TP)",
            value=metrics['true_positive'],
            help="Số giao dịch gian lận được phát hiện đúng"
        )
    
    with col2:
        st.metric(
            label="❌ False Positive (FP)",
            value=metrics['false_positive'],
            help="Số giao dịch hợp pháp bị nhận diện nhầm là gian lận"
        )
    
    with col3:
        st.metric(
            label="Accuracy",
            value=f"{metrics['accuracy'] * 100:.2f}%",
            help="Độ chính xác tổng thể của mô hình"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value=f"{metrics['f1_score'] * 100:.2f}%",
            help="Điểm cân bằng giữa Precision và Recall"
        )


def plot_comparison_chart(models, metric='f1_score'):
    """
    Vẽ biểu đồ so sánh các mô hình theo một metric cụ thể.
    
    Args:
        models (list): Danh sách các FraudDetectionModel
        metric (str): Tên metric cần so sánh
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    model_names = [m.model_name for m in models]
    metric_values = [m.metrics[metric] * 100 for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tạo bar chart
    bars = ax.bar(model_names, metric_values, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Thêm giá trị lên mỗi bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Thiết lập labels và title
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score'
    }
    
    ax.set_ylabel(f'{metric_labels.get(metric, metric)} (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mô hình', fontsize=12, fontweight='bold')
    ax.set_title(f'So sánh {metric_labels.get(metric, metric)} giữa các mô hình', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Xoay labels trục x nếu cần
    plt.xticks(rotation=15, ha='right')
    
    # Thêm grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


def display_data_info(data_info, method_info):
    """
    Hiển thị thông tin về dữ liệu gốc và sau xử lý.
    
    Args:
        data_info (dict): Thông tin về dữ liệu gốc
        method_info (dict): Thông tin về phương pháp xử lý
    """
    st.subheader("Thông tin dữ liệu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dữ liệu gốc:**")
        st.write(f"- Tổng số giao dịch: {data_info['total_transactions']:,}")
        st.write(f"- Giao dịch hợp pháp: {data_info['normal_transactions']:,}")
        st.write(f"- Giao dịch gian lận: {data_info['fraud_transactions']:,}")
        st.write(f"- Tỷ lệ gian lận: {data_info['fraud_percentage']:.3f}%")
    
    with col2:
        st.markdown(f"**Sau xử lý ({method_info['method']}):**")
        st.write(f"- Tổng số mẫu huấn luyện: {method_info['resampled_total']:,}")
        st.write(f"- Mẫu hợp pháp: {method_info['resampled_normal']:,}")
        st.write(f"- Mẫu gian lận: {method_info['resampled_fraud']:,}")
        st.write(f"- Tỷ lệ gian lận: {method_info['resampled_fraud_percentage']:.2f}%")


def get_recommendation(models, method_name):
    """
    Hiển thị khuyến nghị trực quan dựa trên kết quả các mô hình.
    
    Args:
        models (list): Danh sách các FraudDetectionModel
        method_name (str): Tên phương pháp xử lý dữ liệu
    """
    if not models:
        st.warning("Không có mô hình nào được đánh giá.")
        return
    
    # Tìm các mô hình tốt nhất theo từng tiêu chí
    best_model = max(models, key=lambda m: m.metrics['f1_score'])
    best_tp_model = max(models, key=lambda m: m.metrics['true_positive'])
    best_fp_model = min(models, key=lambda m: m.metrics['false_positive'])
    
    st.subheader("📊 Phân tích & Khuyến nghị")
    
    # Hiển thị phương pháp xử lý
    st.info(f"**Phương pháp xử lý dữ liệu:** {method_name}")
    
    # Tạo 3 cột cho 3 tiêu chí đánh giá
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h4 style='margin: 0; color: white;'>🏆 Tổng thể tốt nhất</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>Dựa trên F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Mô hình:** {best_model.model_name}")
        st.metric("F1-Score", f"{best_model.metrics['f1_score']*100:.2f}%")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("TP", f"{best_model.metrics['true_positive']:,}", 
                     help="True Positive - Gian lận phát hiện đúng")
        with col_b:
            st.metric("FP", f"{best_model.metrics['false_positive']:,}",
                     help="False Positive - Nhận diện nhầm")
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h4 style='margin: 0; color: white;'>🎯 Phát hiện tốt nhất</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>True Positive cao nhất</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Mô hình:** {best_tp_model.model_name}")
        st.metric("True Positive", f"{best_tp_model.metrics['true_positive']:,}")
        
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Recall", f"{best_tp_model.metrics['recall']*100:.2f}%",
                     help="Tỷ lệ phát hiện gian lận")
        with col_d:
            st.metric("F1", f"{best_tp_model.metrics['f1_score']*100:.2f}%")
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h4 style='margin: 0; color: white;'>✨ Chính xác nhất</h4>
            <p style='margin: 5px 0; font-size: 0.9em;'>False Positive thấp nhất</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Mô hình:** {best_fp_model.model_name}")
        st.metric("False Positive", f"{best_fp_model.metrics['false_positive']:,}")
        
        col_e, col_f = st.columns(2)
        with col_e:
            st.metric("Precision", f"{best_fp_model.metrics['precision']*100:.2f}%",
                     help="Độ chính xác khi dự đoán gian lận")
        with col_f:
            st.metric("F1", f"{best_fp_model.metrics['f1_score']*100:.2f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Kết luận với highlight
    conclusion_text = f"""
    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                padding: 20px; border-radius: 10px; border-left: 5px solid #ff6b6b;'>
        <h4 style='margin-top: 0; color: #2d3436;'>💡 Kết luận</h4>
        <p style='font-size: 1.05em; line-height: 1.6; color: #2d3436; margin-bottom: 0;'>
            Mô hình <strong style='color: #d63031;'>{best_model.model_name}</strong> kết hợp với 
            phương pháp <strong style='color: #d63031;'>{method_name}</strong> cho ra kết quả 
            <strong>cân bằng tốt nhất</strong> giữa khả năng phát hiện gian lận (True Positive) 
            và tỷ lệ nhận diện nhầm (False Positive).
        </p>
    </div>
    """
    
    st.markdown(conclusion_text, unsafe_allow_html=True)
    
    # Thêm insights nếu có sự khác biệt giữa các mô hình
    if len(models) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📈 Phân tích chi tiết và đề xuất"):
            if best_model.model_name != best_tp_model.model_name:
                st.warning(f"""
                **Lưu ý:** Mô hình **{best_tp_model.model_name}** phát hiện được nhiều gian lận hơn 
                ({best_tp_model.metrics['true_positive']} so với {best_model.metrics['true_positive']}), 
                nhưng có thể có nhiều cảnh báo giả hơn ({best_tp_model.metrics['false_positive']} FP).
                
                **Đề xuất:** Nếu ưu tiên phát hiện tối đa gian lận và chấp nhận một số cảnh báo giả, 
                hãy xem xét sử dụng **{best_tp_model.model_name}**.
                """)
            
            if best_model.model_name != best_fp_model.model_name:
                st.info(f"""
                **Ghi chú:** Mô hình **{best_fp_model.model_name}** có số lượng cảnh báo giả thấp nhất 
                ({best_fp_model.metrics['false_positive']} FP), phù hợp nếu cần giảm thiểu phiền hà cho khách hàng.
                
                **Đề xuất:** Nếu ưu tiên trải nghiệm khách hàng và giảm số lần từ chối nhầm giao dịch hợp pháp,
                hãy xem xét **{best_fp_model.model_name}**.
                """)
            
            # So sánh performance
            f1_scores = [m.metrics['f1_score'] for m in models]
            f1_diff = (max(f1_scores) - min(f1_scores)) * 100
            
            if f1_diff < 1:
                st.success(f"""
                ✅ **Kết quả ổn định:** Các mô hình có hiệu suất tương đương nhau (chênh lệch F1-Score < 1%). 
                Có thể chọn bất kỳ mô hình nào tùy theo tiêu chí ưu tiên (tốc độ, tài nguyên, khả năng giải thích).
                """)
            else:
                st.warning(f"""
                ⚠️ **Chênh lệch đáng kể:** F1-Score chênh lệch {f1_diff:.2f}% giữa các mô hình. 
                Nên chọn mô hình có hiệu suất cao nhất cho production.
                """)


def create_detailed_metrics_table(models):
    """
    Tạo bảng metrics chi tiết với highlight.
    
    Args:
        models (list): Danh sách các FraudDetectionModel
        
    Returns:
        pd.DataFrame: Styled DataFrame
    """
    df = create_metrics_dataframe(models)
    
    # Hàm để highlight giá trị tốt nhất
    def highlight_best(s, props=''):
        # Tìm index của giá trị tốt nhất tùy theo cột
        if s.name in ['True Positive (TP)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']:
            # Với các metrics này, giá trị cao hơn là tốt hơn
            # Chuyển về số để so sánh
            numeric_values = s.str.rstrip('%').astype(float) if s.dtype == 'object' else s
            best_idx = numeric_values.idxmax()
        elif s.name == 'False Positive (FP)':
            # Với FP, giá trị thấp hơn là tốt hơn
            best_idx = s.idxmin()
        else:
            return [''] * len(s)
        
        return ['background-color: lightgreen; font-weight: bold' if i == best_idx else '' for i in range(len(s))]
    
    # Áp dụng styling vào DataFrame
    styled_df = df.style.apply(highlight_best, axis=0)
    
    return styled_df
