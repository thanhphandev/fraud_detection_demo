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
    Đưa ra khuyến nghị dựa trên kết quả các mô hình.
    
    Args:
        models (list): Danh sách các FraudDetectionModel
        method_name (str): Tên phương pháp xử lý dữ liệu
        
    Returns:
        str: Chuỗi chứa khuyến nghị
    """
    if not models:
        return "Không có mô hình nào được đánh giá."
    
    # Tìm mô hình tốt nhất dựa trên F1-score
    best_model = max(models, key=lambda m: m.metrics['f1_score'])
    
    # Tìm mô hình có TP cao nhất
    best_tp_model = max(models, key=lambda m: m.metrics['true_positive'])
    
    # Tìm mô hình có FP thấp nhất
    best_fp_model = min(models, key=lambda m: m.metrics['false_positive'])
    
    recommendation = f"""
    ### Khuyến nghị
    
    Dựa trên kết quả phân tích với phương pháp **{method_name}**:
    
    - **Mô hình tổng thể tốt nhất (F1-Score):** **{best_model.model_name}** 
      - F1-Score: {best_model.metrics['f1_score']*100:.2f}%
      - True Positive: {best_model.metrics['true_positive']}
      - False Positive: {best_model.metrics['false_positive']}
    
    - **Mô hình phát hiện gian lận tốt nhất (TP cao nhất):** **{best_tp_model.model_name}**
      - True Positive: {best_tp_model.metrics['true_positive']}
    
    - **Mô hình ít nhận diện nhầm nhất (FP thấp nhất):** **{best_fp_model.model_name}**
      - False Positive: {best_fp_model.metrics['false_positive']}
    
    **Kết luận:** Mô hình **{best_model.model_name}** kết hợp với phương pháp **{method_name}** 
    cho ra kết quả cân bằng tốt nhất giữa khả năng phát hiện gian lận (True Positive) 
    và tỷ lệ nhận diện nhầm (False Positive).
    """
    
    return recommendation


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
