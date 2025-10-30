"""
Module xử lý dữ liệu cho dự án phát hiện gian lận thẻ tín dụng.

Module này cung cấp các hàm để:
- Tải dữ liệu từ nguồn công khai
- Xử lý dữ liệu mất cân bằng bằng Oversampling và SMOTE
- Chia dữ liệu thành tập huấn luyện và kiểm tra
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import streamlit as st


# URL dữ liệu công khai - Credit Card Fraud Detection từ Kaggle
# Lưu ý: Bạn có thể sử dụng link raw từ GitHub hoặc tải về trước
DATA_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"


@st.cache_data
def load_data():
    """
    Tải dữ liệu Credit Card Fraud Detection từ nguồn công khai.
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu giao dịch thẻ tín dụng
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(base_dir, "data", "creditcard.csv")

        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.info("Bạn có thể tải dữ liệu thủ công từ Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        return None


def get_data_info(df):
    """
    Lấy thông tin tổng quan về dữ liệu.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        
    Returns:
        dict: Dictionary chứa thông tin về dữ liệu
    """
    total_transactions = len(df)
    fraud_transactions = df['Class'].sum()
    fraud_percentage = (fraud_transactions / total_transactions) * 100
    
    return {
        'total_transactions': total_transactions,
        'fraud_transactions': fraud_transactions,
        'normal_transactions': total_transactions - fraud_transactions,
        'fraud_percentage': fraud_percentage
    }


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Chuẩn bị dữ liệu: tách features và target, chuẩn hóa dữ liệu.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu
        test_size (float): Tỷ lệ dữ liệu test
        random_state (int): Seed cho random state
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Tách features và target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Chuẩn hóa dữ liệu (nếu cần thiết, tùy thuộc vào mô hình)
    # Lưu ý: Dữ liệu đã được PCA transform nên có thể không cần scale thêm
    # Nhưng để đảm bảo tính nhất quán, ta vẫn scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def apply_oversampling(X_train, y_train, random_state=42):
    """
    Áp dụng phương pháp Oversampling để xử lý dữ liệu mất cân bằng.
    
    Phương pháp này lặp lại các mẫu gian lận (minority class) để cân bằng với
    các mẫu hợp pháp (majority class).
    
    Args:
        X_train (np.array): Dữ liệu training features
        y_train (pd.Series): Dữ liệu training labels
        random_state (int): Seed cho random state
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled


def apply_smote(X_train, y_train, random_state=42):
    """
    Áp dụng phương pháp SMOTE (Synthetic Minority Oversampling Technique).
    
    SMOTE tạo ra các mẫu gian lận tổng hợp bằng cách nội suy giữa các mẫu
    gian lận hiện có, thay vì chỉ đơn giản lặp lại chúng.
    
    Args:
        X_train (np.array): Dữ liệu training features
        y_train (pd.Series): Dữ liệu training labels
        random_state (int): Seed cho random state
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled


def get_resampling_info(y_original, y_resampled, method_name):
    """
    Lấy thông tin về kết quả resampling.
    
    Args:
        y_original (pd.Series): Labels gốc
        y_resampled (np.array): Labels sau khi resample
        method_name (str): Tên phương pháp resampling
        
    Returns:
        dict: Thông tin về resampling
    """
    original_fraud = y_original.sum()
    original_normal = len(y_original) - original_fraud
    original_total = len(y_original)
    
    resampled_fraud = y_resampled.sum()
    resampled_normal = len(y_resampled) - resampled_fraud
    resampled_total = len(y_resampled)
    
    return {
        'method': method_name,
        'original_total': original_total,
        'original_fraud': original_fraud,
        'original_normal': original_normal,
        'original_fraud_percentage': (original_fraud / original_total) * 100,
        'resampled_total': resampled_total,
        'resampled_fraud': resampled_fraud,
        'resampled_normal': resampled_normal,
        'resampled_fraud_percentage': (resampled_fraud / resampled_total) * 100
    }


def process_data_by_method(method, X_train, X_test, y_train, y_test):
    """
    Xử lý dữ liệu theo phương pháp được chọn.
    
    Args:
        method (str): Phương pháp xử lý ('original', 'oversampling', 'smote')
        X_train, X_test: Training và test features
        y_train, y_test: Training và test labels
        
    Returns:
        tuple: (X_train_processed, y_train_processed, method_info)
    """
    if method == 'Dữ liệu gốc (Mất cân bằng)':
        method_info = get_resampling_info(y_train, y_train, 'Dữ liệu gốc')
        return X_train, y_train, method_info
        
    elif method == 'Xử lý bằng Oversampling':
        X_resampled, y_resampled = apply_oversampling(X_train, y_train)
        method_info = get_resampling_info(y_train, y_resampled, 'Oversampling')
        return X_resampled, y_resampled, method_info
        
    elif method == 'Xử lý bằng SMOTE':
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        method_info = get_resampling_info(y_train, y_resampled, 'SMOTE')
        return X_resampled, y_resampled, method_info
