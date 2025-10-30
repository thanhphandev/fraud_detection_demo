"""
Module chứa các mô hình Machine Learning cho phát hiện gian lận thẻ tín dụng.

Module này cung cấp:
- Logistic Regression
- Decision Tree
- Bayesian Network (sử dụng Gaussian Naive Bayes)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import streamlit as st


class FraudDetectionModel:
    """
    Lớp cơ sở cho các mô hình phát hiện gian lận.
    """
    
    def __init__(self, model_name, model):
        """
        Khởi tạo mô hình.
        
        Args:
            model_name (str): Tên mô hình
            model: Instance của sklearn model
        """
        self.model_name = model_name
        self.model = model
        self.is_trained = False
        self.predictions = None
        self.metrics = {}
        
    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        with st.spinner(f'Đang huấn luyện mô hình {self.model_name}...'):
            self.model.fit(X_train, y_train)
            self.is_trained = True
        
    def predict(self, X_test):
        """
        Dự đoán trên tập test.
        
        Args:
            X_test: Test features
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Mô hình chưa được huấn luyện!")
        
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def evaluate(self, y_test, y_pred=None):
        """
        Đánh giá hiệu suất mô hình.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels (optional, sử dụng self.predictions nếu None)
            
        Returns:
            dict: Dictionary chứa các metrics
        """
        if y_pred is None:
            if self.predictions is None:
                raise ValueError("Chưa có predictions. Vui lòng gọi predict() trước.")
            y_pred = self.predictions
        
        # Tính toán confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Tính các metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        self.metrics = {
            'model_name': self.model_name,
            'confusion_matrix': cm,
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        return self.metrics


class LogisticRegressionModel(FraudDetectionModel):
    """
    Mô hình Hồi quy Logistic.
    
    Hồi quy Logistic là một thuật toán phân loại tuyến tính sử dụng hàm sigmoid
    để dự đoán xác suất của một quan sát thuộc về một lớp cụ thể.
    """
    
    def __init__(self, max_iter=1000, random_state=42):
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1
        )
        super().__init__('Hồi quy Logistic', model)


class DecisionTreeModel(FraudDetectionModel):
    """
    Mô hình Cây quyết định.
    
    Cây quyết định là một thuật toán học máy phi tuyến tính, tạo ra một cấu trúc
    cây với các node quyết định dựa trên các đặc trưng của dữ liệu.
    """
    
    def __init__(self, max_depth=10, random_state=42):
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        super().__init__('Cây quyết định', model)


class BayesianNetworkModel(FraudDetectionModel):
    """
    Mô hình Mạng Bayesian (sử dụng Gaussian Naive Bayes).
    
    Naive Bayes là một thuật toán dựa trên định lý Bayes với giả định về
    tính độc lập có điều kiện giữa các đặc trưng.
    """
    
    def __init__(self):
        model = GaussianNB()
        super().__init__('Mạng Bayesian', model)


def create_model(model_name):
    """
    Factory function để tạo mô hình dựa trên tên.
    
    Args:
        model_name (str): Tên mô hình
        
    Returns:
        FraudDetectionModel: Instance của mô hình
    """
    models = {
        'Hồi quy Logistic': LogisticRegressionModel,
        'Cây quyết định': DecisionTreeModel,
        'Mạng Bayesian': BayesianNetworkModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ!")
    
    return models[model_name]()


def train_and_evaluate_models(model_names, X_train, y_train, X_test, y_test):
    """
    Huấn luyện và đánh giá nhiều mô hình.
    
    Args:
        model_names (list): Danh sách tên các mô hình
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        list: Danh sách các mô hình đã được huấn luyện và đánh giá
    """
    trained_models = []
    
    for model_name in model_names:
        # Tạo mô hình
        model = create_model(model_name)
        
        # Huấn luyện
        model.train(X_train, y_train)
        
        # Dự đoán
        model.predict(X_test)
        
        # Đánh giá
        model.evaluate(y_test)
        
        trained_models.append(model)
    
    return trained_models


def get_best_model(models):
    """
    Tìm mô hình tốt nhất dựa trên F1-score.
    
    Args:
        models (list): Danh sách các mô hình đã được đánh giá
        
    Returns:
        FraudDetectionModel: Mô hình tốt nhất
    """
    if not models:
        return None
    
    best_model = max(models, key=lambda m: m.metrics.get('f1_score', 0))
    return best_model
