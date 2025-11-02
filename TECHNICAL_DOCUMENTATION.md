# 📚 TÀI LIỆU KỸ THUẬT - PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG

> **Đồ án**: Phát hiện Gian lận Thẻ Tín dụng bằng Machine Learning  
> **Từ Research Paper → Production Demo**  
> **Ngày cập nhật**: 31/10/2025

---

## 📋 MỤC LỤC

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Lý thuyết Machine Learning](#2-lý-thuyết-machine-learning)
3. [Xử lý dữ liệu mất cân bằng](#3-xử-lý-dữ-liệu-mất-cân-bằng)
4. [Các mô hình ML triển khai](#4-các-mô-hình-ml-triển-khai)
5. [Pipeline xử lý dữ liệu](#5-pipeline-xử-lý-dữ-liệu)
6. [Đánh giá và Metrics](#6-đánh-giá-và-metrics)
7. [Từ Training đến Production](#7-từ-training-đến-production)
8. [Best Practices](#8-best-practices)

---

## 1. TỔNG QUAN KIẾN TRÚC

### 1.1. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION SYSTEM                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │   app.py - Streamlit Web Application               │     │
│  │   - Sidebar Controls                                │     │
│  │   - Main Display Area                               │     │
│  │   - Interactive Visualizations                      │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    models.py │  │data_processing│  │visualization │      │
│  │              │  │      .py      │  │     .py      │      │
│  │  - Logistic  │  │  - Load Data  │  │  - Confusion │      │
│  │  - D.Tree    │  │  - Prepare    │  │    Matrix    │      │
│  │  - Bayesian  │  │  - Oversample │  │  - Charts    │      │
│  │  - Train     │  │  - SMOTE      │  │  - Tables    │      │
│  │  - Evaluate  │  │  - Split      │  │  - Recommend │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │   data/creditcard.csv (284,807 transactions)       │     │
│  │   - 30 features (Time, Amount, V1-V28)             │     │
│  │   - Binary target (Class: 0/1)                     │     │
│  │   - Highly imbalanced (0.172% fraud)               │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2. Luồng xử lý dữ liệu (Data Flow)

```
📊 RAW DATA                                    🎯 PREDICTIONS
    │
    ├─► Load CSV (creditcard.csv)
    │       │
    │       ▼
    ├─► Preprocess (prepare_data)
    │       │
    │       ├─► Normalize Amount (StandardScaler)
    │       ├─► Drop Time column
    │       ├─► Split features/target (X, y)
    │       └─► Train/Test Split (80/20)
    │               │
    │               ▼
    ├─► Resample Training Data
    │       │
    │       ├─► Option 1: Original (Imbalanced)
    │       ├─► Option 2: Random Oversampling
    │       └─► Option 3: SMOTE
    │               │
    │               ▼
    ├─► Train Models
    │       │
    │       ├─► Logistic Regression
    │       ├─► Decision Tree
    │       └─► Bayesian Network
    │               │
    │               ▼
    ├─► Predict on Test Set
    │       │
    │       ▼
    └─► Evaluate Performance
            │
            ├─► Confusion Matrix
            ├─► Accuracy, Precision, Recall
            ├─► F1-Score
            └─► Recommendations
```

---

## 2. LÝ THUYẾT MACHINE LEARNING

### 2.1. Bài toán phân loại nhị phân (Binary Classification)

**Định nghĩa:**
- Input: Vector đặc trưng X = [Amount, V1, V2, ..., V28] ∈ ℝ²⁹
- Output: Class y ∈ {0, 1} (0 = Legitimate, 1 = Fraud)
- Mục tiêu: Tìm hàm f: ℝ²⁹ → {0, 1}

**Thách thức:**
1. **Imbalanced Data**: Fraud cases chỉ chiếm 0.172%
2. **Cost-sensitive**: FN (bỏ sót gian lận) nghiêm trọng hơn FP
3. **Dimensionality**: 29 features sau PCA

### 2.2. Supervised Learning Pipeline

```python
# 1. Data Preparation
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 2. Resampling (nếu cần)
X_resampled, y_resampled = apply_smote(X_train, y_train)

# 3. Model Training
model.fit(X_resampled, y_resampled)

# 4. Prediction
y_pred = model.predict(X_test)

# 5. Evaluation
metrics = evaluate(y_test, y_pred)
```

### 2.3. Bias-Variance Tradeoff

```
High Bias (Underfitting)     ←→     High Variance (Overfitting)
        │                                      │
        │                                      │
  Logistic Regression                   Deep Decision Tree
        │                                      │
        └──────────────┬───────────────────────┘
                       │
                  Sweet Spot
                 (Best Model)
```

**Trong dự án:**
- Logistic Regression: High bias, low variance
- Decision Tree: Low bias, high variance (cần tuning)
- Bayesian: Moderate bias-variance

---

## 3. XỬ LÝ DỮ LIỆU MẤT CÂN BẰNG

### 3.1. Vấn đề Imbalanced Data

**Dataset gốc:**
```
Class 0 (Legitimate): 284,315 samples (99.828%)
Class 1 (Fraud):          492 samples ( 0.172%)
Imbalance Ratio: 578:1
```

**Hậu quả:**
- Model học thiên vị về majority class
- Precision cao nhưng Recall thấp (bỏ sót gian lận)
- Accuracy không phản ánh hiệu suất thực tế

### 3.2. Phương pháp 1: Random Oversampling

**Nguyên lý:**
```python
# Lặp lại ngẫu nhiên các mẫu minority class
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Kết quả: Class 0 = Class 1 (cân bằng 50-50)
```

**Ưu điểm:**
- ✅ Đơn giản, dễ implement
- ✅ Nhanh, không tốn tài nguyên
- ✅ Không thay đổi phân phối dữ liệu gốc

**Nhược điểm:**
- ❌ Overfitting (lặp lại chính xác cùng mẫu)
- ❌ Không tạo thông tin mới
- ❌ Có thể học "nhiễu" từ các mẫu outlier

**Code implementation:**
```python
def apply_oversampling(X_train, y_train, random_state=42):
    """
    Áp dụng Random Oversampling để cân bằng dữ liệu.
    
    Phương pháp: Lặp lại các mẫu minority class ngẫu nhiên
    cho đến khi cân bằng với majority class.
    """
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled
```

### 3.3. Phương pháp 2: SMOTE

**Nguyên lý:**
```
SMOTE (Synthetic Minority Oversampling Technique)

1. Chọn 1 mẫu minority class: x_i
2. Tìm k nearest neighbors (thường k=5)
3. Chọn ngẫu nhiên 1 neighbor: x_nn
4. Tạo mẫu mới: x_new = x_i + λ × (x_nn - x_i)
   với λ ∈ [0, 1] ngẫu nhiên
5. Lặp lại cho đến khi cân bằng
```

**Minh họa:**
```
     x_nn
      ●
      │╲
      │ ╲ x_new (synthetic)
      │  ⊗
      │ ╱
      │╱
      ●
     x_i
```

**Ưu điểm:**
- ✅ Tạo dữ liệu đa dạng (synthetic samples)
- ✅ Giảm overfitting
- ✅ Cải thiện generalization
- ✅ Học được vùng quyết định tốt hơn

**Nhược điểm:**
- ❌ Tốn thời gian tính toán (k-NN)
- ❌ Có thể tạo mẫu "không hợp lý" nếu dữ liệu nhiễu
- ❌ Không hiệu quả với high-dimensional data

**Code implementation:**
```python
def apply_smote(X_train, y_train, random_state=42):
    """
    Áp dụng SMOTE để tạo mẫu synthetic minority class.
    
    Phương pháp: Nội suy giữa các mẫu minority class và
    k-nearest neighbors của chúng để tạo mẫu mới.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    return X_resampled, y_resampled
```

### 3.4. So sánh Oversampling vs SMOTE

| Aspect | Random Oversampling | SMOTE |
|--------|-------------------|-------|
| **Tốc độ** | Rất nhanh | Chậm hơn (k-NN) |
| **Overfitting** | Cao | Thấp hơn |
| **Diversity** | Không có | Cao |
| **Memory** | Ít | Nhiều hơn |
| **Best for** | Quick prototype | Production model |
| **Performance** | Tốt | Tốt hơn thường xuyên |

---

## 4. CÁC MÔ HÌNH ML TRIỂN KHAI

### 4.1. Logistic Regression

**Lý thuyết:**
```
Hàm dự đoán:
    ŷ = sigmoid(w^T x + b)
    
Sigmoid function:
    σ(z) = 1 / (1 + e^(-z))
    
Loss function (Binary Cross-Entropy):
    L(w,b) = -1/m Σ[y log(ŷ) + (1-y)log(1-ŷ)]
    
Optimization:
    Gradient Descent hoặc L-BFGS
```

**Đặc điểm:**
- **Tuyến tính**: Decision boundary là siêu phẳng
- **Xác suất**: Output là xác suất thuộc class 1
- **Fast training**: Converge nhanh với dữ liệu lớn
- **Interpretable**: Weights thể hiện tầm quan trọng features

**Implementation:**
```python
class LogisticRegressionModel(FraudDetectionModel):
    """
    Mô hình Hồi quy Logistic cho phân loại nhị phân.
    
    Sử dụng hàm sigmoid để map linear combination của
    features thành xác suất [0,1].
    """
    
    def __init__(self, max_iter=1000, random_state=42):
        model = LogisticRegression(
            max_iter=max_iter,      # Số vòng lặp tối đa
            random_state=random_state,
            n_jobs=-1,              # Parallel processing
            solver='lbfgs'          # Optimization algorithm
        )
        super().__init__('Hồi quy Logistic', model)
```

**Khi nào sử dụng:**
- ✅ Baseline model (always start here)
- ✅ Cần interpretability
- ✅ Dữ liệu lớn, cần training nhanh
- ✅ Features có quan hệ gần tuyến tính với target

### 4.2. Decision Tree

**Lý thuyết:**
```
Cấu trúc cây:
                [Amount > 100?]
                /            \
              Yes             No
              /                \
      [V1 > 0.5?]         [V2 > 0.3?]
       /      \             /      \
    Fraud   Normal      Fraud   Normal
    
Splitting Criterion (Gini Impurity):
    Gini(p) = 1 - Σ(p_i²)
    
Information Gain:
    IG = Gini(parent) - Σ(weighted_Gini(children))
```

**Đặc điểm:**
- **Phi tuyến**: Có thể học decision boundary phức tạp
- **Non-parametric**: Không giả định về phân phối dữ liệu
- **Feature importance**: Tự động đánh giá features
- **Overfitting prone**: Cần pruning (max_depth)

**Implementation:**
```python
class DecisionTreeModel(FraudDetectionModel):
    """
    Mô hình Cây quyết định.
    
    Tạo cấu trúc cây phân loại bằng cách chia không gian
    features dựa trên tiêu chí Gini impurity.
    """
    
    def __init__(self, max_depth=10, random_state=42):
        model = DecisionTreeClassifier(
            max_depth=max_depth,        # Giới hạn độ sâu để tránh overfitting
            random_state=random_state,
            criterion='gini',           # Splitting criterion
            min_samples_split=2,        # Minimum samples to split
            min_samples_leaf=1          # Minimum samples in leaf
        )
        super().__init__('Cây quyết định', model)
```

**Khi nào sử dụng:**
- ✅ Dữ liệu có quan hệ phi tuyến
- ✅ Cần interpretability (visualize tree)
- ✅ Features có interactions
- ⚠️ Cần tuning hyperparameters cẩn thận

### 4.3. Bayesian Network (Gaussian Naive Bayes)

**Lý thuyết:**
```
Bayes' Theorem:
    P(Fraud|X) = P(X|Fraud) × P(Fraud) / P(X)
    
Naive Assumption (Independence):
    P(X|Fraud) = P(x₁|Fraud) × P(x₂|Fraud) × ... × P(xₙ|Fraud)
    
Gaussian Distribution (cho continuous features):
    P(xᵢ|Fraud) = 1/√(2πσ²) × e^(-(xᵢ-μ)²/(2σ²))
    
Decision Rule:
    ŷ = argmax_c P(c) × ∏ P(xᵢ|c)
```

**Đặc điểm:**
- **Probabilistic**: Dựa trên lý thuyết xác suất
- **Fast**: Training và prediction rất nhanh
- **Small data**: Hoạt động tốt với ít dữ liệu
- **Independence assumption**: Giả định features độc lập (naive)

**Implementation:**
```python
class BayesianNetworkModel(FraudDetectionModel):
    """
    Mô hình Mạng Bayesian (Gaussian Naive Bayes).
    
    Sử dụng định lý Bayes với giả định các features
    độc lập có điều kiện (conditional independence).
    """
    
    def __init__(self):
        model = GaussianNB()
        # Không cần hyperparameters cho Gaussian NB
        super().__init__('Mạng Bayesian', model)
```

**Khi nào sử dụng:**
- ✅ Dữ liệu nhỏ
- ✅ Cần training/prediction cực nhanh
- ✅ Features gần như độc lập
- ✅ Baseline probabilistic model

### 4.4. So sánh các mô hình

| Model | Complexity | Speed | Overfitting | Interpretability |
|-------|-----------|-------|-------------|------------------|
| **Logistic** | Low | Fast | Low | High |
| **Decision Tree** | Medium-High | Medium | High | High |
| **Bayesian** | Low | Very Fast | Low | Medium |

---

## 5. PIPELINE XỬ LÝ DỮ LIỆU

### 5.1. Data Preprocessing (theo bài báo)

**Quy trình chuẩn:**
```python
def prepare_data(df, test_size=0.2, random_state=42):
    """
    Pipeline xử lý dữ liệu theo đúng bài báo:
    
    1. Normalize Amount (chỉ Amount, không phải V1-V28)
    2. Drop Time column
    3. Split features and target
    4. Train/Test split (stratified)
    """
    # Bước 1: Copy để không ảnh hưởng dữ liệu gốc
    df_processed = df.copy()
    
    # Bước 2: Chuẩn hóa Amount (StandardScaler)
    # Lý do: Amount có range rất lớn, cần normalize
    # V1-V28 đã được PCA nên KHÔNG cần normalize lại
    scaler = StandardScaler()
    df_processed['Amount'] = scaler.fit_transform(
        df_processed['Amount'].values.reshape(-1, 1)
    )
    
    # Bước 3: Loại bỏ Time
    # Lý do: Time không có ý nghĩa prediction trong context này
    df_processed = df_processed.drop('Time', axis=1)
    
    # Bước 4: Tách features và target
    X = df_processed.drop('Class', axis=1)  # 29 features
    y = df_processed['Class']                # Binary target
    
    # Bước 5: Chia train/test với stratify
    # stratify=y đảm bảo tỷ lệ fraud giống nhau trong train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,      # 20% test
        random_state=random_state,
        stratify=y                # Quan trọng với imbalanced data
    )
    
    # Chuyển về numpy array
    X_train = X_train.values
    X_test = X_test.values
    
    return X_train, X_test, y_train, y_test
```

**Lưu ý quan trọng:**
1. ❌ **KHÔNG** chuẩn hóa V1-V28 (đã được PCA)
2. ✅ **CHỈ** chuẩn hóa Amount
3. ✅ Loại bỏ Time **TRƯỚC** khi split
4. ✅ Chuẩn hóa **TRƯỚC** khi split (tránh data leakage)
5. ✅ Sử dụng stratified split

### 5.2. Resampling Strategy

**Factory Pattern:**
```python
def process_data_by_method(method, X_train, y_train):
    """
    Xử lý dữ liệu theo phương pháp được chọn.
    
    Design Pattern: Strategy Pattern
    - Encapsulates resampling algorithms
    - Allows runtime selection
    """
    if method == 'Dữ liệu gốc (Mất cân bằng)':
        # Không resample
        method_info = get_resampling_info(y_train, y_train, 'Original')
        return X_train, y_train, method_info
        
    elif method == 'Xử lý bằng Oversampling':
        X_resampled, y_resampled = apply_oversampling(X_train, y_train)
        method_info = get_resampling_info(y_train, y_resampled, 'Oversampling')
        return X_resampled, y_resampled, method_info
        
    elif method == 'Xử lý bằng SMOTE':
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        method_info = get_resampling_info(y_train, y_resampled, 'SMOTE')
        return X_resampled, y_resampled, method_info
```

### 5.3. Train-Test Split Strategy

**Stratified Splitting:**
```
Original Distribution:
    Fraud: 0.172%
    
Train Set (80%):
    Fraud: 0.172% (same ratio maintained)
    
Test Set (20%):
    Fraud: 0.172% (same ratio maintained)
    
✅ Benefits:
    - Representative samples
    - Consistent evaluation
    - No bias in split
```

---

## 6. ĐÁNH GIÁ VÀ METRICS

### 6.1. Confusion Matrix

**Định nghĩa:**
```
                    Predicted
                  Negative  Positive
                  (Normal)  (Fraud)
Actual  Negative    TN        FP
        (Normal)
        
        Positive    FN        TP
        (Fraud)
```

**Ý nghĩa trong fraud detection:**
- **TN (True Negative)**: ✅ Normal transaction predicted correctly
- **TP (True Positive)**: ✅ Fraud transaction detected correctly
- **FP (False Positive)**: ❌ Normal flagged as fraud (False alarm)
- **FN (False Negative)**: ❌ Fraud missed (Serious!)

**Code implementation:**
```python
def evaluate(self, y_test, y_pred=None):
    """
    Đánh giá mô hình với confusion matrix và metrics.
    """
    if y_pred is None:
        y_pred = self.predictions
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
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
```

### 6.2. Metrics Chi tiết

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Ưu điểm: Dễ hiểu
Nhược điểm: KHÔNG phù hợp với imbalanced data
    → Model predict tất cả là Normal → 99.8% accuracy!
```

**Precision:**
```
Precision = TP / (TP + FP)

Ý nghĩa: Trong số các giao dịch được dự đoán là FRAUD,
         bao nhiêu % thực sự là FRAUD?
         
Impact: FP cao → Nhiều false alarms → Khách hàng khó chịu
```

**Recall (Sensitivity, True Positive Rate):**
```
Recall = TP / (TP + FN)

Ý nghĩa: Trong số các giao dịch THỰC SỰ LÀ FRAUD,
         bao nhiêu % được phát hiện?
         
Impact: FN cao → Bỏ sót gian lận → Loss tiền
Priority: RECALL cao là quan trọng nhất trong fraud detection!
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Ý nghĩa: Harmonic mean của Precision và Recall
         Cân bằng giữa 2 metrics
         
Best for: Imbalanced classification evaluation
```

### 6.3. Precision-Recall Tradeoff

```
High Threshold (0.9):
    → Few predictions as Fraud
    → High Precision, Low Recall
    → Miss many frauds (bad!)
    
Low Threshold (0.1):
    → Many predictions as Fraud
    → Low Precision, High Recall
    → Many false alarms (annoying but safer)
    
Optimal Point:
    → Depends on business requirements
    → Cost of FN vs Cost of FP
```

### 6.4. Metric Selection trong Fraud Detection

**Priority ranking:**
1. **Recall** (Most important) - Không bỏ sót gian lận
2. **F1-Score** - Cân bằng tổng thể
3. **Precision** - Giảm false alarms
4. **Accuracy** (Least important) - Misleading với imbalanced data

---

## 7. TỪ TRAINING ĐẾN PRODUCTION

### 7.1. Model Training Flow

```python
def train_and_evaluate_models(model_names, X_train, y_train, X_test, y_test):
    """
    Pipeline hoàn chỉnh từ training đến evaluation.
    
    Flow:
    1. Iterate through selected models
    2. Create model instance (Factory Pattern)
    3. Train on resampled data
    4. Predict on original test set
    5. Evaluate and collect metrics
    6. Return trained models with results
    """
    trained_models = []
    
    for model_name in model_names:
        # Step 1: Create model
        model = create_model(model_name)
        
        # Step 2: Train
        model.train(X_train, y_train)
        
        # Step 3: Predict
        model.predict(X_test)
        
        # Step 4: Evaluate
        model.evaluate(y_test)
        
        # Step 5: Collect
        trained_models.append(model)
    
    return trained_models
```

### 7.2. Production Demo Architecture

**Component Structure:**
```
app.py (Main Application)
├── Presentation Layer
│   ├── Streamlit UI components
│   ├── Sidebar controls
│   ├── Main display area
│   └── Interactive visualizations
│
├── Business Logic Layer
│   ├── data_processing.py
│   │   ├── load_data()
│   │   ├── prepare_data()
│   │   ├── apply_oversampling()
│   │   ├── apply_smote()
│   │   └── process_data_by_method()
│   │
│   ├── models.py
│   │   ├── FraudDetectionModel (Base Class)
│   │   ├── LogisticRegressionModel
│   │   ├── DecisionTreeModel
│   │   ├── BayesianNetworkModel
│   │   ├── create_model() (Factory)
│   │   └── train_and_evaluate_models()
│   │
│   └── visualization.py
│       ├── plot_confusion_matrix()
│       ├── create_metrics_dataframe()
│       ├── display_metrics_summary()
│       ├── plot_comparison_chart()
│       └── get_recommendation()
│
└── Data Layer
    └── data/creditcard.csv
```

### 7.3. Design Patterns Used

**1. Factory Pattern:**
```python
def create_model(model_name):
    """
    Factory để tạo model instances.
    
    Benefits:
    - Centralized creation logic
    - Easy to extend with new models
    - Decoupling from concrete classes
    """
    models = {
        'Hồi quy Logistic': LogisticRegressionModel,
        'Cây quyết định': DecisionTreeModel,
        'Mạng Bayesian': BayesianNetworkModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ!")
    
    return models[model_name]()
```

**2. Strategy Pattern:**
```python
# Encapsulate resampling algorithms
strategies = {
    'original': lambda X, y: (X, y),
    'oversampling': apply_oversampling,
    'smote': apply_smote
}

# Runtime selection
X_resampled, y_resampled = strategies[method](X_train, y_train)
```

**3. Template Method Pattern:**
```python
class FraudDetectionModel:
    """
    Base class defining template for all models.
    """
    def train(self, X, y):
        # Common training logic
        pass
    
    def predict(self, X):
        # Common prediction logic
        pass
    
    def evaluate(self, y_test):
        # Common evaluation logic
        pass
```

### 7.4. Streamlit Integration

**Caching Strategy:**
```python
@st.cache_data
def load_data():
    """
    Cache data loading để tránh reload nhiều lần.
    
    Benefits:
    - Faster app performance
    - Reduce API calls
    - Better UX
    """
    df = pd.read_csv(file_path)
    return df
```

**Interactive Controls:**
```python
# Sidebar controls
data_method = st.sidebar.selectbox(
    "Phương pháp xử lý:",
    ['Dữ liệu gốc', 'Oversampling', 'SMOTE']
)

selected_models = st.sidebar.multiselect(
    "Chọn mô hình:",
    ['Hồi quy Logistic', 'Cây quyết định', 'Mạng Bayesian']
)

# Action button
if st.sidebar.button("Huấn luyện và Đánh giá"):
    # Execute pipeline
    run_training_pipeline()
```

**Visualization Display:**
```python
# Confusion Matrix
fig_cm = plot_confusion_matrix(cm, model_name)
st.pyplot(fig_cm)

# Metrics cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("True Positive", tp)
with col2:
    st.metric("False Positive", fp)
# ...

# Comparison charts
fig_comparison = plot_comparison_chart(models, 'f1_score')
st.pyplot(fig_comparison)
```

### 7.5. Error Handling & Validation

**Data Validation:**
```python
def load_data():
    try:
        df = pd.read_csv(file_path)
        
        # Validate structure
        required_columns = ['Time', 'Amount', 'Class']
        assert all(col in df.columns for col in required_columns)
        
        # Validate data types
        assert df['Class'].isin([0, 1]).all()
        
        return df
        
    except FileNotFoundError:
        st.error("File not found!")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None
```

**Model Validation:**
```python
def train(self, X_train, y_train):
    # Validate inputs
    if X_train.shape[0] != len(y_train):
        raise ValueError("Mismatch between X and y")
    
    # Training with progress
    with st.spinner(f'Training {self.model_name}...'):
        self.model.fit(X_train, y_train)
        self.is_trained = True
```

---

## 8. BEST PRACTICES

### 8.1. Data Science Best Practices

**1. Always Split BEFORE Resampling:**
```python
# ✅ CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

# ❌ WRONG - Data leakage!
X_resampled, y_resampled = apply_smote(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
```

**2. Use Stratified Split:**
```python
# ✅ CORRECT - Maintains class distribution
train_test_split(X, y, stratify=y)

# ❌ WRONG - Random split may create bias
train_test_split(X, y)
```

**3. Normalize Before Split (if applying to all data):**
```python
# ✅ CORRECT - Normalize Amount before split
df['Amount'] = scaler.fit_transform(df['Amount'])
X_train, X_test = train_test_split(df)

# Note: Nếu normalize sau split, cần fit trên train, transform trên test
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

**4. Don't Normalize PCA Features:**
```python
# ✅ CORRECT
df['Amount'] = scaler.fit_transform(df['Amount'])
# V1-V28 không normalize

# ❌ WRONG
scaler.fit_transform(df[['Amount', 'V1', 'V2', ..., 'V28']])
```

### 8.2. Code Organization

**Module Structure:**
```
src/
├── __init__.py          # Package initialization
├── data_processing.py   # Single responsibility: Data
├── models.py            # Single responsibility: Models
└── visualization.py     # Single responsibility: Viz
```

**Class Design:**
```python
# ✅ GOOD - Base class with inheritance
class FraudDetectionModel:
    def train(self, X, y): pass
    def predict(self, X): pass
    def evaluate(self, y_test): pass

class LogisticRegressionModel(FraudDetectionModel):
    # Specific implementation
```

**Function Design:**
```python
# ✅ GOOD - Single purpose, clear naming
def apply_smote(X_train, y_train, random_state=42):
    """Clear docstring"""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

# ❌ BAD - Multiple responsibilities
def process_everything(df):
    # Load, clean, split, resample, train...
    pass
```

### 8.3. Performance Optimization

**1. Caching:**
```python
@st.cache_data  # Cache expensive operations
def load_data():
    return pd.read_csv(large_file)
```

**2. Parallel Processing:**
```python
LogisticRegression(n_jobs=-1)  # Use all CPU cores
```

**3. Memory Management:**
```python
# Convert to numpy when needed
X_train = X_train.values  # DataFrame → numpy (faster)
```

### 8.4. Documentation

**1. Docstrings:**
```python
def apply_smote(X_train, y_train, random_state=42):
    """
    Áp dụng SMOTE để xử lý dữ liệu mất cân bằng.
    
    Args:
        X_train (np.array): Training features
        y_train (pd.Series): Training labels
        random_state (int): Seed for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled)
        
    Example:
        >>> X_res, y_res = apply_smote(X_train, y_train)
        >>> print(y_res.value_counts())
    """
```

**2. Comments:**
```python
# Good: Explain WHY, not WHAT
# Normalize Amount because it has large variance
# V1-V28 already normalized by PCA, don't touch

# Bad: State the obvious
# Loop through models
for model in models:
```

### 8.5. Testing Strategy

**Unit Tests:**
```python
def test_prepare_data():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Test split ratio
    assert len(X_test) / len(df) ≈ 0.2
    
    # Test stratification
    assert y_train.mean() ≈ y_test.mean()
    
    # Test features
    assert X_train.shape[1] == 29  # Time dropped
```

**Integration Tests:**
```python
def test_full_pipeline():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_res, y_res = apply_smote(X_train, y_train)
    
    model = LogisticRegressionModel()
    model.train(X_res, y_res)
    predictions = model.predict(X_test)
    metrics = model.evaluate(y_test)
    
    assert 'f1_score' in metrics
    assert 0 <= metrics['f1_score'] <= 1
```

### 8.6. Git Best Practices

**Commit Messages:**
```bash
✅ GOOD
git commit -m "fix: normalize only Amount, not V1-V28 (align with paper)"

❌ BAD
git commit -m "fix bug"
```

**Branch Strategy:**
```
main           → Production-ready code
develop        → Development branch
feature/smote  → Feature branches
fix/normalize  → Bugfix branches
```

---

## 9. KẾT LUẬN

### 9.1. Tổng kết kiến thức đã học

**Machine Learning:**
- ✅ Binary Classification
- ✅ Supervised Learning
- ✅ Imbalanced Data Handling
- ✅ Model Evaluation Metrics
- ✅ Bias-Variance Tradeoff

**Algorithms:**
- ✅ Logistic Regression (Linear)
- ✅ Decision Tree (Non-linear)
- ✅ Naive Bayes (Probabilistic)
- ✅ Random Oversampling
- ✅ SMOTE

**Engineering:**
- ✅ Data Pipeline Design
- ✅ Design Patterns (Factory, Strategy, Template)
- ✅ Web Application (Streamlit)
- ✅ Code Organization
- ✅ Documentation

**Best Practices:**
- ✅ Train/Test Split Strategy
- ✅ Stratified Sampling
- ✅ Feature Engineering
- ✅ Model Comparison
- ✅ Production Deployment

### 9.2. Điểm khác biệt giữa Research và Production

| Aspect | Research Paper | Production Demo |
|--------|---------------|-----------------|
| **Focus** | Novel algorithm | User experience |
| **Code** | Jupyter notebooks | Modular, reusable |
| **Metrics** | Academic rigor | Business value |
| **UI** | Plots in paper | Interactive web app |
| **Docs** | Paper itself | Code + README |
| **Reproducibility** | Reported numbers | Runnable code |

### 9.3. Lessons Learned

**1. Alignment với Paper:**
- Đọc kỹ paper để implement đúng pipeline
- Normalize chỉ Amount, không phải V1-V28
- Drop Time column như mô tả
- Reproduce kết quả chính xác

**2. Production Considerations:**
- User-friendly interface (Streamlit)
- Clear documentation
- Error handling
- Performance optimization
- Extensibility (easy to add new models)

**3. Trade-offs:**
- Accuracy vs Interpretability
- Precision vs Recall
- Speed vs Performance
- Simplicity vs Flexibility

### 9.4. Next Steps

**Để cải thiện demo:**
1. Thêm ROC-AUC curve visualization
2. Implement model persistence (save/load)
3. Add hyperparameter tuning
4. Deploy to cloud (Streamlit Cloud, Heroku)
5. Add more models (Random Forest, XGBoost)
6. Implement cost-sensitive learning
7. Add feature importance analysis
8. Create API endpoint

---

## 📚 TÀI LIỆU THAM KHẢO

1. **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. **SMOTE Paper**: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
3. **Imbalanced Learning**: He & Garcia (2009) - "Learning from Imbalanced Data"
4. **Scikit-learn**: [Official Documentation](https://scikit-learn.org/)
5. **Streamlit**: [Official Documentation](https://docs.streamlit.io/)

---

**Document Version**: 1.0  
**Last Updated**: 31/10/2025  
**Author**: Fraud Detection Team
