# Phát hiện Gian lận Thẻ Tín dụng bằng Machine Learning

Ứng dụng demo tương tác sử dụng **Streamlit** để phát hiện gian lận thẻ tín dụng và so sánh hiệu quả của các mô hình Machine Learning khác nhau khi xử lý dữ liệu mất cân bằng.

## Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Dataset](#-dataset)
- [Phương pháp](#-phương-pháp)
- [Kết quả mẫu](#-kết-quả-mẫu)
- [Tác giả](#-tác-giả)

---

## \Giới thiệu

Đây là ứng dụng demo cho đồ án **"Phát hiện gian lận thẻ tín dụng bằng học máy"**. Ứng dụng cho phép người dùng:

- Chọn phương pháp xử lý dữ liệu mất cân bằng (Oversampling, SMOTE)
- Lựa chọn và so sánh nhiều mô hình Machine Learning
- Xem trực quan kết quả qua Confusion Matrix và các biểu đồ
- Nhận khuyến nghị về mô hình tốt nhất

---

## Tính năng

### Giao diện tương tác
- **Sidebar điều khiển**: Chọn phương pháp xử lý dữ liệu và mô hình
- **Main area**: Hiển thị kết quả chi tiết với biểu đồ và bảng số liệu

### Phương pháp xử lý dữ liệu mất cân bằng
1. **Dữ liệu gốc (Imbalanced)**: Không xử lý, tỷ lệ gian lận 0.172%
2. **Random Oversampling**: Lặp lại các mẫu gian lận để cân bằng
3. **SMOTE**: Tạo mẫu gian lận tổng hợp bằng nội suy

### Các mô hình Machine Learning
1. **Hồi quy Logistic** (Logistic Regression)
2. **Cây quyết định** (Decision Tree)
3. **Mạng Bayesian** (Gaussian Naive Bayes)

### Đánh giá và Trực quan hóa
- **Confusion Matrix**: Heatmap chi tiết cho từng mô hình
- **Metrics**: TP, FP, TN, FN, Accuracy, Precision, Recall, F1-Score
- **Bảng so sánh**: So sánh tất cả mô hình cùng lúc
- **Biểu đồ**: So sánh F1-Score và Recall
- **Khuyến nghị**: Tự động đề xuất mô hình tốt nhất

---

## Công nghệ sử dụng

- **Python**: 3.11 (hoặc 3.8+)
- **Streamlit**: Giao diện web tương tác
- **Scikit-learn**: Các mô hình Machine Learning
- **Imbalanced-learn**: Xử lý dữ liệu mất cân bằng (SMOTE, Oversampling)
- **Pandas & NumPy**: Xử lý dữ liệu
- **Matplotlib & Seaborn**: Trực quan hóa

---

## Cấu trúc dự án

```
fraud_detection_demo/
│
├── app.py                      # File chính - Ứng dụng Streamlit
├── requirements.txt            # Dependencies
├── README.md                   # Tài liệu hướng dẫn
│
├── src/                        # Package chứa các module
│   ├── __init__.py
│   ├── data_processing.py      # Module xử lý dữ liệu
│   ├── models.py               # Module các mô hình ML
│   └── visualization.py        # Module trực quan hóa
│
└── data/                       # Thư mục lưu dữ liệu (tùy chọn)
```

### Mô tả các module

#### `src/data_processing.py`
- Tải dữ liệu từ nguồn công khai (GitHub/Kaggle)
- Chuẩn bị và chia dữ liệu (train/test split)
- Xử lý dữ liệu mất cân bằng với Oversampling và SMOTE
- Cung cấp thông tin thống kê về dữ liệu

#### `src/models.py`
- Class `FraudDetectionModel`: Lớp cơ sở cho các mô hình
- Implement 3 mô hình: Logistic Regression, Decision Tree, Bayesian Network
- Hàm huấn luyện, dự đoán và đánh giá mô hình
- Factory pattern để tạo mô hình

#### `src/visualization.py`
- Vẽ Confusion Matrix dưới dạng heatmap
- Tạo bảng so sánh metrics
- Hiển thị metrics dưới dạng cards
- Vẽ biểu đồ so sánh các mô hình
- Tạo khuyến nghị tự động

---

## Cài đặt

### Bước 1: Clone hoặc tải xuống dự án

```bash
# Nếu có Git
cd fraud_detection_demo

# Hoặc tải file ZIP và giải nén
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Kiểm tra cài đặt

```bash
pip list
```

Đảm bảo các package sau đã được cài đặt:
- streamlit
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

---

## Sử dụng

### Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:8501`

### Hướng dẫn sử dụng giao diện

1. **Chọn phương pháp xử lý dữ liệu** ở sidebar:
   - Dữ liệu gốc (Mất cân bằng)
   - Xử lý bằng Oversampling
   - Xử lý bằng SMOTE

2. **Chọn một hoặc nhiều mô hình** để huấn luyện:
   - Hồi quy Logistic
   - Cây quyết định
   - Mạng Bayesian

3. **Nhấn nút "Huấn luyện và Đánh giá"**

4. **Xem kết quả**:
   - Thông tin dữ liệu (gốc và sau xử lý)
   - Kết quả chi tiết từng mô hình với Confusion Matrix
   - Bảng so sánh tổng hợp
   - Biểu đồ so sánh
   - Khuyến nghị mô hình tốt nhất

---

## Dataset

### Nguồn dữ liệu
- **Tên**: Credit Card Fraud Detection
- **Nguồn**: Kaggle - [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **URL tự động tải**: Dataset được tải tự động từ GitHub khi chạy ứng dụng

### Thông tin dataset
- **Tổng số giao dịch**: 284,807
- **Giao dịch gian lận**: 492 (0.172%)
- **Giao dịch hợp pháp**: 284,315 (99.828%)
- **Số features**: 30 (V1-V28 từ PCA, Time, Amount)
- **Target**: Class (0 = hợp pháp, 1 = gian lận)

### Đặc điểm
- Dữ liệu đã được xử lý PCA để bảo mật
- Dữ liệu **rất mất cân bằng** (imbalanced)
- Chỉ có 2 features gốc: `Time` và `Amount`
- Các features khác (V1-V28) đã được transform bằng PCA

---

## Phương pháp

### 1. Xử lý dữ liệu mất cân bằng

#### Vấn đề
Dataset có tỷ lệ gian lận chỉ 0.172%, gây khó khăn cho việc huấn luyện mô hình.

#### Giải pháp

**a) Random Oversampling**
- Lặp lại ngẫu nhiên các mẫu minority class (gian lận)
- Ưu điểm: Đơn giản, nhanh
- Nhược điểm: Có thể gây overfitting

**b) SMOTE (Synthetic Minority Oversampling Technique)**
- Tạo mẫu tổng hợp bằng nội suy giữa các mẫu gần nhau
- Ưu điểm: Tạo dữ liệu đa dạng, giảm overfitting
- Nhược điểm: Tốn thời gian tính toán hơn

### 2. Các mô hình Machine Learning

#### Logistic Regression
- Mô hình tuyến tính sử dụng hàm sigmoid
- Phù hợp cho phân loại nhị phân
- Nhanh và hiệu quả

#### Decision Tree
- Mô hình phi tuyến tạo cấu trúc cây quyết định
- Dễ hiểu và giải thích
- Có thể bị overfitting nếu không điều chỉnh

#### Bayesian Network (Gaussian Naive Bayes)
- Dựa trên định lý Bayes
- Giả định các features độc lập
- Nhanh và hiệu quả với dữ liệu lớn

### 3. Phương pháp đánh giá

#### Confusion Matrix
```
                Predicted
                 0    1
Actual  0      TN   FP
        1      FN   TP
```

#### Các chỉ số
- **True Positive (TP)**: Gian lận phát hiện đúng ✅
- **False Positive (FP)**: Hợp pháp nhận diện nhầm ❌
- **True Negative (TN)**: Hợp pháp phát hiện đúng ✅
- **False Negative (FN)**: Gian lận bị bỏ sót ❌

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

---

## Kết quả mẫu

### Kết quả điển hình với SMOTE

| Metric | Value |
|--------|-------|
| True Positive (TP) | ~140 |
| False Positive (FP) | ~500 |
| Accuracy | ~99.4% |
| Precision | ~22% |
| Recall | ~95% |
| F1-Score | ~36% |

**Giải thích**:
- Mô hình phát hiện được 95% giao dịch gian lận (Recall cao)
- Tuy nhiên có ~500 giao dịch hợp pháp bị nhận diện nhầm (FP)
- Trade-off giữa phát hiện gian lận và tránh nhận diện nhầm

---

## Tài liệu tham khảo

1. **Dataset**: Kaggle - Credit Card Fraud Detection
2. **SMOTE**: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
4. **Imbalanced Learning**: He & Garcia (2009) - "Learning from Imbalanced Data"

---


## Ghi chú

### Yêu cầu hệ thống
- Python 3.8 trở lên (khuyến nghị 3.11)
- RAM: Tối thiểu 4GB (khuyến nghị 8GB)
- Kết nối internet (để tải dữ liệu lần đầu)

### Lưu ý khi sử dụng
- Lần chạy đầu tiên có thể mất thời gian để tải dataset
- Với dataset lớn, quá trình huấn luyện có thể mất 1-2 phút

### Khắc phục sự cố

**Lỗi khi tải dữ liệu:**
```python
# Nếu không tải được tự động, tải thủ công từ Kaggle và đặt vào thư mục data/
# Cập nhật đường dẫn trong data_processing.py
```

**Lỗi import module:**
```bash
# Đảm bảo bạn đang ở đúng thư mục
cd fraud_detection_demo
streamlit run app.py
```

---

## 👨‍💻 Tác giả

- **Đồ án**: Phát hiện Gian lận Thẻ Tín dụng bằng Machine Learning
- **Công nghệ**: Python 3.14 | Streamlit | Scikit-learn
- **Năm**: 2025

---

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

## 🎉 Kết luận

Ứng dụng này cung cấp một giao diện trực quan và dễ sử dụng để demo các kỹ thuật Machine Learning 
trong bài toán phát hiện gian lận. Người dùng có thể dễ dàng so sánh hiệu quả của các phương pháp 
khác nhau và hiểu rõ hơn về cách xử lý dữ liệu mất cân bằng.

