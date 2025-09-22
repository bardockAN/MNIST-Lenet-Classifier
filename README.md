# MNIST LeNet Classifier

Ứng dụng phân loại chữ số viết tay MNIST sử dụng mạng LeNet với PyTorch và giao diện web Streamlit.

## 📋 Mô tả dự án

Dự án này triển khai mạng neural tích chập LeNet để phân loại các chữ số viết tay từ tập dữ liệu MNIST. Ứng dụng bao gồm:

- **Huấn luyện mô hình**: Triển khai kiến trúc LeNet hoàn chỉnh với PyTorch
- **Giao diện web**: Ứng dụng Streamlit cho phép upload và dự đoán ảnh real-time
- **Độ chính xác cao**: Đạt được độ chính xác > 98% trên tập test MNIST

## 🏗️ Kiến trúc mô hình

### LeNet Architecture
```
Input (1x28x28) 
    ↓
Conv2D (6 filters, 5x5, padding='same') + AvgPool2D(2x2) + ReLU
    ↓  
Conv2D (16 filters, 5x5) + AvgPool2D(2x2) + ReLU
    ↓
Flatten 
    ↓
Linear (400 → 120)
    ↓
Linear (120 → 84) 
    ↓
Linear (84 → 10 classes)
```

### Đặc điểm kỹ thuật:
- **Input**: Ảnh grayscale 28x28 pixels
- **Convolutional layers**: 2 lớp với ReLU activation
- **Pooling**: Average pooling 2x2
- **Fully connected**: 3 lớp dense layers
- **Output**: 10 classes (chữ số 0-9)

## 🚀 Cài đặt và Chạy

### 1. Clone repository
```bash
git clone https://github.com/bardockAN/MNIST-Lenet-Classifier.git
cd CNN-applications
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# hoặc
source venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Huấn luyện mô hình
```bash
python train.py
```

### 5. Chạy ứng dụng web
```bash
streamlit run app.py
```

Mở trình duyệt và truy cập: `http://localhost:8501`

## 📁 Cấu trúc dự án

```
MNIST-Lenet-Classifier/
├── train.py            # Script huấn luyện mô hình LeNet
├── app.py              # Ứng dụng Streamlit web interface  
├── requirements.txt    # Dependencies Python cần thiết
├── README.md           # Tài liệu hướng dẫn
├── .gitignore          # Loại trừ files không cần thiết
├── model/              # Thư mục lưu model đã train (tự tạo)
│   └── lenet_model.pt  # Model weights đã huấn luyện
└── data/               # Dataset MNIST (tự download)
    └── MNIST/
        ├── processed/
        └── raw/
```

## 🔧 Chi tiết kỹ thuật

### Hyperparameters mặc định:
- **Batch size**: 256
- **Optimizer**: Adam (learning rate mặc định)
- **Epochs**: 10
- **Train/Validation split**: 90/10
- **Loss function**: CrossEntropyLoss
- **Device**: Tự động detect CUDA/CPU

### Data preprocessing:
- **Normalization**: Tính mean và std từ training data
- **Transforms**: ToTensor + Normalize 
- **Image size**: 28x28 grayscale
- **Dataset split**: Random split cho validation

## 📊 Kết quả

- **Train Accuracy**: ~99%  
- **Validation Accuracy**: ~98%
- **Test Accuracy**: ~98%
- **Training time**: ~2-3 phút (CPU), ~30 giây (GPU)
- **Model size**: ~61KB (lenet_model.pt)

### Training progress:
- Model tự động lưu best weights dựa trên validation loss
- Training có gradient clipping (0.1) để stability
- Real-time monitoring accuracy và loss mỗi epoch

## 🖼️ Sử dụng ứng dụng

1. **Chạy Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Mở trình duyệt**: Truy cập `http://localhost:8501`

3. **Upload ảnh**: Chọn file PNG, JPG, hoặc JPEG

4. **Xem kết quả**: Mô hình sẽ hiển thị:
   - Chữ số được dự đoán (0-9)
   - Độ tin cậy (confidence score)

### Lưu ý quan trọng:
- **Ảnh input**: Nên có nền tối, chữ số màu sáng (giống MNIST)
- **Kích thước**: Ảnh sẽ tự động resize về 28x28
- **Format**: Tự động convert sang grayscale
- **Model file**: Đảm bảo có file `lenet_model.pt` trong thư mục gốc

## 🛠️ Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
streamlit>=1.0.0
numpy>=1.21.0
Pillow>=8.3.0
torchsummary>=1.5.1
matplotlib>=3.4.0
```

## � Cải tiến có thể

- [ ] **Data augmentation**: Rotation, scaling, noise cho robust training
- [ ] **Regularization**: Dropout layers để giảm overfitting  
- [ ] **Learning rate scheduling**: Adaptive learning rate
- [ ] **Advanced architectures**: CNN modern như ResNet, EfficientNet
- [ ] **Real-time drawing**: Canvas để vẽ chữ số trực tiếp trên web
- [ ] **Model optimization**: Quantization để giảm kích thước model
- [ ] **Batch prediction**: Upload multiple images cùng lúc
- [ ] **Confidence visualization**: Histogram của all class probabilities

## 🐛 Troubleshooting

### Lỗi thường gặp:

**1. ModuleNotFoundError**
```bash
# Kiểm tra virtual environment đã activate chưa
# Cài đặt lại dependencies
pip install -r requirements.txt
```

**2. Model file not found**
```bash  
# Đảm bảo đã train model trước
python train.py
# Hoặc copy lenet_model.pt từ thư mục model/
cp model/lenet_model.pt .
```

**3. CUDA out of memory**
```python
# Giảm batch_size trong train.py
BATCH_SIZE = 128  # thay vì 256
```

**4. Streamlit app không chạy**
```bash
# Kiểm tra port 8501 có bị chiếm không
netstat -an | findstr 8501
# Chạy trên port khác
streamlit run app.py --server.port 8502
```

### Debug commands:
```bash
# Kiểm tra PyTorch version
python -c "import torch; print(torch.__version__)"

# Kiểm tra CUDA availability  
python -c "import torch; print(torch.cuda.is_available())"

# Test model loading
python -c "import torch; torch.load('lenet_model.pt', map_location='cpu')"
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Author

**Bùi Đặng Quốc An**
- GitHub: [@bardockAN](https://github.com/bardockAN)  
- Email: 04anbui@gmail.com
- Repository: [MNIST-Lenet-Classifier](https://github.com/bardockAN/MNIST-Lenet-Classifier)

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Streamlit](https://streamlit.io/) - Web application framework  
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Handwritten digits dataset
- [LeNet Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) - Original LeNet architecture by Yann LeCun

## 📈 Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/bardockAN/MNIST-Lenet-Classifier)
![GitHub stars](https://img.shields.io/github/stars/bardockAN/MNIST-Lenet-Classifier)
![GitHub forks](https://img.shields.io/github/forks/bardockAN/MNIST-Lenet-Classifier)

---

⭐ **Star this repo nếu dự án hữu ích cho bạn!**

📋 **Contributions welcome!** - Tạo issue hoặc pull request để đóng góp