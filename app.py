import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

# 1. Định nghĩa lớp mô hình (Model Class Definition)
# Mã code này cần phải giống hệt lớp LeNetClassifier bạn đã huấn luyện.
class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding='same'
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs

# 2. Định nghĩa hàm tiền xử lý ảnh (Image Preprocessing Function)
def preprocess_image(image, mean, std):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    return transform(image).unsqueeze(0)

# 3. Tải mô hình và thực hiện dự đoán (Load Model and Predict)
def predict_image(model, image, device, mean=0.1307, std=0.3081):
    model.eval()
    img_tensor = preprocess_image(image, mean, std).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    return predicted_class, confidence

# Cấu hình thiết bị (CPU/GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải mô hình đã huấn luyện
try:
    lenet_model = LeNetClassifier(num_classes=10)
    lenet_model.load_state_dict(torch.load('lenet_model.pt', map_location=device))
    lenet_model.to(device)
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy tệp mô hình 'lenet_model.pt'. Hãy chắc chắn rằng bạn đã đặt nó vào cùng thư mục với tệp 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()


# 4. Tạo giao diện ứng dụng Streamlit (Streamlit App Interface)
st.title("Ứng dụng phân loại ảnh MNIST")
st.write("Tải lên một hình ảnh chữ số viết tay để mô hình dự đoán.")

uploaded_file = st.file_uploader("Chọn một tệp ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    image = Image.open(uploaded_file).convert("L")  # Chuyển đổi sang ảnh grayscale
    st.image(image, caption='Hình ảnh đã tải lên', use_column_width=True)
    st.write("")
    st.write("Đang dự đoán...")

    # Dự đoán và hiển thị kết quả
    predicted_class, confidence = predict_image(lenet_model, image, device)
    st.success(f"Kết quả dự đoán: **{predicted_class}**")
    st.write(f"Độ tin cậy: **{confidence:.2f}**")

# Chú thích
st.markdown("---")
st.write("### Ghi chú")
st.write("Để ứng dụng này chạy được, bạn cần đảm bảo:")
st.write("1.  Có tệp `lenet_model.pt` trong cùng thư mục với `app.py`.")
st.write("2.  Đã cài đặt tất cả các thư viện cần thiết trong `requirements.txt`.")