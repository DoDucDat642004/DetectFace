from emotion.packages import *

# Kích thước ảnh đầu vào
emotion_image_size = 112

transforms_image = transforms.Compose([
    # Ảnh xám (Grayscale)
    transforms.Grayscale(num_output_channels=1),

    # Resize ảnh
    transforms.Resize((emotion_image_size, emotion_image_size)),

    # Chuyển đổi ảnh PIL/NumPy (0-255) sang Tensor (0.0-1.0)
    transforms.ToTensor(),

    # Chuẩn hóa (Normalize)
    transforms.Normalize(mean=[0.5], std=[0.5]),
])