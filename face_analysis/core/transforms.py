import cv2
from PIL import Image

def image_to_tensor(face_crop, transform):
    """
    Chuyển đổi ảnh cắt từ OpenCV (NumPy array) sang PyTorch Tensor.
    """
    # Chuyển hệ màu(BGR->RGB): OpenCV mặc định dùng BGR, nhưng các model AI (PyTorch/PIL) được huấn luyện trên ảnh RGB.
    # Chuyển đổi format: Từ NumPy array (của OpenCV) sang PIL Image object.
    img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    
    # Transform Image
    return transform(img)