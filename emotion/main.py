from emotion.packages import *
from emotion.device import *
from emotion.load_emotion_model import load_model
from emotion.predict import predict_from_image

# Cấu hình
YOLO_MODEL_PATH = "./yolo/model/yolo11n_openvino_model/"
EMOTION_MODEL_PATH = "./emotion/model/best_emotion.pth"

def predict_single_url_with_crop(url, emotion_model, yolo_model, idx_to_label):
    try:
        # Gửi request
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Kiểm tra dữ liệu
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ValueError(f"URL không trả về ảnh. Content-Type: {content_type}")

        # Mở ảnh
        img_pil = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img_pil) 
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    except (UnidentifiedImageError, ValueError) as e:
        print(f"Không thể mở ảnh từ URL: {url}\nLý do: {e}")
        return None, None
    except Exception as e:
        print(f"Lỗi khi tải ảnh: {e}")
        return None, None

    # Phát hiện khuôn mặt
    results = yolo_model(img_bgr, verbose=False)
    
    # Lấy các bounding box
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        print("Không tìm thấy khuôn mặt nào trong ảnh!")
        return

    print(f"Tìm thấy {len(boxes)} khuôn mặt.")

    # Duyệt qua từng mặt, cắt ra và dự đoán
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        h, w = img_bgr.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # CẮT KHUÔN MẶT
        face_crop = img_pil.crop((x1, y1, x2, y2))
        
        
        pred_label, pred_conf, _, probs = predict_from_image(
            emotion_model, 
            face_crop, # Truyền ảnh mặt đã cắt
            idx_to_label, 
            image_tensor=False
        )
        
        print(f"Face {i+1}: {pred_label} ({pred_conf:.2f})")

if __name__ == "__main__":
    # Load Models
    device = get_device()
    
    # Load Emotion
    emo_model, idx_to_label = load_model(EMOTION_MODEL_PATH)
    emo_model.to(device)
    
    # Load YOLO
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    test_url = "https://img.lovepik.com/photo/60177/2960.jpg_wh860.jpg"
    
    if emo_model:
        predict_single_url_with_crop(test_url, emo_model, yolo_model, idx_to_label)

# Run : python -m emotion.main