import os
import cv2
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO

# Giữ nguyên các import từ code của bạn
from emotion.load_emotion_model import load_model as load_emotion_model
from gender_race_age.load_gra_model import load_model as load_gra_model
from emotion.predict import predict_from_image as predict_emotion_from_image
from gender_race_age.predict import predict_from_image as predict_gra_from_image
from emotion.transforms_image import transforms_image as emotion_transforms_image
from gender_race_age.transform_image import transforms_image as gra_transforms_image
from analysis import analyze_people

# =============================
# CÁC HÀM HỖ TRỢ
# =============================

def load_image_tensor(image_numpy_array, transform_func):
    """Chuyển đổi ảnh từ OpenCV (BGR) sang Tensor."""
    img_rgb = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    return transform_func(img_pil).unsqueeze(0)

# =============================
# HÀM CHÍNH XỬ LÝ ẢNH
# =============================

def process_image_file(input_path, output_path="output_image.jpg"):
    # --- 1. Cấu hình thiết bị ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # --- 2. Load Models ---
    print("Đang tải các model...")
    try:
        # Load model
        yolo_model = YOLO(r"/Users/dodat/Documents/DetectFace/yolo/model/yolo11n_openvino_model/")
        
        emotion_model, idx_to_label = load_emotion_model("./emotion/model/best_emotion.pth")
        emotion_model.to(device)
        
        gra_model = load_gra_model("./gender_race_age/model/best_model.pt")
        gra_model.to(device)
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return

    # --- 3. Đọc ảnh đầu vào ---
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại '{input_path}'")
        return

    frame = cv2.imread(input_path)
    if frame is None:
        print("Lỗi: Không thể đọc file ảnh (file lỗi hoặc không đúng định dạng).")
        return

    height, width = frame.shape[:2]
    print(f"Kích thước ảnh: {width}x{height}")

    # --- 4. Detect & Predict ---
    # Biến lưu dữ liệu để chạy hàm analysis sau này
    people_data = defaultdict(lambda: {"emotion": [], "gender": [], "race": [], "age": []})

    # YOLO Detect
    results = yolo_model.predict(source=frame, conf=0.5, save=False, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    print(f"Tìm thấy {len(detections)} khuôn mặt.")

    for idx, box in enumerate(detections):
        # Với ảnh tĩnh, ID chỉ đơn giản là số thứ tự phát hiện (0, 1, 2...)
        pid = idx 

        x1, y1, x2, y2 = map(int, box[:4])

        # --- Chiến lược Padding ---
        w_box = x2 - x1
        h_box = y2 - y1
        pad_w = int(w_box * 0.25) 
        pad_h = int(h_box * 0.25)

        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(width, x2 + pad_w)
        y2_pad = min(height, y2 + pad_h)

        # Cắt ảnh khuôn mặt
        face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if face_crop.size == 0: continue

        # --- Dự đoán Emotion ---
        emo_tensor = load_image_tensor(face_crop, emotion_transforms_image).to(device)
        emotion_pred, _, emotions, probs = predict_emotion_from_image(
            emotion_model, emo_tensor, idx_to_label, top_k=len(idx_to_label), image_tensor=True
        )

        # --- Dự đoán GRA ---
        gra_tensor = load_image_tensor(face_crop, gra_transforms_image).to(device)
        gra_result = predict_gra_from_image(gra_model, gra_tensor, image_tensor=True)

        # Lưu dữ liệu thống kê (mỗi người chỉ có 1 bản ghi vì là ảnh tĩnh)
        people_data[pid]["emotion"].append(emotion_pred)
        people_data[pid]["gender"].append(gra_result["gender"])
        people_data[pid]["race"].append(gra_result["race"])
        people_data[pid]["age"].append(gra_result["age"])

        # --- Vẽ lên ảnh ---
        # Vẽ Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vẽ Text thông tin
        info_text = f"ID:{pid}|{emotion_pred}|{gra_result['gender']}|{gra_result['age']}"
        # Tính toán vị trí vẽ text để không bị tràn ra ngoài ảnh
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        cv2.putText(frame, info_text, (x1, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Vẽ biểu đồ xác suất cảm xúc
        bar_x = x2 + 5
        # Nếu thanh bar tràn ra lề phải thì vẽ sang bên trái
        if bar_x + 100 > width:
            bar_x = x1 - 110
        
        bar_y = y1
        bar_h = 15
        gap = 4
        for i, (emo, conf) in enumerate(zip(emotions, probs)):
            yy = bar_y + i * (bar_h + gap)
            # Đảm bảo không vẽ ra ngoài chiều cao ảnh
            if yy + bar_h > height: break 

            cv2.rectangle(frame, (bar_x, yy), (bar_x + 100, yy + bar_h), (100, 100, 100), 1)
            cv2.rectangle(frame, (bar_x, yy), (bar_x + int(100 * conf), yy + bar_h), (0, 255, 0), -1)
            cv2.putText(frame, f"{emo[:3]}", (bar_x + 105, yy + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # --- 5. Lưu và Hiển thị kết quả ---
    cv2.imwrite(output_path, frame)
    print(f"Đã lưu ảnh kết quả tại: {output_path}")

    # Hiển thị ảnh (Resize nếu ảnh quá to để xem trên màn hình)
    display_h, display_w = frame.shape[:2]
    max_dim = 800
    if display_h > max_dim or display_w > max_dim:
        scale = max_dim / max(display_h, display_w)
        frame_show = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        frame_show = frame
    
    cv2.imshow("Image Result", frame_show)
    print("Nhấn phím bất kỳ để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 6. Phân tích thống kê (Optional) ---
    print("\n--- THỐNG KÊ ---")
    analyze_people(people_data)

# =============================
# CHẠY CHƯƠNG TRÌNH
# =============================
if __name__ == "__main__":
    # Đường dẫn file ảnh đầu vào
    IMAGE_INPUT = "./input/images/image.jpg"
    
    # Đường dẫn file ảnh đầu ra
    IMAGE_OUTPUT = "./output/image/output_image.jpg"
    
    process_image_file(IMAGE_INPUT, IMAGE_OUTPUT)