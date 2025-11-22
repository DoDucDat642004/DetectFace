import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO

from emotion.load_emotion_model import load_model as load_emotion_model
from gender_race_age.load_gra_model import load_model as load_gra_model
from emotion.predict import predict_from_image as predict_emotion_from_image
from gender_race_age.predict import predict_from_image as predict_gra_from_image
from emotion.transforms_image import transforms_image as emotion_transforms_image
from gender_race_age.transform_image import transforms_image as gra_transforms_image
from analysis import analyze_people

# =============================
# CÁC HÀM HỖ TRỢ (HELPER)
# =============================

def load_image(image_numpy_array, transform_func):
    """Chuyển đổi ảnh từ OpenCV (BGR) sang Tensor để đưa vào model."""
    # Chuyển BGR -> RGB
    img_rgb = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2RGB)
    # Chuyển sang PIL Image
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    # Transform và thêm batch dimension [1, C, H, W]
    return transform_func(img_pil).unsqueeze(0)

def compute_iou(b1, b2):
    """Tính chỉ số IOU để so khớp khuôn mặt giữa các frame."""
    xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
    boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# =============================
# HÀM CHÍNH XỬ LÝ VIDEO
# =============================

def process_video_file(input_path, output_path="output_result.mp4"):
    # --- Cấu hình thiết bị ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # --- Load Models ---
    print("Đang tải các model...")
    # 1. Model Detect khuôn mặt (YOLO)
    yolo_model = YOLO(r"/Users/dodat/Documents/DetectFace/yolo/model/yolo11n_openvino_model/")
    
    # 2. Model Cảm xúc
    emotion_model, idx_to_label = load_emotion_model("./emotion/model/best_emotion.pth")
    emotion_model.to(device)
    
    # 3. Model GRA (Gender/Race/Age)
    gra_model = load_gra_model("./gender_race_age/model/best_model.pt")
    gra_model.to(device)

    # --- Mở Video Input ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file video tại '{input_path}'")
        return

    # --- Cấu hình Video Output ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 10 

    print(f"Thông tin Video: {width}x{height} | {fps} FPS")
    
    # Codec cho mp4
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Kiểm tra xem Writer có khởi tạo thành công không
        if not out.isOpened():
            print("Cảnh báo: 'avc1' không hoạt động, thử lại với 'mp4v'...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except Exception as e:
        print(f"Lỗi khởi tạo VideoWriter: {e}")
        return

    # --- Biến theo dõi (Tracking) ---
    people_data = defaultdict(lambda: {"emotion": [], "gender": [], "race": [], "age": []})
    next_person_id = 0
    tracked_faces = [] # List lưu [(id, bbox), ...]

    print("Bắt đầu xử lý... Nhấn 'q' trên cửa sổ xem trước để dừng sớm.")

    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Đã xử lý hết video.")
            break

        frame_count += 1

        # 1. YOLO Detect
        results = yolo_model.predict(source=frame, conf=0.5, save=False, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

        for box in detections:
            # Lấy tọa độ gốc từ YOLO
            x1, y1, x2, y2 = map(int, box[:4])

            # ---------------------------------------------------------
            # [QUAN TRỌNG] CHIẾN LƯỢC PADDING (MỞ RỘNG VÙNG CẮT)
            # Giúp lấy trọn vẹn tóc, cằm để model GRA/Emotion chuẩn hơn
            # ---------------------------------------------------------
            w_box = x2 - x1
            h_box = y2 - y1
            
            # Mở rộng 20-25% mỗi bên
            pad_w = int(w_box * 0.25) 
            pad_h = int(h_box * 0.25)

            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(width, x2 + pad_w)
            y2_pad = min(height, y2 + pad_h)
            # ---------------------------------------------------------

            # 2. Tracking (Gán ID)
            # Dùng tọa độ gốc để tracking cho chính xác vị trí
            pid = None
            for saved_id, saved_box in tracked_faces:
                if compute_iou((x1, y1, x2, y2), saved_box) > 0.4: # Ngưỡng IOU 0.4
                    pid = saved_id
                    break
            
            if pid is None:
                pid = next_person_id
                next_person_id += 1
                tracked_faces.append((pid, (x1, y1, x2, y2)))
            else:
                # Cập nhật vị trí mới cho ID cũ
                for i, (tid, _) in enumerate(tracked_faces):
                    if tid == pid:
                        tracked_faces[i] = (pid, (x1, y1, x2, y2))

            # 3. Cắt ảnh & Dự đoán
            # Cắt từ vùng đã Padding (x1_pad...)
            face_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_crop.size == 0: continue

            # --- Dự đoán Emotion ---
            emo_tensor = load_image(face_crop, emotion_transforms_image).to(device)
            emotion_pred, _, emotions, probs = predict_emotion_from_image(
                emotion_model, emo_tensor, idx_to_label, top_k=len(idx_to_label), image_tensor=True
            )

            # --- Dự đoán GRA ---
            gra_tensor = load_image(face_crop, gra_transforms_image).to(device)
            gra_result = predict_gra_from_image(gra_model, gra_tensor, image_tensor=True)

            # 4. Lưu dữ liệu thống kê
            people_data[pid]["emotion"].append(emotion_pred)
            people_data[pid]["gender"].append(gra_result["gender"])
            people_data[pid]["race"].append(gra_result["race"])
            people_data[pid]["age"].append(gra_result["age"])

            # 5. Vẽ lên Frame (Visualization)
            # Vẽ Box bao quanh mặt (Box gốc cho đẹp)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ thông tin text
            info_text = f"ID:{pid} | {emotion_pred} | {gra_result['gender']} | {gra_result['age']}"
            cv2.putText(frame, info_text, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Vẽ thanh xác suất cảm xúc (Bar chart)
            bar_x = x2 + 5
            bar_y = y1
            bar_h = 15
            gap = 4
            for i, (emo, conf) in enumerate(zip(emotions, probs)):
                yy = bar_y + i * (bar_h + gap)
                # Khung xám
                cv2.rectangle(frame, (bar_x, yy), (bar_x + 100, yy + bar_h), (100, 100, 100), 1)
                # Thanh màu xanh
                cv2.rectangle(frame, (bar_x, yy), (bar_x + int(100 * conf), yy + bar_h), (0, 255, 0), -1)
                # Chữ bên cạnh
                cv2.putText(frame, f"{emo[:3]}", (bar_x + 105, yy + 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 6. Ghi và Hiển thị
        out.write(frame) # Lưu vào file
        
        # Hiển thị
        display_frame = cv2.resize(frame, (1280, 720)) if width > 1920 else frame
        cv2.imshow("Video Processing", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Người dùng đã dừng thủ công.")
            break

    # --- Dọn dẹp ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Đã lưu video kết quả tại: {output_path}")

    # --- Chạy phân tích thống kê ---
    analyze_people(people_data)

# =============================
# CHẠY CHƯƠNG TRÌNH
# =============================
if __name__ == "__main__":
    VIDEO_INPUT = "./input/videos/video.mp4"  
    
    VIDEO_OUTPUT = "./output/video/output_video.mp4"
    
    if os.path.exists(VIDEO_INPUT):
        process_video_file(VIDEO_INPUT, VIDEO_OUTPUT)