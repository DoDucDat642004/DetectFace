import os
import cv2
import time
import torch
from PIL import Image
from collections import defaultdict

from ultralytics import YOLO

# =============================
# IMPORT MODEL PHỤ
# =============================
from emotion.load_emotion_model import load_model as load_emotion_model
from gender_race_age.load_gra_model import load_model as load_gra_model
from emotion.predict import predict_from_image as predict_emotion_from_image
from gender_race_age.predict import predict_from_image as predict_gra_from_image
from emotion.transforms_image import transforms_image as emotion_transforms_image
from gender_race_age.transform_image import transforms_image as gra_transforms_image
from analysis import analyze_people

# =============================
# HÀM TIỆN ÍCH
# =============================
def load_image(image_numpy_array, transform_func):
    """Chuyển numpy BGR -> tensor sau transform."""
    img = Image.fromarray(cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2RGB)).convert("RGB")
    return transform_func(img).unsqueeze(0)

# =============================
# HÀM CHÍNH
# =============================
def detect_and_predict(video_source=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    yolo_model = YOLO(r"/Users/dodat/Documents/DetectFace/yolo/model/yolo11n_openvino_model/")
    emotion_model, idx_to_label = load_emotion_model("./emotion/model/best_emotion.pth")
    gra_model = load_gra_model("./gender_race_age/model/best_model.pt")

    # --- Khởi tạo camera ---
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Không mở được camera.")
        return

    print("YOLOv8-Face + Emotion Detection đang chạy... (Nhấn 'q' để thoát)")
    people_data = defaultdict(lambda: {"emotion": [], "gender": [], "race": [], "age": []})
    next_person_id = 0
    tracked_faces = []

    # =============================
    # HÀM PHỤ: IOU & ID tracking
    # =============================
    def compute_iou(b1, b2):
        xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
        boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return interArea / (boxAArea + boxBArea - interArea + 1e-6)

    def find_existing_face(x1, y1, x2, y2):
        for pid, (px1, py1, px2, py2) in tracked_faces:
            if compute_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.3:
                return pid
        return None

    # =============================
    # VÒNG LẶP REALTIME
    # =============================
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(source=frame, conf=0.5, save=False, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            pid = find_existing_face(x1, y1, x2, y2)
            if pid is None:
                pid = next_person_id
                next_person_id += 1
                tracked_faces.append((pid, (x1, y1, x2, y2)))
            else:
                for i, (tid, _) in enumerate(tracked_faces):
                    if tid == pid:
                        tracked_faces[i] = (pid, (x1, y1, x2, y2))

            face_crop = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
            if face_crop.size == 0:
                continue

            # --- Dự đoán cảm xúc ---
            emo_tensor = load_image(face_crop, emotion_transforms_image).to(device)
            emotion_pred, _, emotions, probs = predict_emotion_from_image(
                emotion_model, emo_tensor, idx_to_label, top_k=len(idx_to_label), image_tensor=True
            )

            # --- Dự đoán giới tính/chủng tộc/tuổi ---
            gra_tensor = load_image(face_crop, gra_transforms_image).to(device)
            gra_result = predict_gra_from_image(gra_model, gra_tensor, image_tensor=True)

            # --- Lưu dữ liệu ---
            people_data[pid]["emotion"].append(emotion_pred)
            people_data[pid]["gender"].append(gra_result["gender"])
            people_data[pid]["race"].append(gra_result["race"])
            people_data[pid]["age"].append(gra_result["age"])

            # --- Vẽ kết quả ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {pid} | {emotion_pred} | {gra_result['gender']} | {gra_result['age']} | {gra_result['race']}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            bar_x = x2 + 10  # Vẽ bên phải khuôn mặt.
            bar_y = y1
            bar_width = 120  # Chiều rộng thanh.
            bar_height = 20  # Chiều cao mỗi thanh.
            gap = 5  # Khoảng cách giữa các thanh.

            for i, (emo, conf) in enumerate(zip(emotions, probs)):
                y = bar_y + i * (bar_height + gap)
                # Vẽ khung xám viền ngoài.
                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (100, 100, 100), 1)
                # Vẽ phần xanh theo tỷ lệ xác suất.
                cv2.rectangle(frame, (bar_x, y),
                                (bar_x + int(bar_width * conf), y + bar_height),
                                (0, 255, 0), -1)
                # Ghi tên cảm xúc và % xác suất bên phải.
                cv2.putText(frame, f"{emo} {conf*100:.1f}%",
                            (bar_x + bar_width + 10, y + bar_height - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        fps = 1 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Total people: {len(people_data)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLOv8-Face + Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    analyze_people(people_data)

# =============================
# CHẠY CHƯƠNG TRÌNH
# =============================
if __name__ == "__main__":
    detect_and_predict(0)
