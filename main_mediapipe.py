# =============================
# IMPORT THƯ VIỆN CẦN THIẾT
# =============================

import cv2  # Đọc video/camera, xử lý ma trận ảnh, vẽ và hiển thị cửa sổ bằng OpenCV.
import mediapipe as mp  # Sử dụng MediaPipe để phát hiện khuôn mặt realtime.
from PIL import Image  # Dùng Pillow để chuyển numpy array thành ảnh PIL (cho transform model).
import torch  # PyTorch - dùng để load model và chạy inference trên CPU/GPU.
from collections import defaultdict, Counter  # defaultdict tiện cho lưu dữ liệu có giá trị mặc định.
import matplotlib.pyplot as plt  # Vẽ biểu đồ, dùng trong phần phân tích dữ liệu.
import time

# =============================
# IMPORT MODEL VÀ HÀM PHỤ
# =============================

# Import hàm load model cảm xúc
from emotion.load_emotion_model import load_model as load_emotion_model

# Import hàm load model đa nhiệm (giới tính, chủng tộc, tuổi)
from gender_race_age.load_gra_model import load_model as load_gra_model

# Import hàm dự đoán cảm xúc từ ảnh
from emotion.predict import predict_from_image as predict_emotion_from_image

# Import hàm dự đoán giới tính/chủng tộc/tuổi từ ảnh
from gender_race_age.predict import predict_from_image as predict_gra_from_image

# Import các hàm biến đổi (transform) ảnh tương ứng với từng model
from emotion.transforms_image import transforms_image as emotion_transforms_image
from gender_race_age.transform_image import transforms_image as gra_transforms_image

# Import hàm phân tích dữ liệu sau khi nhận diện xong
from analysis import analyze_people


# =============================
# HÀM TIỀN XỬ LÝ ẢNH
# =============================
def load_image(image_numpy_array, transform_func):
    """
    Nhận đầu vào là ảnh (numpy array từ OpenCV, dạng BGR),
    chuyển sang RGB, rồi sang PIL.Image để áp dụng transform,
    cuối cùng thêm chiều batch và trả về tensor.
    """
    img = Image.fromarray(cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2RGB)).convert("RGB")  # Chuyển ảnh sang RGB và kiểu PIL.
    return transform_func(img).unsqueeze(0)  # Áp dụng transform, thêm batch dimension [1, C, H, W].


# =============================
# HÀM CHÍNH: NHẬN DIỆN REALTIME
# =============================
def detect_and_predict(video_source=0):
    """
    Mở camera/video, phát hiện nhiều khuôn mặt cùng lúc,
    dự đoán cảm xúc + giới tính + chủng tộc + tuổi cho từng khuôn mặt,
    và vẽ kết quả realtime trên màn hình.
    """
    # --- 1. Chuẩn bị thiết bị và model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chạy trên GPU nếu có, ngược lại CPU.
    emotion_model, idx_to_label = load_emotion_model("./emotion/model/best_emotion.pth")  # Load model cảm xúc.
    gra_model = load_gra_model("./gender_race_age/model/best_model.pt")  # Load model giới tính - chủng tộc - tuổi.

    # --- 2. Chuẩn bị MediaPipe để phát hiện khuôn mặt ---
    mp_face_detection = mp.solutions.face_detection  # Module phát hiện khuôn mặt.
    mp_drawing = mp.solutions.drawing_utils  # Dùng để vẽ bounding box khuôn mặt.

    # --- 3. Mở camera hoặc file video ---
    cap = cv2.VideoCapture(video_source)  # 0: camera mặc định.
    if not cap.isOpened():
        print("Không mở được camera/video.")
        return

    print("Realtime multi-face tracking đang chạy... Nhấn 'q' để thoát.")

    # --- 4. Cấu trúc lưu thông tin từng người ---
    # people_data: lưu các dự đoán theo ID người.
    people_data = defaultdict(lambda: {"emotion": [], "gender": [], "race": [], "age": []})
    next_person_id = 0  # ID cho người mới.
    tracked_faces = []  # Lưu danh sách các khuôn mặt đang theo dõi: (person_id, (x1, y1, x2, y2)).

    # =============================
    # HÀM PHỤ: TÍNH IOU (Intersection over Union)
    # =============================
    def compute_iou(b1, b2):
        """
        Tính tỷ lệ giao nhau giữa hai bounding boxes (b1, b2).
        Giá trị IOU dùng để xác định xem hai khuôn mặt có phải cùng người không.
        """
        xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
        boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return interArea / (boxAArea + boxBArea - interArea + 1e-6)  # Tránh chia 0.

    # =============================
    # HÀM PHỤ: TÌM XEM KHUÔN MẶT MỚI ĐÃ CÓ ID CHƯA
    # =============================
    def find_existing_face(x1, y1, x2, y2):
        """
        So sánh bbox mới với danh sách khuôn mặt đã track,
        nếu IOU > 0.3 thì coi là cùng người, trả về ID tương ứng.
        """
        for pid, (px1, py1, px2, py2) in tracked_faces:
            if compute_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.3:
                return pid
        return None  # Không tìm thấy => người mới.

    # =============================
    # VÒNG LẶP CHÍNH REALTIME
    # =============================
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as detector:
        while True:
            start_time = time.time()
            ret, frame = cap.read()  # Đọc frame từ camera.
            if not ret:
                break  # Nếu không đọc được => thoát.

            # Lấy kích thước khung hình.
            h, w, _ = frame.shape

            # MediaPipe yêu cầu ảnh RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Phát hiện khuôn mặt.
            results = detector.process(rgb_frame)

            # Nếu có khuôn mặt nào được phát hiện.
            if results.detections:
                for detection in results.detections:
                    # Lấy bounding box tỉ lệ (giá trị [0..1]) và chuyển sang pixel.
                    bbox = detection.location_data.relative_bounding_box
                    x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                    x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

                    # Tìm xem khuôn mặt này đã có ID chưa.
                    pid = find_existing_face(x1, y1, x2, y2)
                    if pid is None:
                        # Nếu là khuôn mặt mới => gán ID mới.
                        pid = next_person_id
                        next_person_id += 1
                        tracked_faces.append((pid, (x1, y1, x2, y2)))
                    else:
                        # Nếu đã tồn tại => cập nhật bbox mới nhất cho ID đó.
                        for i, (tid, _) in enumerate(tracked_faces):
                            if tid == pid:
                                tracked_faces[i] = (pid, (x1, y1, x2, y2))

                    # Cắt vùng khuôn mặt ra khỏi frame.
                    face_crop = frame[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]
                    if face_crop.size == 0:
                        continue  # Nếu cắt lỗi => bỏ qua.

                    # --- DỰ ĐOÁN CẢM XÚC ---
                    emo_tensor = load_image(face_crop, emotion_transforms_image).to(device)
                    emotion_pred, _, emotions, probs = predict_emotion_from_image(
                        emotion_model, emo_tensor, idx_to_label, top_k=len(idx_to_label), image_tensor=True
                    )

                    # --- DỰ ĐOÁN GIỚI TÍNH, CHỦNG TỘC, TUỔI ---
                    gra_tensor = load_image(face_crop, gra_transforms_image).to(device)
                    gra_result = predict_gra_from_image(gra_model, gra_tensor, image_tensor=True)

                    # --- LƯU KẾT QUẢ VÀO people_data ---
                    people_data[pid]["emotion"].append(emotion_pred)
                    people_data[pid]["gender"].append(gra_result['gender'])
                    people_data[pid]["race"].append(gra_result['race'])
                    people_data[pid]["age"].append(gra_result['age'])

                    # --- VẼ KẾT QUẢ LÊN MÀN HÌNH ---
                    mp_drawing.draw_detection(frame, detection)  # Vẽ khung khuôn mặt.
                    cv2.putText(frame, f"ID {pid} | {emotion_pred} | {gra_result}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # =============================
                    # VẼ THANH XÁC SUẤT CẢM XÚC
                    # =============================
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
            # --- Tính FPS ---
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # --- Vẽ tổng số người đã nhận diện ---
            cv2.putText(frame, f"Total people: {len(people_data)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # --- Hiển thị frame ---
            cv2.imshow("Realtime Multi-Person Detection", frame)

            # --- Nhấn 'q' để thoát ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # =============================
    # DỌN DẸP TÀI NGUYÊN VÀ PHÂN TÍCH KẾT QUẢ
    # =============================
    cap.release()  # Giải phóng camera.
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ hiển thị.
    analyze_people(people_data)  # Gọi hàm phân tích dữ liệu (vẽ biểu đồ, lưu file...).


# =============================
# CHẠY CHƯƠNG TRÌNH CHÍNH
# =============================
if __name__ == "__main__":
    detect_and_predict(0)  # Mặc định mở camera (source=0).
