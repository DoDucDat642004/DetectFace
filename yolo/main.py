import cv2
import time
from ultralytics import YOLO

# Load model 
model_path = r"./yolo/model/yolo11n_openvino_model"
try:
    model = YOLO(model_path, task="detect")
    print(f"Đã load model từ: {model_path}")
except Exception as e:
    print(f"Không tìm thấy model : {e}")
    exit()

# Khởi động Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở Webcam.")
    exit()


while True:
    start_time = time.time()
    
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được tín hiệu từ Camera. Đang dừng...")
        break

    # DỰ ĐOÁN
    # conf=0.5: Chỉ lấy các box có độ tin cậy > 50%
    # save=False: Không lưu ảnh ra đĩa
    # stream=True: Tối ưu generator cho video stream
    # verbose=False: Tắt log in ra terminal
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    # VISUALIZATION
    # results[0] là kết quả của frame hiện tại
    for box in results[0].boxes:
        # Lấy tọa độ x1, y1, x2, y2
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # Vẽ hình chữ nhật quanh mặt (Màu xanh lá: 0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Tính toán FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)
    
    # Hiển thị FPS góc trái trên màn hình
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị
    cv2.imshow("YOLO Face Detection", frame)

    # Nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Chương trình đã kết thúc.")

# Run : python -m yolo.main