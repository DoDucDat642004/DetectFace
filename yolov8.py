from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 pretrained (có thể thay bằng yolov8n-face.pt nếu bạn có model fine-tuned)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("YOLOv8 Face Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
