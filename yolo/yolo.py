import os
import cv2
import time
import torch
import requests
import time
from ultralytics import YOLO

# Load the trained model
model = YOLO(r"/Users/dodat/Documents/DetectFace/yolo/model/yolo11n_openvino_model")

# Start the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the YOLO model to the frame
    results = model.predict(source=frame, conf=0.5, save=False)

    # Draw only the boxes (without labels)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Box coordinates
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw the box

    # Display the output on the screen
    cv2.imshow("YOLO Face Detection", frame)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()