from ultralytics import YOLO

model = YOLO(r"./model/yolo11n.pt")

model.export(format='openvino')