import os
import cv2
import torch
from collections import defaultdict

# Import các hàm hỗ trợ
from face_analysis.core.transforms import image_to_tensor
from face_analysis.core.visualize import draw_face, draw_emotion_bars
from face_analysis.analysis.statistics import analyze_people

# Import hàm dự đoán
from emotion.predict import predict_batch_emotion
from gender_race_age.predict import predict_batch_gra

def run_image(img_path, output_dir, ctx):
    """
    Xử lý và phân tích một ảnh tĩnh.

    Args:
        img_path (str): Đường dẫn đến file ảnh đầu vào.
        output_dir (str): Thư mục lưu kết quả.
        ctx (dict): Dictionary chứa context (models, config, device...).
    """
    # Đọc ảnh đầu vào
    frame = cv2.imread(img_path)
    if frame is None:
        raise RuntimeError(f"Không thể đọc ảnh từ đường dẫn: {img_path}")

    # Lấy tài nguyên từ Context
    yolo, emo_model, gra_model, idx_to_label = ctx["models"]
    device = ctx["device"]
    cfg = ctx["cfg"]
    
    # Transform (Resize, Normalize...)
    emo_tf = ctx["emo_tf"]
    gra_tf = ctx["gra_tf"]

    emo_model.eval()
    gra_model.eval()

    h, w = frame.shape[:2] # Lấy chiều cao, chiều rộng ảnh

    # Cấu trúc dữ liệu
    people_data = defaultdict(lambda: {
        "emotion": [], 
        "emotion_conf": [],
        "gender": [], 
        "race": [], 
        "age": [],
        "timeline": []
    })

    # PHÁT HIỆN KHUÔN MẶT
    # verbose=False: Tắt log in ra terminal
    results = yolo.predict(frame, conf=cfg.YOLO_CONF, verbose=False)
    
    # Lấy danh sách các bounding box
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

    # Danh sách chứa các ảnh mặt đã cắt
    emo_batch, gra_batch = [], []
    # Danh sách chứa thông tin tọa độ
    meta_data = []

    for pid, box in enumerate(boxes):
        # Chuyển tọa độ sang số nguyên
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Cắt tọa độ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Cắt vùng ảnh khuôn mặt
        face_crop = frame[y1:y2, x1:x2]
        
        # Bỏ qua ảnh cắt bị lỗi
        if face_crop.size == 0: continue

        # Tiền xử lý và thêm vào batch
        emo_batch.append(image_to_tensor(face_crop, emo_tf))
        gra_batch.append(image_to_tensor(face_crop, gra_tf))
        
        # Lưu lại tọa độ và ID
        meta_data.append((x1, y1, x2, y2, pid))

    os.makedirs(output_dir, exist_ok=True)

    # Nếu không tìm thấy khuôn mặt nào -> Lưu ảnh gốc và kết thúc
    if not emo_batch:
        print("Không tìm thấy khuôn mặt nào trong ảnh.")
        cv2.imwrite(os.path.join(output_dir, "result_image.jpg"), frame)
        analyze_people(people_data, output_dir)
        return people_data

    # Dự đoán Cảm xúc
    emo_labels, emo_confs, emo_probs = predict_batch_emotion(
        emo_model,
        torch.stack(emo_batch).to(device),
        idx_to_label
    )
    
    # Dự đoán Giới tính, Chủng tộc, Tuổi
    gra_results = predict_batch_gra(
        gra_model,
        torch.stack(gra_batch).to(device)
    )

    emo_label_list = list(idx_to_label.values())

    for i, (x1, y1, x2, y2, pid) in enumerate(meta_data):
        # Lấy kết quả tương ứng từ batch
        emo = emo_labels[i]
        emo_conf = float(emo_confs[i])
        gra = gra_results[i]

        # Lưu vào dict để thống kê
        people_data[pid]["emotion"].append(emo)
        people_data[pid]["gender"].append(gra["gender"])
        people_data[pid]["race"].append(gra["race"])
        people_data[pid]["age"].append(gra["age"])

        # Tạo chuỗi thông tin hiển thị
        text = f"ID:{pid} | {emo} | {gra['gender']} | {gra['race']} | {gra['age']}"
        
        # Vẽ khung và text lên ảnh
        draw_face(frame, (x1, y1, x2, y2), text)

        # Vẽ biểu đồ thanh xác suất cảm xúc
        bar_x = min(x2 + 10, w - 120)
        draw_emotion_bars(frame, emo_probs[i], emo_label_list, bar_x, y1)

    # LƯU KẾT QUẢ
    save_path = os.path.join(output_dir, "result_image.jpg")
    cv2.imwrite(save_path, frame)
    print(f"Đã lưu ảnh kết quả: {save_path}")

    analyze_people(people_data, output_dir)
    
    return people_data