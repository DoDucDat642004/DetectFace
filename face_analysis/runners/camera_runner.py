import cv2
import torch
import time
import os 
from collections import defaultdict

# Import các module hỗ trợ
from face_analysis.core.transforms import image_to_tensor
from face_analysis.core.tracker import FaceTracker
from face_analysis.core.visualize import draw_face, draw_emotion_bars

# Import các hàm dự đoán
from emotion.predict import predict_batch_emotion
from gender_race_age.predict import predict_batch_gra

# Import module phân tích
from face_analysis.analysis.statistics import analyze_people

def run_camera(cam_id, output_dir, ctx):
    """
    Args:
        cam_id: ID camera.
        output_dir: Thư mục để lưu kết quả.
        ctx: Dictionary chứa context (models, config, device...).
    """
    # Thư mục output
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo Camera
    cap = cv2.VideoCapture(cam_id)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở camera/video: {cam_id}")

    # Khởi tạo Video Writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    save_path = os.path.join(output_dir, "camera_output.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    
    print(f"Video sẽ được lưu tại: {save_path}")

    # Lấy tài nguyên từ Context (ctx)
    # yolo: Model phát hiện khuôn mặt
    # emo_model: Model phân loại cảm xúc
    # gra_model: Model dự đoán giới tính, chủng tộc, tuổi
    yolo, emo_model, gra_model, idx_to_label = ctx["models"]
    device = ctx["device"]
    cfg = ctx["cfg"]

    emo_model.eval()
    gra_model.eval()

    # Lấy các transform (tiền xử lý ảnh)
    emo_tf = ctx["emo_tf"]
    gra_tf = ctx["gra_tf"]

    # Khởi tạo Tracker (SORT)
    # max_age=20: Giữ ID trong 20 frame nếu bị mất dấu
    # min_hits=3: Cần xuất hiện liên tiếp 3 frame mới cấp ID
    tracker = FaceTracker(
        max_age=20,
        min_hits=3,
        iou_threshold=0.3
    )

    # Lưu lịch sử dữ liệu từng người (Track ID)
    people_data = defaultdict(lambda: {
        "emotion": [],      # List các cảm xúc theo thời gian
        "emotion_conf": [], # Độ tin cậy
        "gender": [], 
        "race": [], 
        "age": [],
        "timeline": []
    })
    
    emo_labels_list = list(idx_to_label.values())
    frame_idx = 0

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            # Copy frame gốc để cắt ảnh (crop) đưa vào model. 
            # Frame 'frame' sẽ dùng để vẽ (draw) đè lên.
            frame_orig = frame.copy()
            frame_idx += 1
            h_frame, w_frame = frame.shape[:2]

            # PHÁT HIỆN KHUÔN MẶT
            results = yolo.predict(frame, conf=cfg.YOLO_CONF, verbose=False)
            
            # Lấy danh sách box
            boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

            # Làm sạch tọa độ box
            boxes_xyxy = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                boxes_xyxy.append([x1, y1, x2, y2])
            
            # TRACKING (Gán ID cho khuôn mặt)
            # Input: List các box tìm thấy
            # Output: List các box kèm ID [[x1, y1, x2, y2, id], ...]
            tracks = tracker.update(boxes_xyxy)

            # Chuẩn bị Batch
            emo_batch, gra_batch, track_meta = [], [], []

            for x1, y1, x2, y2, track_id in tracks:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Cắt ảnh khuôn mặt từ frame gốc
                crop = frame_orig[y1:y2, x1:x2]
                if crop.size == 0: continue

                # Tiền xử lý và đưa vào list
                emo_batch.append(image_to_tensor(crop, emo_tf))
                gra_batch.append(image_to_tensor(crop, gra_tf))
                track_meta.append((x1, y1, x2, y2, int(track_id)))

            # Nếu không có mặt nào -> vẽ frame ảnh gốc và tiếp tục
            if not emo_batch:
                # Tính FPS
                elapsed = time.time() - start
                curr_fps = 1 / max(elapsed, 1e-6)
                
                # Nếu xử lý quá nhanh khi không có khuôn mặt
                if elapsed < 0.033:
                    time.sleep(0.033 - elapsed)
                    curr_fps = 30.0

                cv2.putText(frame, f"FPS:{curr_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                out.write(frame)
                cv2.imshow("Camera Face Analysis", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # DỰ ĐOÁN
            # Chuyển list ảnh thành Tensor Batch [Batch_size, Channel, H, W]
            emo_tensor = torch.stack(emo_batch).to(device)
            gra_tensor = torch.stack(gra_batch).to(device)

            # Dự đoán cảm xúc
            emo_labels, emo_confs, emo_probs = predict_batch_emotion(
                emo_model, emo_tensor, idx_to_label
            )
            
            # Dự đoán Giới tính/Chủng tộc/Tuổi
            gra_results = predict_batch_gra(
                gra_model, gra_tensor
            )

            # LƯU DỮ LIỆU & KẾT QUẢ
            # Duyệt qua từng khuôn mặt trong batch
            for i, (x1, y1, x2, y2, pid) in enumerate(track_meta):
                emo = emo_labels[i]
                emo_conf = emo_confs[i]
                gra = gra_results[i]

                people_data[pid]["emotion"].append(emo)
                people_data[pid]["emotion_conf"].append(float(emo_conf))
                people_data[pid]["gender"].append(gra["gender"])
                people_data[pid]["race"].append(gra["race"])
                people_data[pid]["age"].append(gra["age"])
                
                # Lưu timeline
                people_data[pid]["timeline"].append({
                    "frame": frame_idx,
                    "time": frame_idx / fps,
                    "person_id": pid,
                    "emotion": emo,
                    "confidence": float(emo_conf),
                })

                # Thông tin văn bản
                text = f"ID:{pid} | {emo} | {gra['gender']} | {gra['race']} | {gra['age']}"
                draw_face(frame, (x1, y1, x2, y2), text)

                # Vẽ biểu đồ thanh xác suất cảm xúc bên cạnh mặt
                bar_x = min(x2 + 10, w_frame - 120) # Không vẽ ra ngoài màn hình
                draw_emotion_bars(frame, emo_probs[i], emo_labels_list, bar_x, y1)

            # Tính FPS
            curr_fps = 1 / max(time.time() - start, 1e-6)
            cv2.putText(frame, f"FPS:{curr_fps:.1f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Ghi frame đã vẽ vào video output
            out.write(frame)

            # Hiển thị cửa sổ
            cv2.imshow("Camera Face Analysis", frame)
            
            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Camera stopped and resources released.")

    # Phân tích
    analyze_people(people_data, output_dir)
    return people_data