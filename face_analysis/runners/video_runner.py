import os
import cv2
import torch
import time
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

def run_video(input_path, output_dir, ctx, show=True):
    # Khởi tạo Video Capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {input_path}")

    # Lấy thông số video gốc
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25 
    
    # Tính thời gian cho 1 frame
    # Ví dụ: 25 FPS -> mỗi frame cần 0.04s
    target_frame_time = 1.0 / fps 

    # Thư mục output
    os.makedirs(output_dir, exist_ok=True)

    # Khởi tạo Video Writer
    save_path = os.path.join(output_dir, "result_video.mp4")
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps,
        (w, h),
    )

    # Lấy tài nguyên từ Context
    yolo, emo_model, gra_model, idx_to_label = ctx["models"]
    device = ctx["device"]
    cfg = ctx["cfg"]

    emo_model.eval()
    gra_model.eval()

    emo_tf = ctx["emo_tf"]
    gra_tf = ctx["gra_tf"]

    # Khởi tạo Tracker
    tracker = FaceTracker(max_age=20, min_hits=3, iou_threshold=0.3)

    people_data = defaultdict(lambda: {
        "emotion": [], "emotion_conf": [],
        "gender": [], "race": [], "age": [], "timeline": []
    })

    emo_labels_list = list(idx_to_label.values())
    frame_idx = 0

    try:
        while True:
            # Thời gian xử lý frame
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break 
            
            frame_orig = frame.copy()
            frame_idx += 1

            # PHÁT HIỆN MẶT
            results = yolo.predict(frame, conf=cfg.YOLO_CONF, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results else []

            boxes_xyxy = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes_xyxy.append([x1, y1, x2, y2])

            # TRACKING
            tracks = tracker.update(boxes_xyxy)

            emo_batch, gra_batch, track_meta = [], [], []
            for x1, y1, x2, y2, track_id in tracks:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                crop = frame_orig[y1:y2, x1:x2]
                if crop.size == 0: continue

                emo_batch.append(image_to_tensor(crop, emo_tf))
                gra_batch.append(image_to_tensor(crop, gra_tf))
                track_meta.append((x1, y1, x2, y2, int(track_id)))

            # XỬ LÝ KHÔNG CÓ MẶT
            if not emo_batch:
                out.write(frame)
                
                # Tính thời gian trôi qua
                elapsed = time.time() - start_time
                
                # Nếu quá nhanh
                if show and elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

                if show:
                    cv2.putText(frame, "FPS: Real-time", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("Video Face Analysis", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # DỰ ĐOÁN
            emo_tensor = torch.stack(emo_batch).to(device)
            gra_tensor = torch.stack(gra_batch).to(device)

            emo_labels, emo_confs, emo_probs = predict_batch_emotion(emo_model, emo_tensor, idx_to_label)
            gra_results = predict_batch_gra(gra_model, gra_tensor)

            # VẼ & LƯU
            for i, (x1, y1, x2, y2, pid) in enumerate(track_meta):
                emo = emo_labels[i]
                emo_conf = float(emo_confs[i])
                gra = gra_results[i]

                people_data[pid]["emotion"].append(emo)
                people_data[pid]["emotion_conf"].append(emo_conf)
                people_data[pid]["gender"].append(gra["gender"])
                people_data[pid]["race"].append(gra["race"])
                people_data[pid]["age"].append(gra["age"])
                
                people_data[pid]["timeline"].append({
                    "person_id": pid,
                    "frame": frame_idx,
                    "time": frame_idx / fps,
                    "emotion": emo,
                    "confidence": emo_conf,
                })

                text = f"ID:{pid} | {emo} | {gra['gender']} | {gra['race']} | {gra['age']}"
                draw_face(frame, (x1, y1, x2, y2), text)
                bar_x = min(x2 + 10, w - 120)
                draw_emotion_bars(frame, emo_probs[i], emo_labels_list, bar_x, y1)

            # ĐỒNG BỘ TỐC ĐỘ
            out.write(frame)
            
            elapsed = time.time() - start_time
            # Nếu máy xử lý nhanh hơn video gốc -> sleep
            if show and elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

            if show:
                # Tính FPS hiển thị
                display_fps = 1 / max(time.time() - start_time, 1e-6)
                cv2.putText(frame, f"FPS:{display_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.imshow("Video Face Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")

    analyze_people(people_data, output_dir)
    return people_data