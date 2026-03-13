import numpy as np
from face_analysis.core.sort import Sort

class FaceTracker:
    def __init__(
        self, 
        max_age=100,       # Số frame tối đa giữ lại ID khi khuôn mặt bị mất dấu (bị che khuất, ra khỏi khung hình).
        min_hits=3,       # Số frame liên tiếp cần xuất hiện xác nhận đây là một "người" thật.
        iou_threshold=0.3 # Ngưỡng chồng lấn (Intersection Over Union) ghép cặp khuôn mặt frame trước và frame sau
    ):
        # Khởi tạo bộ theo dõi SORT
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

    def update(self, boxes):
        """
        Cập nhật vị trí các khuôn mặt và trả về ID tương ứng.
        
        Args:
            boxes: List các bounding box phát hiện được từ model detection
                   Dạng: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        
        Returns:
            List các box đã được gán ID
            Dạng: [[x1, y1, x2, y2, track_id], ...]
        """
        # Nếu không có khuôn mặt nào được phát hiện trong frame
        if len(boxes) == 0: 
            # Update với mảng rỗng, SORT tự động tăng biến đếm 'thời gian mất dấu' (time_since_update)
            self.tracker.update(np.empty((0, 5)))
            return []

        # SORT định dạng đầu vào là: [x1, y1, x2, y2, score]
        # Vì model detection trả về box trần : giả định confidence score là 1.0
        detections = np.array(
            [[*b, 1.0] for b in boxes], dtype=np.float32
        )

        # Uupdate SORT
        # Thuật toán sẽ dùng Kalman Filter để dự đoán vị trí mới và thuật toán Hungary để ghép cặp (matching) với các detection mới dựa trên IOU.
        tracks = self.tracker.update(detections)
        
        # Trả về: [x1, y1, x2, y2, track_id]
        return tracks