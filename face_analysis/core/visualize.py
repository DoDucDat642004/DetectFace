import cv2

def draw_face(frame, box, text):
    """
    Vẽ khung chữ nhật quanh mặt và thông tin text (Tuổi, Giới tính...).
    """
    # (x1, y1): Góc trên bên trái, (x2, y2): Góc dưới bên phải
    x1, y1, x2, y2 = box
    
    # Hình chữ nhật màu xanh lá (0, 255, 0) với độ dày nét là 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Viết text
    # max(25, y1-10): text luôn nằm trong khung hình.
    cv2.putText(frame, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_emotion_bars(frame, probs, labels, x, y):
    """
    Vẽ biểu đồ thanh (Bar chart) thể hiện xác suất cảm xúc.
    """
    # Duyệt qua từng cảm xúc và xác suất tương ứng
    for i, lbl in enumerate(labels):
        # Tính tọa độ Y cho từng dòng (mỗi dòng cách nhau 20 pixel)
        yy = y + i * 20
        
        # Vẽ khung viền (Background bar)
        # Tạo một thanh màu xám cố định độ dài 100px
        cv2.rectangle(frame, (x, yy), (x + 100, yy + 15), (80, 80, 80), 1)
        
        # Vẽ thanh giá trị (Value bar)
        # Độ dài thanh màu xanh lá phụ thuộc vào xác suất (probs[i] * 100)
        # Tham số -1 là tô đặc (filled) hình chữ nhật
        cv2.rectangle(frame, (x, yy), (x + int(100 * probs[i]), yy + 15), (0, 255, 0), -1)
        
        # Viết tên cảm xúc bên cạnh thanh biểu đồ
        cv2.putText(frame, lbl, (x + 105, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)