from .packages import *
from .transforms_image import transforms_image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def load_image(path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms_image(img).unsqueeze(0)  # thêm batch dimension

# Biến toàn cục để cache danh sách nhãn (tránh tạo lại mỗi frame)
_CACHED_LABELS = None 

@torch.inference_mode()
def predict_from_image(model, image, idx_to_label, top_k=3, image_tensor=False, show_plot=False):
    global _CACHED_LABELS
    
    # Cache danh sách labels 1 lần duy nhất
    if _CACHED_LABELS is None:
        # Đảm bảo thứ tự đúng index 0, 1, 2...
        _CACHED_LABELS = [idx_to_label[i] for i in range(len(idx_to_label))]

    # 1. Xử lý đầu vào
    if image_tensor:
        img_tensor = image
    else:
        # Nếu là đường dẫn ảnh
        img_tensor = load_image(image).to(device)

    # 2. Dự đoán (Giữ nguyên trên GPU lâu nhất có thể)
    outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1) # Vẫn nằm trên GPU
    
    # 3. Lấy Top-1 nhanh nhất (Dùng torch.max thay vì sort numpy)
    # Item() sẽ lấy giá trị scalar ra khỏi tensor cực nhanh
    max_conf, max_idx = torch.max(probs, dim=1)
    
    pred_label = _CACHED_LABELS[max_idx.item()]
    pred_conf = max_conf.item()

    # 4. Chỉ chuyển sang Numpy khi cần vẽ biểu đồ (Realtime thì bước này tốn thời gian)
    # Nếu bạn cần vẽ thanh bar chart thì mới cần dòng dưới:
    probs_np = probs.cpu().numpy()[0] 
    
    # Trả về dữ liệu
    # probs_np: mảng xác suất (để vẽ biểu đồ)
    # _CACHED_LABELS: danh sách tên (để vẽ biểu đồ)
    return pred_label, pred_conf, _CACHED_LABELS, probs_np

@torch.inference_mode()
def predict_from_url(model, url, idx_to_label, top_k=3, show_image=False):
    """
    Dự đoán cảm xúc từ URL ảnh.
    Hiển thị top xác suất và biểu đồ cho toàn bộ lớp.
    """

    try:
        # --- Gửi request ---
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # --- Kiểm tra dữ liệu hợp lệ ---
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ValueError(f"URL không trả về ảnh. Content-Type: {content_type}")

        # --- Mở ảnh ---
        image = Image.open(BytesIO(response.content)).convert("RGB")

    except (UnidentifiedImageError, ValueError) as e:
        print(f"[ERROR] Không thể mở ảnh từ URL: {url}\nLý do: {e}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Lỗi khi tải ảnh: {e}")
        return None, None

    # --- Hiển thị ảnh ---
    if show_image:
        plt.imshow(image)
        plt.axis("off")
        plt.title("Input Image")
        plt.show()

    # --- Tiền xử lý ---
    img_tensor = transforms_image(image).unsqueeze(0).to(device)

    # --- Dự đoán ---
    return predict_from_image(model, img_tensor, idx_to_label, top_k=top_k, image_tensor=True)






# @torch.inference_mode()
# def predict_from_image(model, image, idx_to_label, top_k=3, image_tensor=False, show_plot=False):
#     if image_tensor:
#         img_tensor = image
#     else:
#         # Tiền xử lý ảnh
#         image_path = image
#         img_tensor = load_image(image_path)
#         img_tensor = img_tensor.to(device)

#     # Dự đoán
#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

#     # --- Xử lý kết quả ---
#     emotions = [idx_to_label[i] for i in range(len(probs))]
#     sorted_idx = probs.argsort()[::-1]  # sắp giảm dần
#     top_k = top_k or len(emotions)

#     # print("Top dự đoán:")
#     # for i in range(top_k):
#     #     lbl = emotions[sorted_idx[i]]
#     #     conf = probs[sorted_idx[i]] * 100
#     #     print(f"{i+1}. {lbl:10s} : {conf:.2f}%")

#     if show_plot:
#         # --- Biểu đồ xác suất ---
#         plt.figure(figsize=(8, 4))
#         plt.barh([emotions[i] for i in sorted_idx[::-1]],
#                 [probs[i]*100 for i in sorted_idx[::-1]],
#                 color="skyblue")
#         plt.xlabel("Probability (%)")
#         plt.title("Emotion Probabilities")
#         plt.tight_layout()
#         plt.show()

#     # --- Trả về nhãn cao nhất ---
#     pred_label = emotions[sorted_idx[0]]
#     pred_conf = probs[sorted_idx[0]]

#     return pred_label, pred_conf, emotions, probs
