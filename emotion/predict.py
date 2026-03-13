from emotion.packages import *
from emotion.device import *
from emotion.transforms_image import transforms_image

device = get_device()

_CACHED_LABELS = None 

def load_image(path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return transforms_image(img).unsqueeze(0)  # thêm batch dimension

@torch.inference_mode()
def predict_from_image(model, image, idx_to_label, top_k=3, image_tensor=False, show_plot=False):
    global _CACHED_LABELS
    
    if _CACHED_LABELS is None:
        _CACHED_LABELS = [idx_to_label[i] for i in range(len(idx_to_label))]

    # Xử lý đầu vào
    if image_tensor:
        # Nếu truyền Tensor đã xử lý
        img_tensor = image
    elif isinstance(image, str):
        # Nếu là đường dẫn ảnh (string) -> Load từ file
        img_tensor = load_image(image).to(device)
    else:
        # Nếu là đối tượng PIL Image (ví dụ từ crop cắt ra) -> Transform trực tiếp
        img_tensor = transforms_image(image).unsqueeze(0).to(device)

    # Dự đoán
    outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1)
    
    # Lấy Top-1
    max_conf, max_idx = torch.max(probs, dim=1)
    
    pred_label = _CACHED_LABELS[max_idx.item()]
    pred_conf = max_conf.item()

    # Chuyển sang Numpy khi cần vẽ biểu đồ
    probs_np = probs.cpu().numpy()[0] 
    
    # probs_np: mảng xác suất
    # _CACHED_LABELS: danh sách tên
    return pred_label, pred_conf, _CACHED_LABELS, probs_np

@torch.inference_mode()
def predict_from_url(model, url, idx_to_label, top_k=3, show_image=False):

    try:
        # Gửi request
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Kiểm tra dữ liệu hợp lệ
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            raise ValueError(f"URL không trả về ảnh. Content-Type: {content_type}")

        # Mở ảnh
        image = Image.open(BytesIO(response.content)).convert("RGB")

    except (UnidentifiedImageError, ValueError) as e:
        print(f"Không thể mở ảnh từ URL: {url}\nLý do: {e}")
        return None, None
    except Exception as e:
        print(f"Lỗi khi tải ảnh: {e}")
        return None, None

    # Hiển thị ảnh
    if show_image:
        plt.imshow(image)
        plt.axis("off")
        plt.title("Input Image")
        plt.show()

    # Tiền xử lý
    img_tensor = transforms_image(image).unsqueeze(0).to(device)

    # Dự đoán
    return predict_from_image(model, img_tensor, idx_to_label, top_k=top_k, image_tensor=True)


@torch.inference_mode()
def predict_batch_emotion(model, batch_tensor, idx_to_label):
    """
    Hàm dự đoán theo BATCH.
    
    Args:
        model: Model đã load.
        batch_tensor: Tensor đầu vào (Batch_Size, C, H, W).
        idx_to_label: Dictionary map index -> tên nhãn.
        
    Returns:
        batch_labels: List[str] - Danh sách nhãn dự đoán từng ảnh.
        batch_confs: List[float] - Danh sách độ tin cậy từng ảnh.
        batch_probs: Numpy Array - Mảng xác suất (Batch_Size, Num_Classes).
    """
    model.eval()
    
    # Dự đoán
    # outputs shape: (Batch_Size, Num_Classes)
    outputs = model(batch_tensor)
    probs = F.softmax(outputs, dim=1)

    # max_confs, max_idxs shape: (Batch_Size,)
    max_confs, max_idxs = torch.max(probs, dim=1)

    # Chuyển dữ liệu về CPU Numpy xử lý danh sách
    idxs_np = max_idxs.cpu().numpy()
    confs_np = max_confs.cpu().numpy()
    probs_np = probs.cpu().numpy()

    # Map index -> label cho từng phần tử trong batch
    batch_labels = [idx_to_label[i] for i in idxs_np]
    batch_confs = [float(c) for c in confs_np]

    return batch_labels, batch_confs, probs_np


