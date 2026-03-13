from gender_race_age.packages import *
from gender_race_age.device import *
from gender_race_age.transform_image import transforms_image

device = get_device()

# Map label -> tên lớp
idx_to_race = {
    0: "White", 
    1: "Black", 
    2: "Latino_Hispanic",
    3: "Asian",   # East Asian + Southeast Asian
    4: "Indian"
}

# Model train với 6 nhóm Age (Binning)
idx_to_age = {
    0: "0-9", 
    1: "10-19", 
    2: "20-29",
    3: "30-39",
    4: "40-59", 
    5: "60+"
}

idx_to_gender = {0: "Male", 1: "Female"}


def load_image(path):
    img = Image.open(path).convert("RGB")
    return transforms_image(img).unsqueeze(0)  # thêm batch dimension

def decode_age_ordinal(logits):
    """
    Decode age theo Ordinal Regression:
    1. Sigmoid đưa về [0, 1]
    2. Đếm số lượng node > 0.5
    """
    probs = torch.sigmoid(logits)             # (B, C-1)
    age_idx = (probs > 0.5).sum(dim=1)        # (B,)
    return age_idx.long()


@torch.inference_mode()
def predict_from_image(model, image, image_tensor=False):
    # Xử lý đầu vào
    if image_tensor:
        # Truyền Tensor đã xử lý
        img_tensor = image
    elif isinstance(image, str):
        # Đường dẫn ảnh (string) -> Load từ file
        image_path = image
        img_tensor = load_image(image_path)
    else:
        # Đối tượng PIL Image -> Transform trực tiếp
        img_tensor = transforms_image(image).unsqueeze(0)
        
    img_tensor = img_tensor.to(device)
    
    # Forward
    model.eval()
    outputs = model(img_tensor)

    # Decode
    # Gender & Race: Classification -> Argmax
    pred_gender_idx = outputs["gender"].argmax(dim=1).item()
    pred_race_idx   = outputs["race"].argmax(dim=1).item()
    
    # Age: Ordinal
    pred_age_idx = decode_age_ordinal(outputs["age"]).item()

    return {
        "gender": idx_to_gender[pred_gender_idx],
        "race": idx_to_race[pred_race_idx],
        "age": idx_to_age[int(pred_age_idx)],
    }

@torch.inference_mode()
def predict_batch_gra(model, batch_tensor):
    """
    Hàm dự đoán GRA theo BATCH.
    """
    model.eval()
    
    # Forward pass
    outputs = model(batch_tensor)
    
    # Xử lý kết quả
    gender_idxs = outputs["gender"].argmax(dim=1).cpu().numpy()
    race_idxs   = outputs["race"].argmax(dim=1).cpu().numpy()
    
    # Age: Decode Ordinal
    age_idxs = decode_age_ordinal(outputs["age"]).cpu().numpy()

    
    # Gom danh sách kết quả
    results = []
    batch_size = batch_tensor.size(0)
    
    for i in range(batch_size):
        results.append({
            "gender": idx_to_gender[gender_idxs[i]],
            "race":   idx_to_race[race_idxs[i]],
            "age":    idx_to_age[int(age_idxs[i])],
        })
        
    return results

@torch.inference_mode()
def predict_from_url(model, url, show_image=True):
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
        print(f"Không thể mở ảnh từ URL: {url}\n {e}")
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
    return predict_from_image(model, img_tensor, image_tensor=True)

