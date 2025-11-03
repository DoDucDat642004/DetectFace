from .packages import *
from .transform_image import transforms_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map ngược label → tên lớp
idx_to_gender = {0: "Male", 1: "Female"}
idx_to_race = {
    0: "White", 
    1: "Black", 
    2: "Latino_Hispanic",
    3: "East Asian",
    4: "Southeast Asian", 
    5: "Indian", 
    6: "Middle Eastern"
}
idx_to_age = {
    0: "0-2", 
    1: "3-9", 
    2: "10-19", 
    3: "20-29",
    4: "30-39", 
    5: "40-49", 
    6: "50-59", 
    7: "60-69", 
    8: "70+"
}

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transforms_image(img).unsqueeze(0)  # thêm batch dimension


def predict_from_image(model, image, image_tensor=False):
    if image_tensor:
        img_tensor = image
    else:
        # Tiền xử lý ảnh
        image_path = image
        img_tensor = load_image(image_path)
        img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

        pred_gender = outputs["gender"].argmax(dim=1).item()
        pred_race   = outputs["race"].argmax(dim=1).item()
        pred_age    = outputs["age"].argmax(dim=1).item()

    return {
        "gender": idx_to_gender[pred_gender],
        "race": idx_to_race[pred_race],
        "age": idx_to_age[pred_age],
    }


def predict_from_url(model, url, show_image=True):
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
    return predict_from_image(model, img_tensor, image_tensor=True)
