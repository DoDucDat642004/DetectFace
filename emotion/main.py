# Nếu muốn run main.py trong thư mục emotion thì bỏ hết dấu '.' trên phần import ở các file trong thư mục này
from .load_emotion_model import load_model
from .predict import predict_from_url, predict_from_image

if __name__ == "__main__":
    # Load model
    model, idx_to_label = load_model()
    if model is None:
        print("Failed to load emotion model.")
    else:
        print("Emotion model loaded successfully.")

        # Prediction from image path
        # img_path = "../img1.jpg"
        # result = predict_from_image(model, img_path, idx_to_label, top_k=3)
        # print(result)

        # URL for testing
        test_url = "https://img.lovepik.com/photo/60177/2960.jpg_wh860.jpg"

        # Predict from URL
        predict_from_url(model, test_url, idx_to_label, top_k=3)