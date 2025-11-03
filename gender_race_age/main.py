# Nếu muốn run main.py trong thư mục gender_race_age thì bỏ hết dấu '.' trên phần import ở các file trong thư mục này
from .load_gra_model import load_model
from .predict import predict_from_image, predict_from_url


if __name__ == "__main__":
    # Load model
    model = load_model()
    if model is None:
        print("Failed to load GRA model.")
    else:
        print("GRA model loaded successfully.")

        # Test prediction
        # img_path = "../img1.jpg"
        # result = predict(model, img_path)
        # print(result)

        # URL for testing
        test_url = "https://img.lovepik.com/photo/60177/2960.jpg_wh860.jpg"
        # Predict from URL
        print(predict_from_url(model, test_url, show_image=True))