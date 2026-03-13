import os
import argparse
import sys

# Import cấu hình và các hàm khởi tạo
from face_analysis.config import Config
from face_analysis.core.device import get_device
from face_analysis.core.models import load_models

# Import các hàm tiền xử lý ảnh
from emotion.transforms_image import transforms_image as emo_tf
from gender_race_age.transform_image import transforms_image as gra_tf

# Import các hàm thực thi chính
from face_analysis.runners.image_runner import run_image
from face_analysis.runners.video_runner import run_video
from face_analysis.runners.camera_runner import run_camera


def main():
    # ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="Phân tích Khuôn mặt (Face Analysis System)")

    # --mode: run
    parser.add_argument(
        "--mode",
        choices=["image", "video", "camera"],
        required=True,
        help="Chọn chế độ : xử lý ảnh, video hoặc webcam"
    )
    
    # --input: Đường dẫn đầu vào (File ảnh, File video hoặc ID Camera)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Đường dẫn đến ảnh/video hoặc ID camera (mặc định là 0)"
    )
    
    # --output: Đường dẫn thư mục đầu ra
    parser.add_argument(
        "--output",
        type=str,
        default="output", # Mặc định lưu vào thư mục 'output'
        help="Đường dẫn thư mục để lưu kết quả"
    )

    args = parser.parse_args()

    # INITIALIZATION
    
    device = get_device()
    
    # Load cấu hình
    cfg = Config()
    
    # Load toàn bộ Model (YOLO, Emotion, Gender/Race/Age)
    models = load_models(cfg, device)

    # Đóng gói tài nguyên vào Context (ctx)
    ctx = {
        "device": device,
        "cfg": cfg,
        "models": models,
        "emo_tf": emo_tf, # Transform cho model cảm xúc
        "gra_tf": gra_tf, # Transform cho model giới tính/tuổi
    }

    
    # XỬ LÝ ẢNH TĨNH
    if args.mode == "image":
        if not args.input:
            parser.error("Chế độ --mode image yêu cầu tham số --input (đường dẫn ảnh).")
        
        # Gọi hàm xử lý ảnh
        run_image(args.input, args.output, ctx)

    # XỬ LÝ VIDEO
    elif args.mode == "video":
        if not args.input:
            parser.error("Chế độ --mode video yêu cầu tham số --input (đường dẫn video).")
            
        # Gọi hàm xử lý video
        run_video(args.input, args.output, ctx)

    # CAMERA / WEBCAM
    elif args.mode == "camera":
        # Nếu input None/"0", chuyển thành số nguyên 0
        # Nếu input "1", "2"... chuyển thành int tương ứng
        # Nếu input đường dẫn (RTSP stream), giữ nguyên string
        if args.input is None:
            cam_id = 0
        elif args.input.isdigit():
            cam_id = int(args.input)
        else:
            cam_id = args.input
            
        print(f"Kết nối tới Camera ID/Source: {cam_id}")
        run_camera(cam_id, args.output, ctx)

if __name__ == "__main__":
    main()

'''
RUN (EXAMPLES):

1. Chạy với Ảnh:
   python -m face_analysis.main --mode image --input ./input/images/image.jpg --output ./output/image/

2. Chạy với Video:
   python -m face_analysis.main --mode video --input ./input/videos/video.mp4 --output ./output/video/

3. Chạy với Webcam:
   python -m face_analysis.main --mode camera --input 0 --output ./output/camera/
'''