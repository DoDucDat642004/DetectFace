# Face Analysis System

**Face Detection · Emotion · Gender · Race · Age · Statistics**

## Giới thiệu

Dự án **Face Analysis System** là một hệ thống phân tích khuôn mặt hoàn chỉnh sử dụng **Deep Learning** , cho phép:

- Phát hiện khuôn mặt (YOLOv11n)
- Nhận diện **Emotion** (EfficientNet_b0)
- Dự đoán **Gender – Race – Age** (EfficientNet_b0 fine-tuning)
- Theo dõi từng người trong video (tracking)
- Phân tích thống kê & trực quan hóa dữ liệu
- Xuất **CSV / Excel**

Hệ thống hỗ trợ **Image · Video · Camera laptop realtime** .

## Chức năng chính

### Face Detection

- Phát hiện nhiều khuôn mặt trong ảnh / video
- Gán ID cho từng người

### Emotion Recognition

- angry, fear, happy, neutral, sad, surprise
- Hiển thị confidence
- Phân tích theo thời gian (timeline)

### Gender – Race – Age

- Gender: Male / Female
- Race: White, Black, Latino, Asian, Indian.
- Age: phân nhóm độ tuổi (0–9, 10–19, 20-29, 30-39, 40-59, 60+)

### Phân tích & Thống kê

- Bar chart, Pie chart
- Grouped Bar: Emotion × Gender
- Heatmap: Emotion × Age
- Stacked Bar: Race distribution
- Time Series: Emotion theo thời gian
- Confidence Distribution

### Xuất dữ liệu

- `people_summary.csv`
- `people_summary.xlsx`
- `people_summary.json`

### Cài đặt

#### Clone repository

git clone https://github.com/your-username/DetectFace.git

cd DetectFace

#### Cài thư viện

pip install -r requirements.txt

### Hướng dẫn sử dụng

###### RUN MODE IMAGE

python -m face_analysis.main --mode image --input ./input/images/image.jpg --output ./output/image/

###### RUN MODE VIDEO

python -m face_analysis.main --mode video --input ./input/videos/video.mp4 --output ./output/video/

###### RUN MODE CAMERA

python -m face_analysis.main --mode camera --input 0 --output ./output/camera/

### Source :

Source video demo : https://www.youtube.com/watch?v=YzcawvDGe4Y

Source image demo : https://www.kaggle.com/datasets/canomercik/wider-face-dataset-for-yolov12-format (test/images/0_Parade_Parade_0_119.jpg)

Source sort.py : https://github.com/abewley/sort/blob/master/sort.py
