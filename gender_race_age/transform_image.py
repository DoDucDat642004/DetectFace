from gender_race_age.packages import *

# Kích thước ảnh đầu vào cho model
gra_image_size = 112
# Chỉ số trung bình và độ lệch chuẩn của bộ ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
transforms_image = transforms.Compose([
    transforms.Resize((gra_image_size, gra_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])