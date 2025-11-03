from .packages import *

emotion_image_size = 112
transforms_image = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((emotion_image_size, emotion_image_size)),
    transforms.CenterCrop(emotion_image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

# transforms_image = transforms.Compose([
#     transforms.Resize((emotion_image_size, emotion_image_size)), 
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
# ])