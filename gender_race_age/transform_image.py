from .packages import *
gra_image_size = 112
transforms_image = transforms.Compose([
    transforms.Resize((gra_image_size, gra_image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])