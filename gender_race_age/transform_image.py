from .packages import *
gra_image_size = 112
mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
transforms_image = transforms.Compose([
    transforms.Resize((gra_image_size,gra_image_size)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean,std)
])