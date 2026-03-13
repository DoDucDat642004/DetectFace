import os               # Đọc/ghi file, đường dẫn
import math             
import requests         # Gửi HTTP requests
from io import BytesIO  # Xử lý dữ liệu nhị phân
from typing import Dict
import numpy as np      
import cv2              # OpenCV: Thư viện xử lý ảnh/video
from PIL import Image, UnidentifiedImageError # Pillow: Đọc ảnh
import torch                    
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms        
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from ultralytics import YOLO