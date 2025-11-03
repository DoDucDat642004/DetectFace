import os
import requests
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

from PIL import Image, UnidentifiedImageError


import matplotlib.pyplot as plt
