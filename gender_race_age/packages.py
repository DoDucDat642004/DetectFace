import torch
import torch.nn as nn
import torchvision.models as models

import os
import requests
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


import matplotlib.pyplot as plt

from typing import Dict