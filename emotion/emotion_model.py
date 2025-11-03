from .packages import *

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y
    
class EmotionResNet18_SE(nn.Module):
    def __init__(self, num_classes=8, reduction=8, dropout=0.3, drop_block=0.05, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        base_model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        print("Loaded pretrained ResNet18 backbone (RGB, ImageNet)")
        
        # Lấy toàn bộ feature layers
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
        # Chèn SE block ở tầng cuối (tăng focus vùng khuôn mặt)
        self.se = SEBlock(512, reduction=reduction)
        
        # Global pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

