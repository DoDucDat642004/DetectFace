from emotion.packages import *

class CNNBlock(nn.Module):
    """Một khối Conv + BN + SiLU (Swish) chuẩn"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    """
    Tập trung vào các channel quan trọng.
    Giảm số chiều -> SiLU -> Tăng lại số chiều -> Sigmoid -> Nhân với input
    """
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution
    Cấu trúc: Expand -> Depthwise -> SE -> Pointwise
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio, reduction=4):
        super(MBConv, self).__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        layers = []
        
        # 1. Expansion Phase (expand_ratio != 1)
        if self.expand:
            layers.append(CNNBlock(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0))

        # 2. Depthwise Convolution (groups = hidden_dim)
        layers.append(CNNBlock(
            hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
        ))

        # 3. Squeeze and Excitation
        layers.append(SqueezeExcitation(hidden_dim, reduced_dim))

        # 4. Pointwise Convolution (Giảm channel -> out_channels)
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x) # Skip connection
        else:
            return self.conv(x)

class EfficientNetB0_Backbone(nn.Module):
    def __init__(self, input_size=224, in_channels=1):
        super(EfficientNetB0_Backbone, self).__init__()
        
        # EfficientNet-B0
        # (expand_ratio, channels, layers, stride, kernel_size)
        self.config = [
            # Stage 1
            (1,  16, 1, 1, 3),
            # Stage 2
            (6,  24, 2, 2, 3),
            # Stage 3
            (6,  40, 2, 2, 5),
            # Stage 4
            (6,  80, 3, 2, 3),
            # Stage 5
            (6, 112, 3, 1, 5),
            # Stage 6
            (6, 192, 4, 2, 5),
            # Stage 7
            (6, 320, 1, 1, 3),
        ]

        first_stride = 2 if input_size >= 224 else 1
        
        # Stem (Lớp Conv đầu tiên) -> 32 channels
        self.stem = CNNBlock(in_channels, 32, kernel_size=3, stride=first_stride, padding=1)
        
        # Xây dựng khối MBConv
        self.blocks = nn.ModuleList([])
        in_channels = 32
        
        for expand_ratio, out_channels, layers, stride, kernel_size in self.config:
            # Tính padding để giữ kích thước (same padding)
            padding = (kernel_size - 1) // 2 
            
            self.blocks.append(
                MBConv(in_channels, out_channels, kernel_size, stride, padding, expand_ratio)
            )
            in_channels = out_channels
            
            # Các layer còn lại trong stage luôn có stride = 1
            for _ in range(layers - 1):
                self.blocks.append(
                    MBConv(in_channels, out_channels, kernel_size, 1, padding, expand_ratio)
                )

        # Last Conv : 320 -> 1280
        self.last_conv = CNNBlock(320, 1280, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.last_conv(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class EmotionEfficientNet(nn.Module):
    def __init__(self, num_classes=7, input_size=112, in_channels=1, dropout=0.3):
        super(EmotionEfficientNet, self).__init__()
                
        # Custom Backbone
        self.backbone = EfficientNetB0_Backbone(input_size=input_size, in_channels=in_channels)
        
        # Spatial Attention
        self.spatial_att = SpatialAttention(kernel_size=7)
        
        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # [B, 1, 112, 112] -> Backbone -> [B, 1280, 7, 7]
        features = self.backbone(x)
        
        # Spatial Attention
        features = self.spatial_att(features)
        
        # Pooling & Classify
        pooled = self.avgpool(features)
        out = self.classifier(pooled)
        return out
