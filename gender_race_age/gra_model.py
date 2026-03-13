from gender_race_age.packages import *
from gender_race_age.device import *

device = get_device()

class SpatialAttention(nn.Module):
    """
    Cơ chế sự chú ý không gian (Spatial Attention Module - SAM).
    Tập trung vào phần quan trọng mắt mũi miệng trên ảnh.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        # Input channel = 2
        # Output channel = 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid() # Đưa về khoảng [0, 1]

    def forward(self, x):
        # x shape: [Batch, Channel, H, W]
        
        # Nén thông tin theo trục Channel
        avg_out = torch.mean(x, dim=1, keepdim=True) # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
        
        # Ghép lại
        x_cat = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        
        # Tính toán Attention Map
        attn = self.sigmoid(self.conv(x_cat)) # [B, 1, H, W]
        
        # Nhân trọng số vào Feature Map gốc
        return x * attn


class MultiTaskEfficientNetB0(nn.Module):
    def __init__(
        self,
        num_age_classes=6,
        dropout=0.5,
        freeze_backbone=True,
        gender_weight=None, 
        race_weight=None,
    ):
        super().__init__()
        self.num_age_classes = num_age_classes

        # Tải weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features

        # Vấn đề: EfficientNet gốc giảm size ảnh đi 1 nửa ngay layer đầu (Stride=2).
        # Với ảnh input nhỏ (112x112), cần giữ size lâu hơn để không mất chi tiết.
        
        # Lấy layer conv đầu tiên
        old_conv = self.features[0][0]
        
        # Tạo layer Conv2d giống nhưng với Stride=1
        new_conv = nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,# ***
            padding=old_conv.padding,
            bias=False
        )
        
        # Copy trọng số đã học từ ImageNet sang.
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight)
            
        # Gán lại vào mạng
        self.features[0][0] = new_conv

        # ATTENTION & POOLING
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # FREEZE STRATEGY
        if freeze_backbone:
            # Đóng băng toàn bộ backbone
            for p in self.features.parameters():
                p.requires_grad = False
            
            # Mở Layer đầu (do vừa đổi stride, cần học lại)
            for p in self.features[0].parameters():
                p.requires_grad = True
            
            # Mở băng các Block cuối (để học đặc trưng khuôn mặt thay vì vật thể chung)
            for name, p in self.features.named_parameters():
                if "features.6." in name or "features.7." in name:
                    p.requires_grad = True

        # Kích thước vector đặc trưng của EfficientNet-B0
        feat_dim = 1280

        # Gender: Phân loại 2 lớp
        self.gender_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 2))
        
        # Race: Phân loại 5 lớp
        self.race_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 5))
        
        # Age: Ordinal Regression (6 lớp)
        self.age_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_age_classes - 1) 
        )

        # LOSS FUNCTIONS
        self.loss_gender_fn = nn.CrossEntropyLoss(weight=gender_weight, label_smoothing=0.1)
        self.loss_race_fn = nn.CrossEntropyLoss(weight=race_weight, label_smoothing=0.1)
        self.loss_age_fn = nn.BCEWithLogitsLoss()

        # UNCERTAINTY WEIGHTING
        # Chỉnh tay hệ số (vd: 0.5*Gender + 1.5*Age...) -> tự học
        # log_vars tương đương log(variance) của từng task.
        # Task nhiễu cao (khó học) -> Variance cao -> Trọng số Loss giảm.
        self.log_vars = nn.Parameter(torch.zeros((3)))

    def forward(self, x):
        # Trích xuất đặc trưng
        x = self.features(x)

        # Spatial Attention
        x = self.spatial_attention(x) 
        
        # Global Average Pooling & Flatten
        x = self.pool(x).flatten(1)

        return {
            "gender": self.gender_head(x),
            "race":   self.race_head(x),
            "age":    self.age_head(x)
        }