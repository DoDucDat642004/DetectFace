from .packages import *


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6, use_se=True):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True)
            ]
            
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True)
        ]
        
        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(hidden) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.project(self.se(self.conv(x)))
        return out + x if self.use_res else out

class MultiTaskFaceModel(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        
        width_mult = 1.3 
        
        def C(v): return max(16, int(v * width_mult))

        # --- BACKBONE (Feature Extractor) ---
        self.stem = nn.Sequential(
            ConvBNReLU(3, C(32), stride=2),
            ConvBNReLU(C(32), C(48)),
        )

        # Stage 1: Low level features
        self.stage1 = nn.Sequential(
            InvertedResidual(C(48), C(64), stride=2, expand_ratio=4),
            InvertedResidual(C(64), C(64), expand_ratio=4),
            InvertedResidual(C(64), C(64), expand_ratio=4),
        )

        # Stage 2: Mid level features
        self.stage2 = nn.Sequential(
            InvertedResidual(C(64), C(128), stride=2, expand_ratio=6),
            InvertedResidual(C(128), C(128), expand_ratio=6),
            InvertedResidual(C(128), C(128), expand_ratio=6),
            InvertedResidual(C(128), C(128), expand_ratio=6),
        )

        # Stage 3: High level (Semantic) features - Quan trọng cho Age/Race
        self.stage3 = nn.Sequential(
            InvertedResidual(C(128), C(192), stride=2, expand_ratio=6),
            InvertedResidual(C(192), C(256), expand_ratio=6),
            InvertedResidual(C(256), C(256), expand_ratio=6), 
        )

        # Final PW & Pooling
        self.final_pw = ConvBNReLU(C(256), C(320), kernel=1, padding=0)
        
        self.pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        
        # Feature vector size: 320 * width_mult * 2 (avg + max)
        feat_dim = C(320) * 2 

        # --- ASYMMETRIC HEADS (Priority: Age > Race > Gender) ---
        
        # 1. GENDER HEAD (Priority: Low)
        self.gender_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        # 2. RACE HEAD (Priority: Medium)
        self.race_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7)
        )

        # 3. AGE HEAD (Priority: High)
        # Age cần không gian vector lớn để phân biệt các nếp nhăn/kết cấu nhỏ.
        self.age_head = nn.Sequential(
            nn.Linear(feat_dim, 1024),  # Wide expansion
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),       # Deep processing
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 9)
        )

        # Uncertainty Params (Learnable Loss weights)
        self.log_var_g = nn.Parameter(torch.tensor(0.0))
        self.log_var_r = nn.Parameter(torch.tensor(0.0))
        self.log_var_a = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_pw(x)
        
        # Global Pooling (Concat Avg + Max để giữ lại chi tiết nổi bật và nền)
        feats = torch.cat([p(x).flatten(1) for p in self.pool], dim=1)
        
        # Direct routing to heads (No bottleneck)
        return {
            "gender": self.gender_head(feats),
            "race": self.race_head(feats),
            "age": self.age_head(feats)
        }

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
