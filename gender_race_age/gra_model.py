from .packages import *

class MultiTaskFaceModel(nn.Module):
    def __init__(self, backbone_type="resnet34", pretrained=True, shared_dim=256):
        super().__init__()
        # Backbone
        if backbone_type == "resnet34":
            base = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif backbone_type == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone type")

        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(feat_dim)

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Heads
        self.gender_head = nn.Linear(shared_dim, 2)
        self.race_head   = nn.Linear(shared_dim, 7)
        self.age_head    = nn.Linear(shared_dim, 9)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        shared = self.shared(x)
        return {
            "gender": self.gender_head(shared),
            "race": self.race_head(shared),
            "age": self.age_head(shared)
        }