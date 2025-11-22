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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlockSE(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16, drop_prob=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=reduction)
        # self.drop_prob = drop_prob

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # if self.training and self.drop_prob > 0:
        #     out = F.dropout2d(out, p=self.drop_prob, training=True)

        out = self.se(out)
        out = F.relu(out + identity, inplace=True)
        return out


class EmotionResNet(nn.Module):
    def __init__(self, num_classes=6, reduction=8, dropout=0.3, drop_prob=0.00):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False), # 64 -> 32
            nn.BatchNorm2d(32), # 64 -> 32
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_layer(32, 32, 2, stride=1, reduction=reduction, drop_prob=drop_prob)
        self.layer2 = self._make_layer(32, 64, 2, stride=2, reduction=reduction, drop_prob=drop_prob)
        self.layer3 = self._make_layer(64, 128, 1, stride=2, reduction=reduction, drop_prob=drop_prob) 
        self.layer4 = self._make_layer(128, 256, 2, stride=2, reduction=reduction, drop_prob=drop_prob)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_c, out_c, blocks, stride, reduction, drop_prob):
        layers = []
        layers.append(BasicBlockSE(in_c, out_c, stride, reduction, drop_prob))
        for _ in range(1, blocks):
            layers.append(BasicBlockSE(out_c, out_c, 1, reduction, drop_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x