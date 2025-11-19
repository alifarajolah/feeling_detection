import torch
import torch.nn as nn

class MBConv(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio=6, stride=1):
        super().__init__()
        hidden_dim = in_c * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        self.use_residual = in_c == out_c and stride == 1

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x += identity
        return x

class AttentionBlock(nn.Module):
    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.max(x, dim=2, keepdim=True)[0]
        max_pool = torch.max(max_pool, dim=3, keepdim=True)[0]
        pooled = avg_pool + max_pool
        scale = torch.sigmoid(pooled)
        return x * scale

class FeelingDetectionNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(2)
        )

        self.blocks = nn.Sequential(
            MBConv(64, 96, expand_ratio=4, stride=2),
            MBConv(96, 144, expand_ratio=6, stride=1),
            MBConv(144, 144, expand_ratio=6, stride=1),
            MBConv(144, 192, expand_ratio=6, stride=2),
            MBConv(192, 256, expand_ratio=6, stride=1),
            MBConv(256, 320, expand_ratio=6, stride=1),
        )

        self.attention = AttentionBlock()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

# تست سریع
if __name__ == "__main__":
    model = FeelingDetectionNet(7)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")