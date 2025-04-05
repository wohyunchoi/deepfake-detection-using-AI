import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model

class HybridDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridDeepfakeDetector, self).__init__()

        # XceptionNet
        self.xception = models.xception(pretrained=True)
        self.xception.fc = nn.Identity()  # Fully Connected Layer 제거하여 특징만 추출

        # Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=512)

        # Hybrid
        self.feature_fusion = nn.Linear(512 + 2048, 1024)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        xception_features = self.xception(x)  # (batch, 2048)

        swin_features = self.swin(x)  # (batch, 512)

        fused_features = torch.cat((xception_features, swin_features), dim=1)  # (batch, 2560)
        fused_features = self.feature_fusion(fused_features)
        
        out = self.fc(fused_features)
        return out