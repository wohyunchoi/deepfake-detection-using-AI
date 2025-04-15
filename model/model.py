import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model

class HybridDeepfakeDetector_XS(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridDeepfakeDetector_XS, self).__init__()

        # XceptionNet
        self.xception = create_model('xception', pretrained=False, num_classes=0)

        # Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=512)

        # Hybrid
        self.feature_fusion = nn.Linear(512 + 2048, 1024)
        self.fc = nn.Linear(1024, num_classes)

        # Xavier Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        xception_features = self.xception(x)  # (batch, 2048)

        swin_features = self.swin(x)  # (batch, 512)

        fused_features = torch.cat((xception_features, swin_features), dim=1)  # (batch, 2560)
        fused_features = self.feature_fusion(fused_features)
        
        out = self.fc(fused_features)
        return out
    


class HybridDeepfakeDetector_ES(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridDeepfakeDetector_ES, self).__init__()

        # EfficientNet-b3 (원하는 등급으로 변경 가능)
        self.efficientnet = create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.efficientnet_out_dim = self._get_output_dim(self.efficientnet)

        # Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=512)

        # Hybrid Fusion
        self.feature_fusion = nn.Linear(self.efficientnet_out_dim + 512, 1024)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _get_output_dim(self, model):
        # 테스트 입력으로 feature dimension 계산
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        return out.shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        eff_features = self.efficientnet(x)  # (batch, ~1536 for b3)
        swin_features = self.swin(x)        # (batch, 512)

        fused_features = torch.cat((eff_features, swin_features), dim=1)
        fused_features = self.feature_fusion(fused_features)
        out = self.fc(fused_features)
        return out