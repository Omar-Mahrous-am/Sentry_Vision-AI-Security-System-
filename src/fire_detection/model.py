import torch.nn as nn
from torchvision import models


class MobileNetFireDetection(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetFireDetection, self).__init__()
        
        if pretrained:
            self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)
        
        in_features = self.mobilenet.classifier[1].in_features
        
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1)
        )
    
    def forward(self, x):
        return self.mobilenet(x)
