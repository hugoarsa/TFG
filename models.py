import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights

class ResNet18(nn.Module):
    """
    ResNet-18 model for multi-label classification.
    Source: "Deep Residual Learning for Image Recognition" by He et al., 2015.
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, num_classes, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    """
    ResNet-50 model for multi-label classification.
    Source: "Deep Residual Learning for Image Recognition" by He et al., 2015.
    https://arxiv.org/abs/1512.03385
    """
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class DenseNet121(nn.Module):
    """
    DenseNet-121 model for multi-label classification.
    Source: "Densely Connected Convolutional Networks" by Huang et al., 2017.
    https://arxiv.org/abs/1608.06993
    """
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 model for multi-label classification.
    Source: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Tan and Le, 2019.
    https://arxiv.org/abs/1905.11946
    """
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 model for multi-label classification.
    Source: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Tan and Le, 2019.
    https://arxiv.org/abs/1905.11946
    """
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB3, self).__init__()
        self.model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)