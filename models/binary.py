from torch import nn
from torchvision import models


def ResNetBinary(backbone='resnet152', num_hidden=512, pretrained=True):
    resnet = getattr(models, backbone)(pretrained)
    in_features = resnet.fc.in_features

    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),

        nn.BatchNorm1d(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features=in_features, 
                  out_features=num_hidden),
        nn.ReLU(),

        nn.BatchNorm1d(num_hidden),
        nn.Dropout(0.5),
        nn.Linear(in_features=num_hidden, out_features=2),
    )
    model = nn.Sequential(
        nn.Sequential(*list(resnet.children())[:-2]),
        head
    )

    return model

def DenseNetBinary(backbone='densenet121', num_hidden=512, pretrained=True):
    densenet = getattr(models, backbone)(pretrained)
    in_features = densenet.classifier.in_features

    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),

        nn.BatchNorm1d(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features=in_features, 
                  out_features=num_hidden),
        nn.ReLU(),

        nn.BatchNorm1d(num_hidden),
        nn.Dropout(0.5),
        nn.Linear(in_features=num_hidden, out_features=2),
    )
    model = nn.Sequential(
        nn.Sequential(*list(densenet.children())[:-1]),
        head
    )

    return model
