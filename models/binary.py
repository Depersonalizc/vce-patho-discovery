from torch import nn
from torchvision import models


def resnet50_binary(num_hidden=512):
    resnet = models.resnet50(pretrained=True)
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