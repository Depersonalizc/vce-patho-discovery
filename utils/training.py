from torchvision import transforms

RESNET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RESNET_NORM,
])

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    RESNET_NORM,
])
