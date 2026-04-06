import torch
from torchvision import transforms
from PIL import Image


def get_mean_std(dataset):
    # Compose preprocessing: Resize all images to 128x128, then convert to Tensor
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    total_pixels = 0
    sum_pixels = torch.zeros(3, dtype=torch.float32)

    # First pass: compute mean
    for img, _ in dataset:
        if isinstance(img, str) or not isinstance(img, Image.Image):
            continue
        try:
            img_tensor = preprocess(img)
        except:
            continue
        pixels = img_tensor.view(3, -1)
        sum_pixels += pixels.sum(dim=1)
        total_pixels += pixels.size(1)

    if total_pixels == 0:
        raise ValueError("No valid images found!")

    mean = sum_pixels / total_pixels

    # Second pass: compute standard deviation
    sum_squared_diff = torch.zeros(3, dtype=torch.float32)
    for img, _ in dataset:
        if isinstance(img, str) or not isinstance(img, Image.Image):
            continue
        try:
            img_tensor = preprocess(img)
        except:
            continue
        pixels = img_tensor.view(3, -1)
        diff = pixels - mean.unsqueeze(1)
        sum_squared_diff += (diff ** 2).sum(dim=1)

    std = torch.sqrt(sum_squared_diff / total_pixels)

    return mean, std


def augmentation_main_transform(mean, std):
    
    # Validation/Test transform 
    main_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Fire class training transform 
    fire_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # No-fire class training transform (minority class)
    no_fire_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2)
    ])
    
    return main_transform, fire_train_transform, no_fire_train_transform