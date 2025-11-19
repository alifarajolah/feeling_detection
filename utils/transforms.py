from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),  # حالا می‌شه (1, 48, 48)
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])