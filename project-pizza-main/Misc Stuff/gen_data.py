import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold


def generate_data(data_dir, batch_size, image_size, train_split, seed, augment_data):
    # Define data transformations
    if augment_data:
        data_transform = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    # Create PyTorch ImageFolder dataset
    dataset = ImageFolder(root=data_dir, transform=data_transform)

    # Calculate sizes for train, validation, and test sets
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    validation_size = int((total_size - train_size) / 2)
    test_size = int((total_size - train_size) / 2)

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(seed))

    # Create PyTorch DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    train_steps = len(train_loader.dataset) // batch_size
    val_steps = len(val_loader.dataset) // batch_size

    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, train_steps, val_steps
