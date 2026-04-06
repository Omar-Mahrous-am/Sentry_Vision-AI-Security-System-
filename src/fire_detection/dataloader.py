from .dataset import Fire_Dataset, Subset_Dataset

from torch.utils.data import random_split
from torch.utils.data import DataLoader


def get_dataloaders(dataset: Fire_Dataset, batch_size,
                    train_fac, val_fac, test_fac,
                    main_transform, fire_train_transform, no_fire_train_transform):
    
    total_size = len(dataset)
    
    train_size = int(total_size * train_fac)
    val_size = int(total_size * val_fac)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_dataset = Subset_Dataset(train_dataset, main_transform, fire_train_transform, no_fire_train_transform, is_train=True)
    val_dataset = Subset_Dataset(val_dataset, main_transform, fire_train_transform, no_fire_train_transform, is_train=False)
    test_dataset = Subset_Dataset(test_dataset, main_transform, fire_train_transform, no_fire_train_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader