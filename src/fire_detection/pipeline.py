from .config import config
from .dataset import Fire_Dataset
from .dataloader import get_dataloaders
from .model import MobileNetFireDetection
from .preprocessing import get_mean_std, augmentation_main_transform
from .train import train_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def run_pipeline():
    print(f"--- Starting Pipeline on {config.DEVICE} ---")
    
    # 1. Create dataset (no transforms — they'll be applied by Subset_Dataset)
    print("Loading Data...")
    dataset = Fire_Dataset(
        root_dir=config.DATA_DIR,
        transform=None,
        frames_per_video=config.FRAMES_PER_VIDEO
    )
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset.error_logs) > 0:
        print(f"  Warnings: {len(dataset.error_logs)} files had errors during loading")
    
    # 2. Compute dataset mean/std for normalization
    print("Computing dataset statistics...")
    mean, std = get_mean_std(dataset)
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std:  {std.tolist()}")
    
    # 3. Build augmentation transforms
    main_transform, fire_train_transform, no_fire_train_transform = augmentation_main_transform(mean, std)
    
    # 4. Create data loaders with train/val/test split
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        train_fac=config.TRAIN_SPLIT,
        val_fac=config.VAL_SPLIT,
        test_fac=config.TEST_SPLIT,
        main_transform=main_transform,
        fire_train_transform=fire_train_transform,
        no_fire_train_transform=no_fire_train_transform
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # 5. Initialize Model
    model = MobileNetFireDetection(pretrained=True).to(config.DEVICE)
    
    # 6. Loss with class imbalance weighting
    # Count labels to compute pos_weight
    num_fire = sum(1 for _, label, *_ in dataset.samples if label == 1)
    num_no_fire = sum(1 for _, label, *_ in dataset.samples if label == 0)
    
    if num_fire > 0 and num_no_fire > 0:
        pos_weight = torch.tensor([num_no_fire / num_fire]).to(config.DEVICE)
        print(f"  Class balance — Fire: {num_fire}, No Fire: {num_no_fire}, pos_weight: {pos_weight.item():.4f}")
    else:
        pos_weight = None
        print("  Warning: Could not compute class weights. Using unweighted loss.")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 7. Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 8. Train
    print("Beginning Training...")
    history, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        save_path=config.BEST_MODEL_PATH,
        epochs=config.EPOCHS,
        patience=config.PATIENCE
    )
    
    # 9. Save final weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'lr': config.LEARNING_RATE,
            'epochs': config.EPOCHS
        }
    }, config.MODEL_PATH)
    
    print(f"\nPipeline complete.")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"  Final model saved to: {config.MODEL_PATH}")
    
    return history, best_acc

if __name__ == "__main__":
    run_pipeline()