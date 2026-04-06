import torch
import os

class Config:
    # Paths
    DATA_DIR = "data/fire_dataset"
    MODELS_DIR = "models"
    MODEL_PATH = os.path.join(MODELS_DIR, "mobile_net_model.pth")
    BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_accuracy_model.pth")
    
    # Data split ratios
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 20
    PATIENCE = 5
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 1  # Binary classification
    FRAMES_PER_VIDEO = 10
    THRESHOLD = 0.5
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalization (Standard ImageNet stats)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

# Instance for easy import
config = Config()