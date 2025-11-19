import torch

class Config:

    DATA_ROOT = "data"
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR   = f"{DATA_ROOT}/test"

    NUM_CLASSES = 7
    INPUT_SIZE = 48

    BATCH_SIZE = 128
    EPOCHS = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = 0.1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {DEVICE.upper()}")

    MODEL_SAVE_PATH = "checkpoints/feeling_detection.pth"