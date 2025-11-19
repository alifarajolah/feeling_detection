import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW

from config import Config
from models.feeling_detection_net import PersianFERNet
from utils.transforms import get_train_transforms, get_val_transforms

def main():
    os.makedirs("checkpoints", exist_ok=True)


    train_dataset = datasets.ImageFolder(Config.TRAIN_DIR, transform=get_train_transforms())
    val_dataset   = datasets.ImageFolder(Config.VAL_DIR,   transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)  # ← 0 و pin_memory=False

    val_loader   = DataLoader(val_dataset,   batch_size=Config.BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)  # ← 0 و pin_memory=False

    os.makedirs("checkpoints", exist_ok=True)


    model = PersianFERNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler = GradScaler()

    best_acc = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()


        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                with autocast():
                    outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch:3d} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"   → Best model saved! ({best_acc:.2f}%)")

    print(f"\nTraining finished! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()