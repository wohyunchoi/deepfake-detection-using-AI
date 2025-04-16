def label_to_int(x):
    return int(x)

if __name__ == '__main__':
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from safetensors.torch import save_file, load_file
    from model import XceptionDeepfakeDetector, HybridDeepfakeDetector_XS, HybridDeepfakeDetector_ES
    import os
    import argparse
    from torch.amp import autocast, GradScaler

    # Argument Parse
    parser = argparse.ArgumentParser(description="Train Hybrid Deepfake Detector")
    parser.add_argument("-e", "--start-epoch", type=int, default=0, help="Start epoch (for resuming training)")
    parser.add_argument("-m", "--model-class", type=str, default="XS", choices=["XS", "ES"],
                        help="Model class to use: XS or ES")
    args = parser.parse_args()

    # Hyperparameter
    BATCH_SIZE = 16
    EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset
    train_dataset = datasets.ImageFolder("./archive/train", transform=transform)
    val_dataset = datasets.ImageFolder("./archive/valid", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # Model
    model_class = args.model_class
    if model_class == "X":
        model = XceptionDeepfakeDetector(num_classes=2).to(DEVICE)
    elif model_class == "XS":
        model = HybridDeepfakeDetector_XS(num_classes=2).to(DEVICE)
    elif model_class == "ES":
        model = HybridDeepfakeDetector_ES(num_classes=2).to(DEVICE)
    else:
        model = HybridDeepfakeDetector_XS(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scaler = GradScaler(device=DEVICE)

    # Load Checkpoint
    start_epoch = args.start_epoch
    if start_epoch > 0:
        checkpoint_path = f"{model_class}_ckpt/checkpoint_epoch_{start_epoch}.safetensors"
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)

    # Logging
    log_file = f"{model_class}_ckpt/train_log.txt"
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("Epoch, Train_Loss, Val_Acc, TP, TN, FP, FN, Precision, Recall, F1\n")

    print(f"Train Start - Model:{model_class}, Epoch:{start_epoch+1}")
    # Train & Validation
    for epoch in range(start_epoch, EPOCHS):
        # Train
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with autocast(device_type="cuda"):
                    outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Accuracy
        val_acc = accuracy_score(all_labels, all_preds)

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        # Precision, Recall, F1
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Print results
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        with open(log_file, 'a') as f:
            f.write(f"{epoch+1}, {train_loss:.4f}, {val_acc:.4f}, {tp}, {tn}, {fp}, {fn}, {precision:.4f}, {recall:.4f}, {f1:.4f}\n")

        # Save checkpoint
        checkpoint = {k: v for k, v in model.state_dict().items()}
        save_file(checkpoint, f"{model_class}_ckpt/checkpoint_epoch_{epoch+1}.safetensors")