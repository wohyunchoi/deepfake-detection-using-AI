import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import load_file
from model import HybridDeepfakeDetector_XS, HybridDeepfakeDetector_ES
from sklearn.metrics import classification_report

# Hyperparameter
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset
test_dataset = datasets.ImageFolder("./archive/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = HybridDeepfakeDetector_XS(num_classes=2).to(DEVICE)
checkpoint = load_file("XS_ckpt/checkpoint_epoch_10.safetensors")
model.load_state_dict(checkpoint["model"])
model.eval()

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))