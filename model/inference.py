import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
from model import XceptionDeepfakeDetector, SwinDeepfakeDetector, HybridDeepfakeDetector_XS, HybridDeepfakeDetector_ES
from safetensors.torch import load_file


def load_model(model_name, weight_path, device):
    if model_name == "X":
        model = XceptionDeepfakeDetector(num_classes=2)
    elif model_name == "S":
        model = SwinDeepfakeDetector(num_classes=2)
    elif model_name == "XS":
        model = HybridDeepfakeDetector_XS(num_classes=2)
    elif model_name == "ES":
        model = HybridDeepfakeDetector_ES(num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = load_file(weight_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, 3, 224, 224)


def infer(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    return predicted_class, confidence


if __name__ == "__main__":
    # Argument Parse
    parser = argparse.ArgumentParser(description="Train Hybrid Deepfake Detector")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to model weights (.safetensors))")
    parser.add_argument("-m", "--model", type=str, default="X", choices=["X", "S", "XS", "ES"],
                        help="Model class to use: XS or ES")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weights, device)
    image_tensor = preprocess_image(args.image)
    label, conf = infer(model, image_tensor, device)

    label_name = "FAKE" if label == 0 else "REAL"
    print(f"[Result] Prediction: {label_name}, Confidence: {conf:.4f}")