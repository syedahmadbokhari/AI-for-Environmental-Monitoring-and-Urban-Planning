import argparse
import os


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"Model loaded  : {model_path}")
    print(f"Epoch         : {checkpoint['epoch']}")
    print(f"Val accuracy  : {checkpoint['val_acc']:.4f}")
    print(f"Classes       : {class_names}")

    return model, class_names


infer_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image_bgr: np.ndarray, model, class_names, device, threshold: float):
    image_rgb = image_bgr[:, :, ::-1].copy()
    tensor = infer_tf(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    trash_prob = probs_dict.get("trash", 0.0)

    label = "trash" if trash_prob >= threshold else "no_trash"
    confidence = trash_prob if label == "trash" else probs_dict.get("no_trash", 0.0)

    return label, confidence, probs_dict


def draw_label(frame: np.ndarray, label: str, confidence: float) -> np.ndarray:
    h, w = frame.shape[:2]
    is_trash = label == "trash"
    color    = (0, 0, 220) if is_trash else (0, 200, 0)
    bg_color = (0, 0, 80)  if is_trash else (0, 80, 0)
    text     = f"{label.upper()}  {confidence:.1%}"

    cv2.rectangle(frame, (0, 0), (w, 52), bg_color, -1)
    cv2.putText(frame, text, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    bar_w = int(w * confidence)
    cv2.rectangle(frame, (0, 46), (bar_w, 52), color, -1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Trash detection — single image inference")
    parser.add_argument("--image",     required=True,              help="Path to input image")
    parser.add_argument("--model",     default="best_model.pth",   help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,    help="Trash confidence threshold (0-1)")
    parser.add_argument("--output",    default=None,               help="Path to save annotated image (optional)")
    parser.add_argument("--no-show",   action="store_true",        help="Do not display the image window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, class_names = load_model(args.model, device)

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: could not read image: {args.image}")
        return

    label, confidence, probs = predict(frame, model, class_names, device, args.threshold)

    print(f"\nImage      : {args.image}")
    print(f"Prediction : {label.upper()}")
    print(f"Confidence : {confidence:.1%}")
    print(f"All probs  : { {k: f'{v:.3f}' for k, v in probs.items()} }")

    annotated = draw_label(frame.copy(), label, confidence)

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Saved to   : {args.output}")

    if not args.no_show:
        cv2.imshow("Trash Detection", annotated)
        print("\nPress any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

