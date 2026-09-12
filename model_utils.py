"""
Adapted directly from Predict_Image.py.
No changes to the model architecture, preprocessing, or Grad-CAM logic —
only restructured so it can be loaded once and called per-request from FastAPI
instead of run as a one-shot CLI script.
"""
import base64
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# --- Must match the architecture used during training exactly (unchanged) ---
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_block1 = CNNBlock(3, 32)
        self.conv_block2 = CNNBlock(32, 64)
        self.conv_block3 = CNNBlock(64, 128)
        self.conv_block4 = CNNBlock(128, 256)
        self.conv_block5 = CNNBlock(256, 512)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Loading — identical logic to Predict_Image.py's load_trained_model()
# ---------------------------------------------------------------------------
def load_trained_model(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # weights_only=False is required here because checkpoint["model"] is a
    # full pickled model object (architecture + weights), not a bare state_dict.
    # This requires CNN / CNNBlock to be importable from this module, which is
    # why they're defined above exactly as in the original training/predict scripts.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    classes = checkpoint["classes"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model = checkpoint["model"]
    model.to(device)
    model.eval()

    val_transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return model, val_transformation, classes, device


# ---------------------------------------------------------------------------
# Grad-CAM — unchanged from Predict_Image.py
# ---------------------------------------------------------------------------
def grad_cam(model, img_tensor, class_id, target_layer):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(img_tensor)
    model.zero_grad()
    output[0, class_id].backward()

    h1.remove()
    h2.remove()

    acts = activations['value'][0]
    grads = gradients['value'][0]
    weights = grads.mean(dim=(1, 2))

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay_heatmap(pil_image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, pil_image.size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig = np.array(pil_image.convert("RGB"))
    overlaid = np.uint8(orig * (1 - alpha) + heatmap_color * alpha)
    return overlaid


# ---------------------------------------------------------------------------
# Prediction — same logic as Predict_Image.py's predict_image(), but takes a
# PIL.Image directly (from the uploaded file, in memory) instead of a file path,
# and returns the heatmap as a base64 PNG string instead of showing it via plt.
# ---------------------------------------------------------------------------
def predict_image(model, pil_img: Image.Image, val_transformation, classes, device):
    img_tensor = val_transformation(pil_img).unsqueeze(0).to(device)

    model.eval()
    output = model(img_tensor)
    class_id = torch.argmax(output, dim=1).item()
    class_name = classes[class_id]

    target_layer = model.conv_block5.block[0]  # last Conv2d layer
    heatmap = grad_cam(model, img_tensor, class_id, target_layer)
    overlaid_image = overlay_heatmap(pil_img, heatmap)

    # Encode the overlaid heatmap as a base64 PNG so it can travel in JSON
    heatmap_pil = Image.fromarray(overlaid_image)
    buffer = io.BytesIO()
    heatmap_pil.save(buffer, format="PNG")
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return class_name, heatmap_b64
