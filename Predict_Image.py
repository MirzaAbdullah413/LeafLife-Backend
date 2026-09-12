import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# --- Must match the architecture used during training exactly ---
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


def load_trained_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def predict_image(model, image_path, val_transformation, classes, device):
    if isinstance(image_path, str):
        pil_img = Image.open(image_path).convert("RGB")
    else:
        pil_img = image_path.convert("RGB")
    img_tensor = val_transformation(pil_img).unsqueeze(0).to(device)

    model.eval()
    output = model(img_tensor)
    class_id = torch.argmax(output, dim=1).item()
    class_name = classes[class_id]
    print("Predicted class:", class_name)

    target_layer = model.conv_block5.block[0]  # last Conv2d layer
    heatmap = grad_cam(model, img_tensor, class_id, target_layer)
    overlaid_image = overlay_heatmap(pil_img, heatmap)

    return class_name, overlaid_image


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python Predict_Image.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    trained_model, val_transformation, classes, device = load_trained_model(model_path)
    class_name, heatmap_image = predict_image(
        trained_model, image_path, val_transformation, classes, device
    )

    plt.imshow(heatmap_image)
    plt.title(f"Predicted: {class_name}")
    plt.axis("off")
    plt.show()

sys.modules['__main__'].CNN = CNN
sys.modules['__main__'].CNNBlock = CNNBlock 