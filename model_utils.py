import base64
import cv2

from Predict_Image import load_trained_model
from Predict_Image import predict_image as _predict_image

def predict_image(model, pil_image, val_transformation, classes, device):
    class_name, heatmap_image = _predict_image(
        model, pil_image, val_transformation, classes, device
    )

    bgr_image = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", bgr_image)

    if not success:
        raise ValueError("Failed to encode heatmap")
    heat_map = base64.b64encode(buffer).decode("utf-8")
    return class_name, heat_map