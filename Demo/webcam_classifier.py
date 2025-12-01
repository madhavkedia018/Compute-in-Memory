import Overlay_module

bnn_classifier = Overlay_module.CnvClassifier(Overlay_module.NETWORK_CNVW1A1,'cifar10',Overlay_module.RUNTIME_BNN)

import cv2
import time
import numpy as np
from PIL import Image
from IPython.display import display
import ipywidgets as widgets
import threading

# ------------------------------------
# CIFAR-like preprocessing for webcam
# ------------------------------------
def preprocess_frame_to_cifar(frame_rgb):
    # 1. Center crop the frame to a square
    H, W, _ = frame_rgb.shape
    side = min(H, W)
    x1 = (W - side) // 2
    y1 = (H - side) // 2
    frame_cropped = frame_rgb[y1:y1+side, x1:x1+side]

    # 2. Apply CLAHE to normalize contrast
    lab = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    L = clahe.apply(L)
    lab2 = cv2.merge((L, A, B))
    frame_cropped = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    # 3. Light blur (CIFAR images are soft & small)
    frame_cropped = cv2.GaussianBlur(frame_cropped, (3,3), 0)

    # 4. Resize to CIFAR-10 size
    frame_resized = cv2.resize(frame_cropped, (32,32), interpolation=cv2.INTER_LINEAR)

    # 5. Convert to PIL for BNN
    pil_img = Image.fromarray(frame_resized)
    pil_img.load()
    return pil_img


# ------------------------------------
# Live video widget
# ------------------------------------
image_widget = widgets.Image(format='jpeg')
display(image_widget)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

predicted_label = "..."

print("🎥 Webcam started with CIFAR-like preprocessing...")


# Background classification thread
def classify_thread(frame_rgb):
    global predicted_label
    pil_img = preprocess_frame_to_cifar(frame_rgb)
    class_id = bnn_classifier.classify_image(pil_img)
    predicted_label = bnn_classifier.class_name(class_id)


# ------------------------------------
# Main loop
# ------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # spawn classifier thread
    frame_copy = frame_rgb.copy()
    threading.Thread(target=classify_thread, args=(frame_copy,), daemon=True).start()

    # overlay prediction
    cv2.putText(
        frame_rgb,
        predicted_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # update widget
    _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    image_widget.value = jpeg.tobytes()

cap.release()

