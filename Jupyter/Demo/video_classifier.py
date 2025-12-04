import Overlay_module

bnn_classifier = Overlay_module.CnvClassifier(Overlay_module.NETWORK_CNVW1A1,'cifar10',Overlay_module.RUNTIME_BNN)

# Stable center-crop + deterministic preprocessing + threaded BNN classification
import cv2
import time
import numpy as np
from PIL import Image
from IPython.display import display
import ipywidgets as widgets
import threading
from collections import deque, Counter

# ------------------ USER PARAMETERS ------------------
VIDEO_PATH = "TestAnimals7.mp4"   # <-- change to your video file
TARGET_SIZE = (32, 32)            # BNN input size (CIFAR default)
CROP_FRACTION = 0.60              # center crop 60% of frame height & width
SMOOTH_WINDOW = 7                 # majority vote window
# -----------------------------------------------------

# ------------------ Preprocessing --------------------
def preprocess_pil_for_bnn(pil, target_size=TARGET_SIZE):
    pil.load()                      # force decode
    pil = pil.convert("RGB")
    pil = pil.resize(target_size, Image.BILINEAR)
    return pil

def stable_center_crop(frame_rgb, frac=CROP_FRACTION):
    H, W, _ = frame_rgb.shape
    w = int(W * frac)
    h = int(H * frac)
    x1 = (W - w) // 2
    y1 = (H - h) // 2
    return frame_rgb[y1:y1+h, x1:x1+w]

# ------------------ Widgets & video open ----------------
image_widget = widgets.Image(format='jpeg')
display(image_widget)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

# smoothing buffer & thread lock
smooth_buffer = deque(maxlen=SMOOTH_WINDOW)
predicted_label = "..."
lock = threading.Lock()

# stats
class_times = deque(maxlen=200)
frame_count = 0
start_time = time.time()

# ------------------ Classification thread ----------------
def classify_thread(pil_for_bnn):
    global predicted_label

    t0 = time.time()
    class_id = bnn_classifier.classify_image(pil_for_bnn)
    label = bnn_classifier.class_name(class_id)
    t1 = time.time()

    with lock:
        predicted_label = label
        smooth_buffer.append(label)
        class_times.append((t1 - t0) * 1000.0)

# ------------------ Main loop ----------------
print("▶ Playing video with stable center cropping... (interrupt to stop)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("◼ Video ended.")
        break

    frame_count += 1
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame_rgb.shape

    # -------- STABLE CENTER CROP -----------
    crop_rgb = stable_center_crop(frame_rgb, frac=CROP_FRACTION)

    # Preprocess
    pil_for_bnn = preprocess_pil_for_bnn(Image.fromarray(crop_rgb))

    # Threaded classification
    threading.Thread(target=classify_thread, args=(pil_for_bnn,), daemon=True).start()

    # Temporal smoothing
    with lock:
        if len(smooth_buffer) > 0:
            display_label = Counter(smooth_buffer).most_common(1)[0][0]
        else:
            display_label = predicted_label

    # Overlay label
    cv2.putText(
        frame_rgb, 
        f"{display_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, 
        (255, 0, 0), 
        2
    )

    # FPS + classify time
    avg_class_time = np.mean(class_times) if len(class_times) else 0.0
    elapsed = time.time() - start_time
    fps_display = frame_count / elapsed
    cv2.putText(
        frame_rgb,
        f"FPS:{fps_display:.1f} CL:{avg_class_time:.1f}ms",
        (10, H-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, 
        (255,255,255), 
        2
    )

    # Display
    _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    image_widget.value = jpeg.tobytes()

cap.release()
print("✔ Done.")
