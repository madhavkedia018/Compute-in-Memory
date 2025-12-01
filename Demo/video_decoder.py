import bnn
from PIL import Image
import cv2
from collections import deque, Counter

# ----------------------
# Load BNN models
# ----------------------
hw_classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1, 'cifar10', bnn.RUNTIME_HW)
print("Classes:", hw_classifier.classes)

video_path = "TestAnimals5.mp4"
cap = cv2.VideoCapture(video_path)

# ----------------------
# Sampling / windowing
# ----------------------
frame_interval = 5  # classify every 5th frame
native_fps = cap.get(cv2.CAP_PROP_FPS)
if native_fps <= 1: native_fps = 30

sample_fps = native_fps / frame_interval
time_window_sec = 1.5
window_size = max(5, int(sample_fps * time_window_sec))

confidence_threshold = 0.7  # faster adapt
min_segment_sec = 0.5        # minimum segment length

print(f"[INFO] fps={native_fps}, sample_fps={sample_fps:.2f}, window={window_size}")

decision_window = deque(maxlen=window_size)

segments = []
cur_label = None
cur_start = 0
conf_sum = 0
conf_count = 0

frame_id = 0
proc = 0

try:
    RESAMPLE = Image.Resampling.LANCZOS
except:
    RESAMPLE = Image.ANTIALIAS

def weighted_vote(window):
    """Confidence-weighted voting by recency"""
    weights = {}
    N = len(window)
    for i, lbl in enumerate(window):
        w = (i+1) / N          # more recent = higher weight
        weights[lbl] = weights.get(lbl, 0) + w
    top = max(weights, key=weights.get)
    conf = weights[top] / sum(weights.values())
    return top, conf

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_interval != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((32, 32), RESAMPLE)
    raw = hw_classifier.class_name(hw_classifier.classify_image(img))

    decision_window.append(raw)
    proc += 1
    t = proc / sample_fps

    # Weighted window vote
    if len(decision_window) < window_size:
        print(f"t={t:.2f}s -> {raw}")
        continue

    top, win_conf = weighted_vote(list(decision_window))

    print(f"t={t:.2f}s -> RAW:{raw} | WIN:{top} ({win_conf*100:.1f}%)")

    # ---------- FAST SWITCH RULE ----------
    last3 = list(decision_window)[-3:]
    if len(last3) == 3 and len(set(last3)) == 1 and last3[-1] != top:
     print(f"[FAST SWITCH] {top} -> {last3[-1]}")
     top = last3[-1]
     win_conf = 1.0  # force high confidence switch

    # ---------- SEGMENT LOGIC --------------
    stable = win_conf >= confidence_threshold

    if cur_label is None and stable:
        cur_label = top
        cur_start = t
        conf_sum = win_conf
        conf_count = 1

    elif stable and top == cur_label:
        conf_sum += win_conf
        conf_count += 1

    elif stable and top != cur_label:
        duration = t - cur_start
        if duration >= min_segment_sec:
            segments.append({
                "label": cur_label,
                "start_s": cur_start,
                "end_s": t,
                "duration_s": duration,
                "avg_conf": conf_sum / conf_count
            })
        cur_label = top
        cur_start = t
        conf_sum = win_conf
        conf_count = 1

# -------- Close last segment --------
video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_duration = video_frames / native_fps

if cur_label is not None:
    duration = video_duration - cur_start
    if duration >= 0.3:
        segments.append({
            "label": cur_label,
            "start_s": cur_start,
            "end_s": video_duration,
            "duration_s": duration,
            "avg_conf": conf_sum / max(conf_count, 1)
        })

# -------- Print output ----------
print("\n===== FINAL TIMELINE =====")
for i, seg in enumerate(segments, 1):
    print(f"{i:02d}. {seg['label']:>5s} "
          f"{seg['start_s']:6.2f}s → {seg['end_s']:6.2f}s "
          f"({seg['duration_s']:5.2f}s) "
          f"conf={seg['avg_conf']*100:5.1f}%")
