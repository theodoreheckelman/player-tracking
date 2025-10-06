import cv2
import argparse
import numpy as np
import pandas as pd
from detection import PlayerDetector
from tracker import ByteTrackWrapper   # use BYTETrack now
from ocr import JerseyRecognizer
from calibration import pixel_to_field, compute_homography
from visualizer import FieldVisualizer
from utils import ensure_int_box, crop_lower_torso, auto_detect_field_corners

# --------------------------
# CLI Arguments
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="path to input video")
parser.add_argument("--homography", default=None, help="path homography.npz")
parser.add_argument("--out", default="outputs/tracking.csv")
parser.add_argument("--skip-visual", action="store_true", help="disable video and field visualization")
parser.add_argument("--make-homography", action="store_true", help="pick points for homography")
args = parser.parse_args()

# --------------------------
# Initialize models
# --------------------------
detector = PlayerDetector(
    exp_file="train/YOLOX/yolox/exp/my_player_dataset.py",
    weights_path="train/YOLOX/YOLOX_outputs/my_player_dataset/best_ckpt.pth",
    conf_thresh=0.2,  # lower for testing
)
tracker = ByteTrackWrapper(track_thresh=0.2, match_thresh=0.8, buffer_size=30, frame_rate=30)
ocr = JerseyRecognizer(gpu=False)  # set gpu=True if available

# --------------------------
# Load or compute homography
# --------------------------
H = None
if args.homography:
    data = np.load(args.homography)
    H = data['H']
elif args.make_homography:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for homography pick")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read frame for homography pick")

    # Auto detect corners
    corners = auto_detect_field_corners(frame)
    field_pts = np.array([[0,0], [120,0], [120,53.3], [0,53.3]], dtype=np.float32)
    H = compute_homography(corners, field_pts)
    np.savez("homography.npz", H=H, pixel_pts=corners, field_pts=field_pts)
    print("Saved automatic homography to homography_auto.npz")

# --------------------------
# Initialize video capture
# --------------------------
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")

# --------------------------
# Initialize top-down field visualizer
# --------------------------
field_vis = FieldVisualizer(scale=10)  # 10 pixels per yard

# --------------------------
# Drawing helper
# --------------------------
def draw_detections(frame, dets, color_map=None):
    h, w = frame.shape[:2]
    if color_map is None:
        color_map = {}
    for det in dets:
        x1, y1, x2, y2 = det["xyxy"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
        cls = det.get("cls", -1)
        track_id = det.get("track_id", -1)
        color = color_map.get(cls, (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}"
        if cls >= 0:
            label += f" cls{cls}"
        cv2.putText(frame, label, (x1, max(y1-5,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# class color mapping
color_map = {
    detector.player_class_idx: (0, 255, 0),
    detector.football_class_idx: (0, 0, 255),
}

# --------------------------
# Tracking loop
# --------------------------
records = []
all_field_coords = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # --- Detection ---
    dets = detector.detect(frame)  # raw detections
    print(dets)
    if not dets:
        continue

    # --- Tracking ---
    tracks = tracker.update(dets, frame)  # track_id + xyxy

    # --- Combine tracks with cls for drawing and recording ---
    draw_list = []
    player_coords = []

    for i, t in enumerate(tracks):
        tid = t["track_id"]
        x1, y1, x2, y2 = ensure_int_box(t["xyxy"], frame.shape)
        cls = dets[i]["cls"] if i < len(dets) else -1

        # Crop lower torso for jersey OCR
        crop = crop_lower_torso((x1, y1, x2, y2), frame)
        jersey = ocr.read_number(crop) if crop is not None else None

        # Estimate player center in pixels
        cx, cy = int((x1 + x2) / 2), int(y2)

        # Map to field coordinates if homography exists
        if H is not None:
            field_x, field_y = pixel_to_field(H, (cx, cy))
            field_x = np.clip(field_x, 0, 120)
            field_y = np.clip(field_y, 0, 53.3)
        else:
            field_x, field_y = None, None

        draw_list.append({
            "xyxy": [x1, y1, x2, y2],
            "cls": cls,
            "track_id": tid
        })

        player_coords.append((field_x, field_y))

        # Record info
        records.append({
            "frame": frame_idx,
            "track_id": tid,
            "jersey": jersey,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy,
            "field_x": field_x,
            "field_y": field_y,
            "cls": cls
        })

    all_field_coords.append(player_coords)

    # --- Draw detections
    if not args.skip_visual:
        draw_detections(frame, draw_list, color_map=color_map)
        cv2.imshow("Detection", cv2.resize(frame, (640, 360)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# --------------------------
# Top-down field visualization
# --------------------------
if not args.skip_visual:
    print("Displaying top-down field view...")
    for frame_idx, player_coords in enumerate(all_field_coords, 1):
        vis_img = np.zeros((int(53.3*field_vis.scale), int(120*field_vis.scale), 3), dtype=np.uint8)
        if player_coords:
            vis_img = field_vis.draw_players(player_coords)

        cv2.imshow("Top-Down Field View", cv2.resize(vis_img, (640, 360)))
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    print("Press any key to close the field view window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------
# Save CSV
# --------------------------
df = pd.DataFrame(records)
df.to_csv(args.out, index=False)
print(f"Tracking data saved to {args.out}")
