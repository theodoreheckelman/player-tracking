import cv2
import numpy as np
import argparse
from utils import ensure_int_box


def compute_homography(pixel_pts, field_pts):
    pixel = np.array(pixel_pts, dtype=np.float32)
    field = np.array(field_pts, dtype=np.float32)
    H, status = cv2.findHomography(pixel, field, method=cv2.RANSAC)
    return H


def pixel_to_field(H, px):
    pts = np.array(px, dtype=np.float32).reshape(-1, 1, 2)
    fld = cv2.perspectiveTransform(pts, H)
    return fld[0, 0, 0], fld[0, 0, 1]


def field_to_uv(field_pt, field_dims=(120.0, 53.3)):
    x, y = field_pt
    fx, fy = field_dims
    u = x / fx
    v = y / fy
    return float(u), float(v)


def auto_detect_field_corners(frame):
    """Automatically estimate field corners using edge + HoughLines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) < 4:
        raise RuntimeError("Not enough lines detected for field corner estimation")

    # Collect endpoints
    pts = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        pts.append((x1, y1))
        pts.append((x2, y2))
    pts = np.array(pts)

    # Extreme point heuristic
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return corners


def pick_points_and_save(video_path, out_npz="homography.npz", field_dims=(120.0, 53.3)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for homography pick")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read frame")

    # Auto corner detection
    pts = auto_detect_field_corners(frame)
    print("Auto-detected field corners:", pts)

    length, width = field_dims
    dst = [(0, 0), (length, 0), (length, width), (0, width)]
    H = compute_homography(pts, dst)

    np.savez(out_npz, H=H, pixel_pts=np.array(pts), field_pts=np.array(dst))
    print("Homography saved to", out_npz)


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="homography.npz")
    parser.add_argument("--make-homography", action="store_true")
    args = parser.parse_args()

    if args.make_homography:
        pick_points_and_save(args.video, args.out)
