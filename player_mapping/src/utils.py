import cv2
import numpy as np


def apply_clahe_bgr(img, clipLimit=3.0, tileGridSize=(8,8)):
    """Apply CLAHE to BGR image by converting to LAB and enhancing L channel."""
    if img is None or img.size == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


def crop_lower_torso(box, img, lower_frac=0.40):
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    crop_y1 = int(y1 + (1.0 - lower_frac) * h)
    crop = img[crop_y1:y2, x1:x2]
    return crop


def ensure_int_box(xyxy, frame_shape):
    x1,y1,x2,y2 = xyxy
    h, w = frame_shape[:2]
    x1 = int(max(0, min(w-1, x1)))
    x2 = int(max(0, min(w-1, x2)))
    y1 = int(max(0, min(h-1, y1)))
    y2 = int(max(0, min(h-1, y2)))
    return x1,y1,x2,y2


import cv2
import numpy as np

def auto_detect_field_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) < 4:
        raise RuntimeError("Not enough lines detected for field corner estimation")

    # Collect all endpoints
    pts = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        pts.append((x1,y1))
        pts.append((x2,y2))

    pts = np.array(pts)
    # Use k-means to find 4 clusters (corners)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(pts)
    corners = kmeans.cluster_centers_

    # Sort corners clockwise: top-left, top-right, bottom-right, bottom-left
    def order_clockwise(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]])
    corners = order_clockwise(corners)
    return corners
