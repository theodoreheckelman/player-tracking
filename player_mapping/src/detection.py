from ultralytics import YOLO
import numpy as np


class PlayerDetector:
    def __init__(self, model_name="yolov8m.pt", conf_thresh=0.5, iou_thresh=0.7):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, frame):
        results = self.model(frame, imgsz=1280, conf=self.conf_thresh, augment=True)  # higher res, lower conf
        dets = []

        for r in results:
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    cls = int(b.cls[0].cpu().numpy())
                    name = self.model.model.names[cls]
                    if name != "person":
                        continue

                    xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    score = float(b.conf[0].cpu().numpy())
                    if score < self.conf_thresh:
                        continue

                    # --- Post-filter small boxes (remove false positives) ---
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    if w < 10 or h < 20:  # tweak based on footage
                        continue

                    dets.append({"xyxy": xyxy, "score": score})

        return dets
