import torch
import numpy as np
import sys
import os
import cv2

yolox_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../train/YOLOX"))
sys.path.append(yolox_path)
from yolox.exp import get_exp
from yolox.models import YOLOX


class PlayerDetector:
    def __init__(self, exp_file, weights_path, conf_thresh=0.2, imgsz=640):
        # Load YOLOX experiment (architecture)
        self.exp = get_exp(exp_file)
        self.device = torch.device("cpu")
        self.model = self.exp.get_model().to(self.device)

        # Load weights (.pth)
        checkpoint = torch.load(weights_path, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Grab class names
        if hasattr(self.exp, "class_names") and self.exp.class_names is not None:
            self.class_names = self.exp.class_names
        else:
            self.class_names = [str(i) for i in range(self.exp.num_classes)]

        # Player and football class indices
        try:
            self.player_class_idx = self.class_names.index("Player")
        except ValueError:
            self.player_class_idx = 18  # fallback to your dataset id
        try:
            self.football_class_idx = self.class_names.index("football")
        except ValueError:
            self.football_class_idx = 22  # fallback to your dataset id

    @staticmethod
    def letterbox(img, new_shape=640, color=(114, 114, 114)):
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img_padded

    def detect(self, frame):
        img_resized = self.letterbox(frame, self.imgsz)
        h0, w0 = frame.shape[:2]           # original shape
        h, w = img_resized.shape[:2]       # letterboxed shape

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        dets = []
        for output in outputs:
            output = output.cpu().numpy()
            if output.shape[0] == 0:
                continue
            for det in output:
                x1, y1, x2, y2 = det[:4]

                # Scale boxes back to original frame
                r_w = w0 / w
                r_h = h0 / h
                x1 = int(x1 * r_w)
                x2 = int(x2 * r_w)
                y1 = int(y1 * r_h)
                y2 = int(y2 * r_h)

                scores = det[4]
                class_probs = det[5:]
                cls = int(class_probs.argmax())
                conf = scores * class_probs[cls]

                if conf < self.conf_thresh:
                    continue
                w_box, h_box = x2 - x1, y2 - y1
                if w_box < 5 or h_box < 5:
                    continue

                dets.append({
                    "xyxy": [x1, y1, x2, y2],
                    "score": float(conf),
                    "cls": cls
                })
        return dets

