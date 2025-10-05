import easyocr
import cv2
from utils import apply_clahe_bgr, ensure_int_box


class JerseyRecognizer:
    def __init__(self, languages=['en'], gpu=True):
        self.reader = easyocr.Reader(languages, gpu=gpu)


    def read_number(self, crop):
        if crop is None or crop.size == 0:
            return None
        # Apply CLAHE to help numbers pop
        enhanced = apply_clahe_bgr(crop)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        # Upsample small crops to help OCR
        h, w = gray.shape
        if max(h,w) < 120:
            scale = int(120 / max(1, max(h,w)))
            gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        # EasyOCR expects color images normally; send 3-channel
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        try:
            res = self.reader.readtext(img, detail=1, allowlist='0123456789')
        except Exception:
            res = self.reader.readtext(img, detail=1)
        best = None
        best_conf = -1
        for bbox, text, conf in res:
            digits = ''.join(ch for ch in text if ch.isdigit())
            if len(digits) >= 1 and conf > best_conf:
                best_conf = conf
                best = digits
        return best