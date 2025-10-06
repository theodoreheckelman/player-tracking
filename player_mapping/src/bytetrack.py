import numpy as np
from collections import deque

class STrack:
    def __init__(self, tlwh, score, track_id, cls):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.track_id = track_id
        self.cls = cls  # store class
        self.is_activated = True
        self.history = deque(maxlen=30)

    def tlbr(self):
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h])

class BYTETracker:
    def __init__(self, track_thresh=0.5):
        self.track_thresh = track_thresh
        self.next_id = 1
        self.tracks = {}

    def update(self, dets, frame=None):
        active = {}
        for det in dets:
            x1, y1, x2, y2 = det["xyxy"]
            score = det["score"]
            cls = det.get("cls", 0)  # default to 0 if missing
            if score < self.track_thresh:
                continue
            w, h = x2 - x1, y2 - y1
            track = STrack([x1, y1, w, h], score, self.next_id, cls)
            self.next_id += 1
            active[track.track_id] = track

        self.tracks = active

        # Return tracks as dicts for compatibility with main.py
        track_list = []
        for t in active.values():
            track_list.append({
                "track_id": t.track_id,
                "xyxy": t.tlbr().astype(int).tolist(),
                "score": t.score,
                "cls": t.cls
            })
        return track_list
