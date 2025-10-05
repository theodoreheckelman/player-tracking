import numpy as np
from collections import deque

class STrack:
    def __init__(self, tlwh, score, track_id):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.track_id = track_id
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
            if score < self.track_thresh:
                continue
            w, h = x2 - x1, y2 - y1
            track = STrack([x1, y1, w, h], score, self.next_id)
            self.next_id += 1
            active[track.track_id] = track
        self.tracks = active
        return list(active.values())
