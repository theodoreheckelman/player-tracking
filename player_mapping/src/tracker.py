from bytetrack import BYTETracker, STrack

class ByteTrackWrapper:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, buffer_size=30, frame_rate=30):
        """
        A wrapper around BYTETracker with parameters compatible with main.py
        """
        self.tracker = BYTETracker(track_thresh=track_thresh)
        self.match_thresh = match_thresh
        self.buffer_size = buffer_size
        self.frame_rate = frame_rate

    def update(self, dets, frame):
        """
        Args:
            dets: list of {"xyxy": [x1,y1,x2,y2], "score": float}
            frame: np.ndarray (unused but kept for API consistency)
        Returns:
            list of dicts: [{"track_id": int, "xyxy": [x1,y1,x2,y2]}]
        """

        # Get tracks from BYTETracker (your minimal version)
        tracks = self.tracker.update(dets, frame)

        output = []
        for t in tracks:
            # t could be either STrack or a dict with tlwh; compute tlbr
            if hasattr(t, "tlbr"):
                x1, y1, x2, y2 = map(int, t.tlbr())
                tid = t.track_id
            else:
                # fallback if t is a dict {"tlwh": [...], "track_id": ...}
                tlwh = t.get("tlwh", [0,0,0,0])
                x, y, w, h = tlwh
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                tid = t.get("track_id", -1)

            output.append({
                "track_id": tid,
                "xyxy": [x1, y1, x2, y2]
            })

        return output
