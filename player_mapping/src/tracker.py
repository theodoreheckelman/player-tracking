from bytetrack import BYTETracker, STrack

class ByteTrackWrapper:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, buffer_size=30, frame_rate=30):
        """
        A wrapper around BYTETracker with parameters compatible with main.py
        """
        self.tracker = BYTETracker(track_thresh=track_thresh)
        # store extra params (not fully used in our minimal BYTE implementation)
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
        tracks = self.tracker.update(dets, frame)
        output = []
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr())
            output.append({
                "track_id": t.track_id,
                "xyxy": [x1, y1, x2, y2]
            })
        return output

