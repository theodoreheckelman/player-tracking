# import numpy as np
# from norfair import Detection, Tracker
# import math


# class PlayerTracker:
#     def __init__(self, dist_thresh=75):
#         self.tracker = Tracker(distance_function=self._dist,
#                                distance_threshold=dist_thresh)

#     def _dist(self, detection, tracked_object):
#         det_pt = detection.points[0]
#         obj_pt = tracked_object.estimate[0]
#         return np.linalg.norm(det_pt - obj_pt)

#     def update(self, detections, frame=None):
#         """
#         detections: list of {'xyxy':[x1,y1,x2,y2], 'score':float}
#         returns dict mapping track_id -> box [x1,y1,x2,y2]
#         """
#         norfair_dets = []
#         for d in detections:
#             x1, y1, x2, y2 = d['xyxy']
#             cx = (x1 + x2) / 2.0
#             cy = (y1 + y2) / 2.0
#             norfair_dets.append(
#                 Detection(points=np.array([[cx, cy]]),
#                           scores=np.array([d['score']]))
#             )

#         tracked = self.tracker.update(norfair_dets)

#         out = {}
#         for obj in tracked:
#             tid = int(obj.id)
#             cx, cy = obj.estimate[0]

#             # match to nearest original box
#             best = None
#             best_dist = float('inf')
#             for d in detections:
#                 x1, y1, x2, y2 = d['xyxy']
#                 bx = (x1 + x2) / 2.0
#                 by = (y1 + y2) / 2.0
#                 dist = math.hypot(bx - cx, by - cy)
#                 if dist < best_dist:
#                     best_dist = dist
#                     best = d['xyxy']

#             if best is not None:
#                 out[tid] = best

#         return out
