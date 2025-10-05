import cv2
import numpy as np

class FieldVisualizer:
    def __init__(self, field_dims=(120, 53.3), scale=10):
        """
        field_dims: (length, width) in yards
        scale: pixels per yard
        """
        self.field_length, self.field_width = field_dims
        self.scale = scale
        # create blank field
        self.img = np.zeros((int(self.field_width*scale), int(self.field_length*scale), 3), dtype=np.uint8)
        self._draw_field_lines()

    def _draw_field_lines(self):
        # Draw end zones
        cv2.rectangle(self.img, (0,0), (self.img.shape[1]-1, self.img.shape[0]-1), (0,255,0), 2)
        # Optional: add yard lines every 10 yards
        for y in range(0, int(self.field_length)+1, 10):
            x1 = 0
            y1 = int(y * self.scale)
            x2 = self.img.shape[1]-1
            y2 = y1
            cv2.line(self.img, (x1, y1), (x2, y2), (255,255,255), 1)

    def draw_players(self, coords):
        # Create blank top-down field image
        img = np.zeros((int(53.3 * self.scale), int(120 * self.scale), 3), dtype=np.uint8)
        
        # Draw each player as a circle
        for x, y in coords:
            px = int(x * self.scale)
            py = int(y * self.scale)
            cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
        
        return img


