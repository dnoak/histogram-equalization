import rembg
import cv2
from dataclasses import dataclass

@dataclass
class RemoveBackground:
    session = rembg.new_session()
    
    def crop_bg(self, bg_removed):
        alpha_bg_removed = bg_removed[:, :, 3]
        nonzero_coords = cv2.findNonZero(alpha_bg_removed)
        x, y, w, h = cv2.boundingRect(nonzero_coords)
        return bg_removed[y:y+h, x:x+w]

    def remove_bg(self, image):
        return rembg.remove(image, session=self.session)
