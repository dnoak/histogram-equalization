from pathlib import Path
import cv2
import PIL.Image
import numpy as np

class Image:
    @staticmethod
    def show(image, max_res=768):
        cv2.imshow('image', Image.resize_max(image, max_res))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @staticmethod
    def show_pillow(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        PIL.Image.fromarray(image).show()

    @staticmethod
    def save(image, filename):
        if Path(filename).parents[0].exists() == False:
            Path(filename).parents[0].mkdir(parents=True, exist_ok=True) 
        cv2.imwrite(filename, image)

    @staticmethod
    def resize(image, fxy):
        return cv2.resize(image, (0, 0), fx=fxy, fy=fxy)
    
    @staticmethod
    def is_portrait(shape, threshold=1):
        return shape[0] * threshold > shape[1]

    @staticmethod
    def alpha_to_white(image):
        background = np.ones(image.shape[:2] + (3,), dtype=np.uint8) * 255
        overlay = image
        alpha_channel = overlay[:, :, 3] / 255
        overlay_colors = overlay[:, :, :3]
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
        h, w = overlay.shape[:2]
        background_subsection = background[0:h, 0:w]
        composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
        background[0:h, 0:w] = composite
        return background
    
    @staticmethod
    def to_square(image):
        (h, w) = image.shape[:2]
        if h == w: return image
          
        pad = np.abs(h - w)
        pad_A = pad // 2
        pad_B = pad - pad_A
        if h > w:
            top = 0; bottom = 0
            left = pad_A; right = pad_B
        else:
            top = pad_A; bottom = pad_B
            left = 0; right = 0

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        #print(image.shape)
        #assert image.shape[0] == image.shape[1]
        return image
    
    @staticmethod
    def resize_max(image, max_res):
        print(image.shape)
        max_dim = max(image.shape)
        if max_dim > max_res:
            resize_scale = max_res / max_dim
        else:
            max_dim_arg = np.argmax(image.shape)
            resize_scale = max_res / image.shape[max_dim_arg]
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        print(image.shape)
        print()
        return image

    @staticmethod
    def rotate(image, angle, r90cw=False):
        if r90cw:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2) 
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    