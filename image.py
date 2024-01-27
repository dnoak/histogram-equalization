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
    def resize_max(image, max_res):
        max_dim = max(image.shape)
        if max_dim > max_res:
            resize_scale = max_res / max_dim
        else:
            max_dim_arg = np.argmax(image.shape)
            resize_scale = max_res / image.shape[max_dim_arg]
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
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
    