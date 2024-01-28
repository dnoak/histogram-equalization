import cv2
import numpy as np
from image import Image as im
from tqdm import tqdm


def coin_threshold(image, resize_factor):
    img_resized = im.resize(image, resize_factor)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
    img_canny = cv2.Canny(img_blur, 50, 9)
    img_dilate = cv2.dilate(img_canny, np.ones((4, 2)), iterations=11)
    img_erode = cv2.erode(img_dilate, np.ones((13, 7)), iterations=4)
    return img_erode

image = coin_threshold(cv2.imread('data\_8205735.JPG', 1), 1)
im.show(image)

for i in tqdm(range(0, 360, 2)):
    #im.show(image)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotatation_matrix = cv2.getRotationMatrix2D(center, i, 1.0)
    rotated = cv2.warpAffine(
        image, rotatation_matrix, image.shape[:2][::-1],
        flags=cv2.INTER_LINEAR
    )
    #im.show(rotated)

    image = np.bitwise_or(image, rotated)

im.show(image)