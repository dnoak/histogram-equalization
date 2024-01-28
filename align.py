from glob import glob
import cv2
import numpy as np
from image import Image as im
from dataclasses import dataclass
from scipy import signal
import matplotlib.pyplot as plt

@dataclass
class CoinAlign:
    @staticmethod
    def coin_threshold(image, resize_factor):
        img_resized = im.resize(image, resize_factor)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 2)
        img_canny = cv2.Canny(img_blur, 50, 9)
        img_dilate = cv2.dilate(img_canny, np.ones((4, 2)), iterations=11)
        img_erode = cv2.erode(img_dilate, np.ones((13, 7)), iterations=4)
        return cv2.bitwise_not(img_erode)

    @staticmethod
    def square_bounding_box(image, margin):
        xaxis_sum = np.sum(image, axis=0)
        xaxis_grad = np.abs(np.gradient(xaxis_sum))
        xaxis_grad = np.convolve(xaxis_grad, np.ones(5), mode='same')
        x_start_peak = np.argmax(xaxis_grad[:len(xaxis_grad)//2])
        x_start = np.argwhere(xaxis_grad[:x_start_peak] == 0)[-2][0]
        x_end_peak = np.argmax(xaxis_grad[len(xaxis_grad)//2:]) + len(xaxis_grad)//2
        x_end = len(xaxis_grad) - np.argwhere(xaxis_grad[x_end_peak:] == 0)[-2][0]

        yaxis_sum = np.sum(image, axis=1)
        yaxis_grad = np.abs(np.gradient(yaxis_sum))
        yaxis_grad = np.convolve(yaxis_grad, np.ones(5), mode='same')
        y_start_peak = np.argmax(yaxis_grad[:len(yaxis_grad)//2])
        y_start = np.argwhere(yaxis_grad[:y_start_peak] == 0)[-2][0]
        y_end_peak = np.argmax(yaxis_grad[len(yaxis_grad)//2:]) + len(yaxis_grad)//2
        y_end = len(yaxis_grad) - np.argwhere(yaxis_grad[y_end_peak:] == 0)[-2][0]

        xsys_xeye = np.array([[x_start, y_start], [x_end, y_end]])
        axis_sizes = xsys_xeye[1] - xsys_xeye[0]
        axis_diff = np.abs(axis_sizes[0] - axis_sizes[1])

        # plt.plot(xaxis_grad)
        # plt.plot(yaxis_grad)
        # plt.show()

        min_axis_index = np.argmin(axis_sizes)
        xsys_xeye[0][min_axis_index] -= axis_diff // 2
        xsys_xeye[1][min_axis_index] += axis_diff // 2

        xsys_xeye[0, 0] -= np.max(axis_sizes) * margin
        xsys_xeye[0, 1] -= np.max(axis_sizes) * margin
        xsys_xeye[1, 0] += np.max(axis_sizes) * margin
        xsys_xeye[1, 1] += np.max(axis_sizes) * margin

        return *xsys_xeye[0], *xsys_xeye[1]

    @staticmethod
    def circle_mask(image, radius_ratio=1.05):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(
            mask, (image.shape[1] // 2, image.shape[0] // 2),
            int(image.shape[1] // 2 * radius_ratio), 255, -1
        )
        fg = cv2.bitwise_and(image, image, mask=mask)

        mask = cv2.bitwise_not(mask)
        background = np.full(image.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_and(background, background, mask=mask)

        return cv2.bitwise_or(fg, bk) 
    
    @staticmethod
    def align(image, resize_factor=1, margin=0):
        threshold = CoinAlign.coin_threshold(image, resize_factor)
        x0, y0, x1, y1 = CoinAlign.square_bounding_box(threshold, margin)
        image = image[y0:y1, x0:x1]
        return CoinAlign.circle_mask(image)


if __name__ == '__main__':
    ca = CoinAlign()
    for i in glob('data/*.JPG'):
        image = cv2.imread(i)
        ca.align(image)

