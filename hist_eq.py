import cv2
import numpy as np
from glob import glob
from image import Image as im
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import signal

@dataclass
class ColorHistogram:
    @staticmethod
    def __smooth_norm_hist(histogram):
        histogram = histogram.flatten().astype(np.uint32)
        histogram = np.convolve(histogram, np.ones(9), mode='same')
        histogram = np.convolve(histogram, np.ones(9), mode='same')
        histogram = np.convolve(histogram, np.ones(9), mode='same')
        histogram = histogram / np.max(histogram) * 255
        return histogram

    @staticmethod
    def rgb_cv2_gray(image):
        alpha_channel = image[:, :, 3]
        gray_channel = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        gray_channel_not_alpha = gray_channel[np.where(alpha_channel > 0)]

        histogram = cv2.calcHist([gray_channel_not_alpha], [0], None, [256], [0, 256])
        return ColorHistogram.__smooth_norm_hist(histogram)
    
    @staticmethod
    def rgb_weighted_gray(image, weights):
        weights = np.array(weights) / np.sum(weights)
        rw, gw, bw = weights
        wg_image = image.astype(np.float32)

        alpha_channel = wg_image[:, :, 3]
        gray_channel = (wg_image[:, :, 0] * bw + wg_image[:, :, 1] * gw + wg_image[:, :, 2] * rw)
        gray_channel = gray_channel / np.max(gray_channel) * 255
        gray_channel_not_alpha = gray_channel[np.where(alpha_channel > 0)]

        histogram = cv2.calcHist([gray_channel_not_alpha.astype(np.uint8)], [0], None, [256], [0, 256])
        return ColorHistogram.__smooth_norm_hist(histogram)
    
    @staticmethod
    def hsv(image, channel):
        alpha_channel = image[:, :, 3]
        hsv_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, channel]
        hsv_channel_not_alpha = hsv_channel[np.where(alpha_channel > 0)]

        histogram = cv2.calcHist([hsv_channel_not_alpha], [0], None, [256], [0, 256])
        return ColorHistogram.__smooth_norm_hist(histogram)
    
    @staticmethod   
    def lab(image, channel):
        lab_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, channel]
        histogram = cv2.calcHist([lab_channel], [0], None, [256], [0, 256])
        return ColorHistogram.__smooth_norm_hist(histogram)

@dataclass
class ColorEqualization:
    @staticmethod
    def plot_histogram(channel, ranges=None, show=True, color='blue'):
        plt.plot(channel)
        if ranges is not None:
            plt.plot(ranges, channel[ranges], "x", c=color)
        plt.show() if show else None

    @staticmethod
    def find_equalization_ranges(histogram, slope_thresh, show_histogram):
        #histogram[0:10] = histogram[10]
        cumsum = np.cumsum(histogram)
        cumsum = cumsum / np.max(cumsum) * 255
        
        range_start = np.where(cumsum > slope_thresh)[0][0]
        range_end = 255 - np.where((max(cumsum) - cumsum)[::-1] > slope_thresh)[0][0]
        
        if show_histogram:
            # ColorEqualization.plot_histogram(grad, show=True)
            ColorEqualization.plot_histogram(histogram, [range_start], show=False, color='red')
            ColorEqualization.plot_histogram(histogram, [range_end], show=True, color='green')
        
        return [range_start, range_end], histogram

    @staticmethod
    def back_projection(image, ranges, space_name, channels):
        alpha_channel = image[:, :, 3]
        transformed_space = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{space_name}"))
        for channel in channels:
            value_channel = transformed_space[:, :, channel]
            clipped_channel = np.clip(value_channel, ranges[0], ranges[1])
            normalized_channel = cv2.normalize(clipped_channel, None, 0, 255, cv2.NORM_MINMAX)
            transformed_space[:, :, channel] = normalized_channel
        equalized_image = cv2.cvtColor(transformed_space, getattr(cv2, f"COLOR_{space_name}2BGR"))
        return cv2.merge([equalized_image, alpha_channel])

    @staticmethod
    def start(image, equalization, slope_thresh, channels, show_histogram):
        histogram = getattr(ColorHistogram, equalization['fn'])(image, **equalization['args'])
        ranges, histogram = ColorEqualization.find_equalization_ranges(histogram, slope_thresh, show_histogram)
        space_name = equalization['fn'].split('_')[0].upper()
        equalized_image = ColorEqualization.back_projection(image, ranges, space_name, channels)
        return equalized_image


if __name__ == '__main__':
    hce = ColorEqualization(
        images=glob('data/*.JPG'),
        save_path='saved'
        )

    hce.start(
        equalization={
            'fn': 'rgb_cv2_gray',
            'args': {}
        },
        channels=[0, 1, 2],
    )
