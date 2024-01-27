import random
import cv2
import numpy as np
from glob import glob
from image import Image as im
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal

@dataclass
class ColorHistogram:
    @staticmethod
    def rgb(image, channel):
        rgb_channel = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, :, channel]
        histogram = cv2.calcHist([rgb_channel], [0], None, [256], [0, 256])
        return histogram.flatten().astype(np.uint32)

    @staticmethod
    def rgb_cv2_gray(image):
        rgb_channel = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        histogram = cv2.calcHist([rgb_channel], [0], None, [256], [0, 256])
        return histogram.flatten().astype(np.uint32)
    
    def rgb_weighted_gray(image, weights):
        weights = np.array(weights) / np.sum(weights)
        rw, gw, bw = weights
        wg_image = (image[:, :, 0] * rw + image[:, :, 1] * gw + image[:, :, 2] * bw)
        histogram = cv2.calcHist([wg_image.astype(np.uint8)], [0], None, [256], [0, 256])
        return histogram.flatten().astype(np.uint32)
    
    @staticmethod
    def hsv(image, channel):
        hsv_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, channel]
        histogram = cv2.calcHist([hsv_channel], [0], None, [256], [0, 256])
        return histogram.flatten().astype(np.uint32)
    
    @staticmethod   
    def lab(image, channel):
        lab_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, channel]
        histogram = cv2.calcHist([lab_channel], [0], None, [256], [0, 256])
        return histogram.flatten().astype(np.uint32)

@dataclass
class ColorEqualization:
    images: list[str]
    save_path: str
    batch_size: int = 3

    def plot_histogram(self, channel, ranges=None, show=True, color='blue'):
        plt.plot(channel)
        if ranges is not None:
            plt.plot(ranges, channel[ranges], "x", c=color)
        plt.show() if show else None


    def find_equalization_ranges(self, histogram):
        histogram = np.convolve(histogram, np.ones(5), mode='same')
        histogram = histogram / np.max(histogram) * 255

        grad = np.gradient(histogram)
        # firs grad increase > 0.1
        range_start = np.where(grad > 0.1)[0][0]

        valleys = signal.find_peaks(-histogram)[0].tolist()
        max_peak_index = int(np.argmax(histogram))
        sorted_valleys_and_max_peak = np.array(sorted(valleys+[max_peak_index]))

        # left valley near to max peak
        range_end = sorted_valleys_and_max_peak[
            np.where(sorted_valleys_and_max_peak == max_peak_index)[0][0] - 1
        ]
        
        #self.plot_histogram(histogram, [range_start], show=False, color='red')
        #self.plot_histogram(histogram, range_end, show=True, color='green')
        
        return [range_start, range_end], histogram


    def back_projection(self, image, ranges, space_name, channels):
        transformed_space = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{space_name}"))
        for channel in channels:
            value_channel = transformed_space[:, :, channel]
            clipped_channel = np.clip(value_channel, ranges[0], ranges[1])
            normalized_channel = cv2.normalize(clipped_channel, None, 0, 255, cv2.NORM_MINMAX)
            transformed_space[:, :, channel] = normalized_channel
        
        equalized_image = cv2.cvtColor(transformed_space, getattr(cv2, f"COLOR_{space_name}2BGR"))
        return equalized_image

    def start(self, equalization, channels):
        for image_path in tqdm(self.images):
            image = cv2.imread(image_path)
            histogram = getattr(ColorHistogram, equalization['fn'])(image, **equalization['args'])
            ranges, histogram = self.find_equalization_ranges(histogram)
            
            space_name = equalization['fn'].split('_')[0].upper()
            equalized_image = self.back_projection(image, ranges, space_name, channels)
            #im.show_pillow(equalized_image)
            #im.save(equalized_image, f"{self.save_path}/{equalization['fn']}_{image_path.split('/')[-1]}")


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
