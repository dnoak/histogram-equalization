import shutil
import cv2
import os
import random
import contextlib
from legacy.align import CoinAlign
from hist_eq import ColorEqualization
from glob import glob
from pathlib import Path
from tqdm import tqdm
from image import Image as im
from timeit import default_timer

@contextlib.contextmanager
def timer(message="Time"):
    t0 = default_timer()
    yield
    t1 = default_timer()
    print(f"{message}: {t1 - t0:.2f}s\n")

def equalize_and_align(
        images_path, save_folder_path, equalization, 
        channels, show_histogram, align_margin, samples=None, random_seed=1010):
    random.seed(random_seed)
    images_paths = glob(f'{images_path}/*.*')

    os.makedirs(f"{save_folder_path}", exist_ok=True)
    os.makedirs(f"{save_folder_path}/error", exist_ok=True)

    if samples is not None:
        images_paths = random.sample(images_paths, samples)

    for image_path in tqdm(images_paths):
        try:
            equalized_image = ColorEqualization.start(
                image_path=image_path,
                equalization=equalization,
                channels=channels,
                show_histogram=show_histogram)
            aligned_image = CoinAlign.align(equalized_image, margin=align_margin)
            save_path = f"{save_folder_path}/{Path(image_path).name}"
            im.save(aligned_image, save_path)
        except Exception as e:
            print(e)
            print(f"Error processing {image_path}")
            try:
                save_path = f"{save_folder_path}/error/{Path(image_path).name}"
                shutil.copy(image_path, save_path)
            except Exception as e:
                print(e)
                with open(f"{save_folder_path}/error/error_on_save.txt", 'a+') as f:
                    f.write(f"{image_path}\n")

    with open(f"{save_folder_path}/equalization_info.txt", 'w') as f:
        f.write(f"equalization={equalization}\n")
        f.write(f"channels={channels}\n")
