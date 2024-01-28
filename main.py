import random
import contextlib
from align import CoinAlign
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

def equalize_and_align(images_path, save_folder_path, equalization, channels, align_margin, samples=None):
    images_paths = glob(f'{images_path}/*.*')
    if samples is not None:
        images_paths = random.sample(images_paths, samples)
    for image_path in tqdm(images_paths):
        #with timer():
        equalized_image = ColorEqualization.start(
            image_path,
            equalization=equalization,
            channels=channels
        )
        #with timer():
        aligned_image = CoinAlign.align(equalized_image, margin=align_margin)
        #with timer():
        save_path = f"{save_folder_path}/{Path(image_path).name}"
        im.save(aligned_image, save_path)
    with open(f"{save_folder_path}/info.txt", 'w') as f:
        f.write(f"equalization={equalization}\n")
        f.write(f"channels={channels}\n")


equalize_and_align(
    images_path='teste',
    save_folder_path='results_2',
    equalization={
        'fn': 'rgb_weighted_gray',
        'args': {'weights': [1, 1, 1]}
    },
    channels=[0, 1, 2],
    align_margin=0.005,
    samples=None
)
