import random
from align import CoinAlign
from hist_eq import ColorEqualization
from glob import glob
from pathlib import Path
from tqdm import tqdm
from image import Image as im

def equalize_and_align(images_path, save_folder_path, equalization, channels, align_margin, samples=None):
    images_paths = glob(f'{images_path}/*.*')
    if samples is not None:
        images_paths = random.sample(images_paths, samples)
    for image_path in tqdm(images_paths):
        equalized_image = ColorEqualization.start(
            image_path,
            equalization=equalization,
            channels=channels
        )
        aligned_image = CoinAlign.align(equalized_image, margin=align_margin)
        save_path = f"{save_folder_path}/{Path(image_path).name}"
        im.save(aligned_image, save_path)
    with open(f"{save_folder_path}/info.txt", 'w') as f:
        f.write(f"equalization={equalization}\n")
        f.write(f"channels={channels}\n")
