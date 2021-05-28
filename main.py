import torch
import numpy as np
from utility import runner
from PIL import Image
import os
import argparse
from preload import preloader


def process():
    current_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        default="./",
        help="path to image",
    )

    parser.add_argument(
        "--seg_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights of segmentation",
    )

    parser.add_argument(
        "--lpr_weights",
        type=str,
        required=True,
        default="./",
        help="Path to weights of segmentation",
    )

    args = parser.parse_args()

    model = preloader(args.seg_weights)
    with torch.no_grad():

        im = Image.open(args.image)
        frame = np.array(im)
        array_image, data_dictionary = runner(frame, model, args.lpr_weights)
        final_image = Image.fromarray(array_image, "RGB")
        final_image.save(f"{current_path}/result.png")  # for Separate Use
        print(data_dictionary)


if __name__ == "__main__":
    process()

# python main.py --image demo_images/test2.jpeg --seg_weights weights/hrnetv2_hrnet_plate_199.pth --lpr_weights weights/iter2.pth
