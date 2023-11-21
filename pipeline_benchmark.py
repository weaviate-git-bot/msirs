#!/usr/bin/env python3

# Provide source image names
# find all samples from those images in dataset
# LOOP:
# do retrieval, get smallest distance value
# safe image and distance value

from pipeline_v2 import PipelineV2
from pathlib import Path
import os, re, pickle, glob
import skimage

HOME = str(Path.home())
IMPORTANT_CLASSES = [
    # "rid",
    "cra",
    # "cli",
    # "aec",
    # "ael",
    # "fsf",
    # "sfe",
    # "fsg",
    # "fse",
    # "sfx",
]


def benchmark_loop(imgs: list):
    images_path = HOME + "/segmentation/segmented/"
    res_dict = {}
    for img in imgs:
        if img.split("/")[-2] in IMPORTANT_CLASSES:
            img_name = img.split("/")[-1]
            print(f"Querying {img}")
            results, distances = pipe.query_image(img)
            print(f"Best distance: {distances[0]}")
            all_imgs = os.listdir(images_path)
            # TODO: persist image with distance info
            box = eval(results[0][1])
            print(f"{ box = }")
            res_name = results[0][0].split("/")[-1]
            res = [
                i for i in all_imgs if re.findall(res_name, i) and re.findall("img", i)
            ][0]

            cutout = skimage.io.imread(images_path + res)
            cutout = cutout[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]
            res_dict[img_name] = [res, distances[0]]
            print("Added into result dict..")

        # persist dict
    with open("pipeline_benchmark_results.pickle", "wb") as f:
        pickle.dump(res_dict, f)


def find_domars_samples(
    img_name: str, search_directory="/home/pg2022/data/data/"
) -> list:
    all_imgs = glob.glob(search_directory + "**/*.jpg", recursive=True)
    imgs = [i for i in all_imgs if re.findall(img_name, i)]
    print(imgs)
    return imgs


if __name__ == "__main__":
    pipe = PipelineV2("http://localhost:8080")
    print("Instantiated pipeline")
    test_path = HOME + f"/segmentation/segmented/"
    all_imgs = os.listdir(test_path)
    all_imgs2 = [i for i in all_imgs if re.findall("img", i)]
    all_imgs = [i for i in all_imgs if re.findall("mrf", i)]

    all_imgs = [test_path + i for i in all_imgs][63:]
    all_imgs2 = [test_path + i for i in all_imgs2]
    all_imgs3 = [re.sub("img", "mrf", i) for i in all_imgs2]
    pipe.check_db()

    imgs = find_domars_samples("D09_030608_1812_XI_01N359W")
    imgs = find_domars_samples("D01_027450_2077_XI_27N186W")
    imgs = find_domars_samples("F01_036186_1762_XI_03S004W")

    imgs = imgs
    print(f"{imgs = }")

    benchmark_loop(imgs)
