#!/usr/bin/env python3

import glob, re
import skimage
from skimage.exposure import equalize_adapthist
from pathlib import Path

from skimage.util.dtype import img_as_float

HOME = str(Path.home())
if __name__ == "__main__":
    direc = "/Users/dusc/DoMars_Stripes/"
    chosen_source_name = [
        "b06_011909_1323_xn_47s329w",
        "d09_030608_1812_xi_01n359w",
        "f01_036186_1762_xi_03s004w",
        "p03_002147_1865_xi_06n208w",
        "b01_009849_2352_xn_55n263w",
    ]
    files = [direc + i + ".jpg" for i in chosen_source_name]
    rois_to_find = [106, 1247, 124, 238, 8]
    print(sum(rois_to_find))
    domars_dir = HOME + "/codebase-v1/data/data/"

    for file in files:
        img = skimage.io.imread(file)
        img = equalize_adapthist(img, clip_limit=0.03)
        img = (img * 255).astype("uint8")
        print(img[0, 0])
        file_name = file.split(".")[0] + "_cont.jpg"
        skimage.io.imsave(file_name, img)

    # print(domars_dir)
    # file_names = glob.glob(domars_dir + "/**/*.jpg", recursive=True)
    # c = 0
    # for file in file_names:
    #     if re.findall(target_img, file):
    #         c += 1
    #         print(file)
    # print(c)
