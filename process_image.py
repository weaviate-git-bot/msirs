#!/usr/bin/env python3

from matplotlib import validate_backend
import numpy as np
import skimage.io
import glob, re
import matplotlib.pyplot as plt
from skimage import feature
from skimage import measure
import time, copy, cv2
from sklearn.cluster import DBSCAN, OPTICS
import matplotlib.patches as mpatches
from PIL import Image


CATEGORIES = {
    0: "aec",
    1: "ael",
    2: "cli",
    3: "cra",
    4: "fse",
    5: "fsf",
    6: "fsg",
    7: "fss",
    8: "mix",
    9: "rid",
    10: "rou",
    11: "sfe",
    12: "sfx",
    13: "smo",
    14: "tex",
}


color_info = {
    "aec": (31, 119, 180),
    "ael": (174, 199, 232),
    "cli": (255, 127, 14),
    "rid": (197, 176, 213),
    "fsf": (152, 223, 138),  # DONE
    "sfe": (196, 156, 148),
    "fsg": (214, 39, 40),
    "fse": (44, 160, 44),
    "fss": (255, 152, 150),
    "cra": (255, 187, 120),
    "sfx": (227, 119, 194),  # DONE
    "mix": (148, 103, 189),
    "rou": (140, 86, 74),  # DONE
    "smo": (247, 182, 210),
    "tex": (127, 127, 127),  # DONE
}


interesting_classes = [
    color_info["cra"],
    color_info["aec"],
    color_info["ael"],
    color_info["cli"],
    color_info["rid"],
    color_info["fsf"],
    color_info["sfe"],
    color_info["fsg"],
    color_info["fse"],
    color_info["sfx"],
]


def post_process(img_file):
    pp_info = {}
    img = Image.open(img_file).convert("RGB")
    img = np.array(img)
    color_list = [color_info[CATEGORIES[i]] for i in CATEGORIES.keys()]
    for color in color_list:
        col_sum = img[np.where(np.all(img == color, axis=-1))]
        pp_info[color] = np.sum(col_sum) / np.sum(img)

    return pp_info


def find_bounding_box(img, val):
    # builds a box around an area of a certain color (denoted by val)
    min_x = np.shape(img)[0] + 1
    min_y = np.shape(img)[1] + 1
    max_x = -1
    max_y = -1

    # TODO: get min and max coordinates of pixels of certain color
    idx = np.where(
        (img[:, :, 0] == val[0]) & (img[:, :, 1] == val[1]) & (img[:, :, 2] == val[2])
    )
    min_x = np.min(idx[0])
    max_x = np.max(idx[0])
    min_y = np.min(idx[1])
    max_y = np.max(idx[1])
    # print(f"Found bounding box: {min_x},{max_x},{min_y},{max_y}")
    box_size_x = max_x - min_x
    box_size_y = max_y - min_y
    border_x = 0
    border_y = 0
    if box_size_x < 50 or box_size_y < 50:
        # too small dont use
        # gets filtered out in detect_region method
        return [1, 0, 1, 0]
    elif box_size_x < 100 or box_size_y < 100:
        border_x = 200 - box_size_x
        border_y = 200 - box_size_y

    elif box_size_x < 200 or box_size_y < 200:
        border_x = 200 - box_size_x
        border_y = 200 - box_size_y

    if min_y >= border_y:
        min_y = min_y - border_y
    else:
        min_y = 0
    if min_x >= border_x:
        min_x = min_x - border_x
    else:
        min_x = 0
    if max_x < np.shape(img)[1] - border_x:
        max_x = max_x + border_x
    else:
        max_x = np.shape(img)[1]
    if max_y < np.shape(img)[0] - border_y:
        max_y = max_y + border_y
    else:
        max_y = np.shape(img)[0]

    return [min_x, max_x, min_y, max_y]


def detect_regions(img_path):
    # TODO: make bounding box at least 50x50
    # TODO: kind of exclude the edge of the image. cut off 50 edge pixels for ROI preselection, but add back for creation of bounding boxes
    img = skimage.io.imread(img_path)
    fig = plt.figure()
    plt.imshow(img)
    plt.title("Original Image")
    plt.show()
    unique_vals = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    print(len(unique_vals))
    print(interesting_classes)
    boxes = []
    if len(unique_vals) > 1:
        for val in unique_vals:
            if tuple(val[:3]) in interesting_classes:
                print("HEYO")
                c_img = copy.deepcopy(img)
                for v in range(len(val)):
                    c_img[c_img[:, :, v] != val[v]] = np.array([0, 0, 0, 255])
                    fig = plt.figure()
                    plt.imshow(c_img)
                    plt.title(f"Segments of {val}")
                    plt.show()

                pix_to_cluster = []
                for x in range(np.shape(c_img)[0]):
                    for y in range(np.shape(c_img)[1]):
                        if list(c_img[x, y]) == list(val):
                            pix_to_cluster.append([x, y])

                optics = DBSCAN(metric="cityblock", eps=2)
                res = optics.fit(np.array(pix_to_cluster)).labels_
                print(res)
                num_of_seg = len(np.unique(res))
                # print(f"There are {num_of_seg} segements")
                if num_of_seg > 1:
                    l_img = copy.deepcopy(c_img)
                    for idx in range(len(pix_to_cluster)):
                        if res[idx] >= 0:
                            l_img[pix_to_cluster[idx][0], pix_to_cluster[idx][1], :] = [
                                res[idx] * 10 + 100,
                                res[idx] * 100 + 1,
                                res[idx] * 10 + 100,
                                255,
                            ]
                    seg_colors = unique_vals = np.unique(
                        l_img.reshape(-1, l_img.shape[2]), axis=0
                    )
                    seg_colors = [i for i in seg_colors if i[0] != 0]

                    plt.imshow(l_img)
                    plt.title(f"Detected patches of identical color")
                    plt.show()

                    for sc in seg_colors:
                        box = find_bounding_box(l_img, sc)
                        if box[1] - box[0] > 0 and box[3] - box[2] > 0:
                            boxes.append(box)

                else:
                    # only one segment, create bounding box
                    box = find_bounding_box(c_img, val)
                    if box[1] - box[0] > 0 and box[3] - box[2] > 0:
                        boxes.append(box)

    return boxes


def extract_regions(img_path, boxes, interesting_classes, color_info):
    img_path_og = re.sub("mrf", "og_img", img_path)
    img = skimage.io.imread(img_path_og)
    mrf = skimage.io.imread(img_path)
    cutouts = []
    region_info = []
    for box in boxes:
        patch = mrf[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]

        if tuple(patch[0, 0, :][:-1]) in interesting_classes:
            cutouts.append(img[int(box[0]) : int(box[1]), int(box[2]) : int(box[3]), :])
            region_info.append(
                list(color_info.keys())[
                    list(color_info.values()).index((tuple(patch[0, 0, :][:-1])))
                ]
            )
    return cutouts, region_info


if __name__ == "__main__":
    USER = "dusc"
    file_names_mrf = [
        "/Users/dusc/segmentation/segmented/ESP_061636_1615_RED_img_row_22528_col_4096_w_1024_h_1024_x_0_y_0_densenet1612_mrf.png"
    ]
    # file_names_mrf = ["../segmentation/segmenttest.png"]
    file_names_mrf = [
        "/Users/dusc/segmentation/segmented/ESP_046128_2465_RED_img_row_2048_col_7168_w_1024_h_1024_x_0_y_0_densenet1611_mrf.png"
    ]

    for file in file_names_mrf:
        boxes = detect_regions(file)
        print(boxes)
        cutouts, region_info = extract_regions(
            file, boxes, interesting_classes, color_info
        )
        print(file)
        # print(boxes)
        print(region_info)
        # print(cutouts)

        if len(cutouts) > 1:
            img = skimage.io.imread(file)
            fig = plt.imshow(img)
            plt.title("OG image")
            plt.show()
            fig = plt.figure()
            rows = 4
            cols = 2
            counter = 1
            for cutout in cutouts:
                fig.add_subplot(rows, cols, counter)
                plt.imshow(cutout)
                plt.title("Cutouts")
                plt.axis("off")
                counter += 1
            plt.show()
