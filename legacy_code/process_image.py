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
import seaborn as sns


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
    # TODO: pass all vals and loop over them in here for better performance
    # builds a box around an area of a certain color (denoted by val)
    img_x = np.shape(img)[1]
    img_y = np.shape(img)[0]

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
    # elif box_size_x < 100 or box_size_y < 100:
    #     border_x = 200 - box_size_x
    #     border_y = 200 - box_size_y
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
    if max_x < img_x - border_x:
        max_x = max_x + border_x
    else:
        max_x = img_x
    if max_y < img_y - border_y:
        max_y = max_y + border_y
    else:
        max_y = img_y

    return [min_x, max_x, min_y, max_y]


def detect_regions(img_path):
    print("Entering detect_region method...")
    img = skimage.io.imread(img_path)
    print("Loaded image")
    # fig = plt.figure()
    # plt.imshow(img)
    # plt.title("og")
    # plt.show()
    start = time.monotonic()
    # unique_vals = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    int_vals = interesting_classes
    boxes = []
    for val in int_vals:
        if val[:3] in img[:, :, :3]:
            try:
                start = start_col = time.monotonic()
                c_img = copy.deepcopy(img)[:, :, :3]
                print(f"Created copy of img in {time.monotonic()-start}s")
                start = time.monotonic()
                # c_img = np.where(
                #     c_img != np.array(list(val)), np.array(list(val)), np.array([0, 0, 0])
                # )
                c_img = np.all(c_img == val, axis=-1)
                print(f"Initial recoloring took {time.monotonic()-start}s")
                plt.imsave("recoloring_test_1.png", c_img.astype("uint8"))
                pix_to_cluster = []
                start = time.monotonic()
                pix_to_cluster = np.where(c_img == True)
                pix_to_cluster_arr = copy.deepcopy(pix_to_cluster)
                pix_to_cluster = [
                    [pix_to_cluster[0][i], pix_to_cluster[1][i]]
                    for i in range(len(pix_to_cluster[0]))
                ]
                print(f"Extracted pixels to cluster in {time.monotonic()-start}s")
                start = time.monotonic()
                optics = DBSCAN(metric="cityblock", eps=1, min_samples=5)
                print(f"Done with clustering in {time.monotonic()-start}")
                res = optics.fit(np.array(pix_to_cluster)).labels_
                res_labels = np.unique(res)
                print(f"LABELS: { res_labels }")

                print(f"Finished to bounding box in {time.monotonic()-start_col}s!")
                if len(res_labels) > 1:
                    l_img = np.zeros(np.shape(img))[:, :, :3]
                    palette = sns.color_palette(None, len(res_labels))
                    seg_colors = []
                    start = time.monotonic()
                    # FIXME: this doesnt work yet
                    # colors = [[res[i]] * 3 for i in range(len(pix_to_cluster))]
                    # l_img[pix_to_cluster_arr] = colors
                    # plt.imsave("wow_much_optimization.png", l_img.astype("uint8"))
                    for idx in range(len(pix_to_cluster)):
                        if res[idx] >= 0:
                            color = [res[idx] + 1] * 3
                            if color not in seg_colors:
                                seg_colors.append(color)
                            l_img[
                                pix_to_cluster[idx][0], pix_to_cluster[idx][1], :
                            ] = color
                    print(f"Final recoloring took {time.monotonic()-start}s")

                    print(len(seg_colors))
                    for sc in seg_colors:
                        start = time.monotonic()
                        box = find_bounding_box(l_img, sc)
                        print(f"Found bounding box in {time.monotonic()-start}s")
                        if box[1] - box[0] > 0 and box[3] - box[2] > 0:
                            boxes.append(box)

                else:
                    # only one segment, create bounding box
                    start = time.monotonic()
                    c_img2 = np.zeros(np.shape(img))
                    c_img2[:, :, 0] = c_img * val[0]
                    c_img2[:, :, 1] = c_img * val[1]
                    c_img2[:, :, 2] = c_img * val[2]
                    print(c_img2[0:4, 0:4, :])
                    box = find_bounding_box(c_img2, val)
                    print(f"Found bounding box in {time.monotonic()-start}s")
                    if box[1] - box[0] > 0 and box[3] - box[2] > 0:
                        boxes.append(box)
            except Exception:
                pass
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

    #     if len(cutouts)>1:
    #         fig = plt.imshow(img)
    #         plt.title("OG image")
    #         plt.show()
    #         fig = plt.figure()
    #         rows = 4
    #         cols = 2
    #         counter = 1
    #         for cutout in cutouts:
    #             fig.add_subplot(rows, cols, counter)
    #             plt.imshow(cutout)
    #             plt.title("Cutouts")
    #             plt.axis("off")
    #             counter += 1
    #         plt.show()
