#!/usr/bin/env python3
from pipeline_v2 import PipelineV2
from pathlib import Path
import os, re

HOME = str(Path.home())
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

    query_path = HOME + "/query/"
    img = query_path + os.listdir(query_path)[0]
    img_name = img.split("/")[-1:][0]
    img = query_path + img_name

    results, distances = pipe.query_image(img)
    print("AWOOOOOOOOOGA")
    pipe.store_for_ui(HOME + f"/server-test/", results, img)
    pipe.clear_queue()
