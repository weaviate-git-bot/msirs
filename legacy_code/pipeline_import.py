#!/usr/bin/env python3

from pipeline_v2 import PipelineV2
from pathlib import Path
import os, re

HOME = str(Path.home())
if __name__ == "__main__":
    pipe = PipelineV2("http://localhost:8080")
    print("Instantiated pipeline")
    test_path = HOME + f"/segmentation/domars_benchmark/"
    # test_path = HOME + f"/segmentation/segmented/"
    all_imgs = os.listdir(test_path)
    all_imgs2 = [i for i in all_imgs if re.findall("img", i)]
    all_imgs = [i for i in all_imgs if re.findall("mrf", i)]

    all_imgs = [test_path + i for i in all_imgs][63:]
    all_imgs2 = [test_path + i for i in all_imgs2]
    all_imgs3 = [re.sub("img", "mrf", i) for i in all_imgs2]
    pipe.check_db()

    # pipe.build_db(all_imgs2, all_imgs3)
