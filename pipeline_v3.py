#!/usr/bin/env python3

import shutil
from sys import exception
import numpy as np
import weaviate, re, os, random, pickle
import skimage.io
from matplotlib import pyplot as plt
from process_image import detect_regions, extract_regions, post_process
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import glob, time
from domars_map import MarsModel, HIRISE_Image, segment_image
from process_image2 import process_image
from process_image import detect_regions
from PIL import Image
from scipy import spatial
from pathlib import Path
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from skimage.color import rgb2gray, gray2rgb
from process_image2 import initial_rois, check_cutout

from weaviate_client import WeaviateClient
import tensorflow as tf
from senet_model import SENet

IMAGE_THRESHOLD_CERT = 11
ROI_THRESHOLD_CERT_LOW = 5
ROI_THRESHOLD_CERT_HIGH = 10
CERT_THRESHOLD = 8

HOME = str(Path.home())

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
]

# TODO: add path here as constant
MODEL_PATH = ""


class PipelineV3:
    def __init__(self, db_adr: str, schema=None, model_path=None):
        self.client = WeaviateClient(db_adr, schema)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

        if model_path == None:
            model_path = MODEL_PATH

        self.model = SENet(model_path=model_path)

    def query_image(self, img: np.array) -> bool:
        try:
            vector = self.model.get_descriptor(img)
            self.client.query_image(vector)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def build_database(self, image_files: list) -> bool:
        excep = False
        for img_file in image_files:
            try:
                img = skimage.io.imread(img_file)
                self.client.add_to_db(img)

            except Exception as e:
                print(f"Skipping {img_file}: {e}")
                excep = True

        return excep

    def get_certainty(self) -> None:
        # TODO: is this needed?
        pass

    def store_for_ui(self):
        # TODO: implement
        pass

    @staticmethod
    def clear_queue():
        # TODO: implement
        pass


if __name__ == "__main__":
    pipe = PipelineV3("http://localhost:8080")

    pipe.client.check_db()
