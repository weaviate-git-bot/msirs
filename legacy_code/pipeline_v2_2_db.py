#!/usr/bin/env python3

import shutil
import numpy as np
import weaviate, re, os, random
import skimage.io
from matplotlib import pyplot as plt
from process_image import detect_regions, extract_regions, post_process
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from domars_map import MarsModel, HIRISE_Image, segment_image
from process_image2 import process_image
from process_image import detect_regions
from PIL import Image
from scipy import spatial
from pathlib import Path
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

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


data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

hyper_params = {
    "batch_size": 64,
    "num_epochs": 15,
    "learning_rate": 1e-2,
    "optimizer": "sgd",
    "momentum": 0.9,
    "model": "densenet161",
    "num_classes": 15,
    "pretrained": False,
    "transfer_learning": False,
}


class PipelineV2_2:
    def __init__(self, db_adr: str):
        self.client = weaviate.Client(db_adr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        model = MarsModel(hyper_params)
        print(HOME + "/models")
        checkpoint = torch.load(
            HOME + "/models/" + hyper_params["model"] + ".pth",
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        self.model = model
        self.model.eval()
        return_nodes = {"features.denseblock4.denselayer24.conv2": "layer4"}
        self.fe = create_feature_extractor(self.model.net, return_nodes=return_nodes)

    def add_to_db(self, data_object: dict, vec: list):
        self.client.data_object.create(data_object, "DoMars16k", vector=vec)

    def query_image(self, file: str, num_to_retrieve=10):
        # preprocess file
        ctx_image = data_transform(Image.open(file).convert("RGB"))
        img = ctx_image.unsqueeze(0)

        # create descriptor
        img_desc = self.fe(img.to(self.device))
        img_desc = img_desc["layer4"].cpu().detach().numpy()
        img_desc = img_desc.reshape(-1)
        print("#############")
        print("QUERY IMAG DESC SUM")
        print(sum(img_desc))
        print("#############")
        desc_vec = {"vector": img_desc}

        result = (
            self.client.query.get("DoMars16k", ["sourceName"])
            .with_near_vector(desc_vec)
            .with_additional(["distance"])
            .with_limit(num_to_retrieve)
            .do()
        )
        # print(result)
        res = [
            [i["sourceName"], i["coordinates"]]
            for i in result["data"]["Get"]["DoMars16k"]
        ]
        distances = [
            i["_additional"]["distance"]
            for i in result["data"]["Get"]["DoMars16k"]
        ]
        print(f"Distances to query: {distances}")
        return res

    def build_db(self, img_files: list, mrf_files: list):
        for file_idx in range(len(img_files)):
            file_name = img_files[file_idx][:-21]
            descriptors, boxes = self.determine_rois(
                img_files[file_idx], mrf_files[file_idx]
            )
            print(f"Found {len(boxes)} ROIs!")
            for item in range(len(boxes)):
                if self.check_for_existing_entries(
                    file_name, boxes[item], descriptors[item]
                ):
                    data = {"SourceName": file_name, "Coordinates": str(boxes[item])}
                    self.add_to_db(data, descriptors[item])

    def build_dm16k_db(self, file_name):
        img = skimage.io.imread(file_name)
        img = data_transform(Image.fromarray(img).convert("RGB"))
        img = img.unsqueeze(0)
        descriptor = self.get_descriptor(img)
        data = {"SourceName": file_name}
        self.add_to_db(data, descriptor)

    def check_for_existing_entries(
        self, source_name: str, box: str, vector: list
    ) -> bool:
        vec = {"vector": vector}
        results = (
            self.client.query.get("DoMars16k", ["sourceName"])
            .with_near_vector(vec)
            .do()
        )
        res = [
            i["sourceName"]
            for i in results["data"]["Get"]["DoMars16k"]
            if i["sourceName"] == source_name and eval(i["coordinates"]) == box
        ]
        if len(res) < 1:
            print("New entry. Adding to database...")
            return True
        else:
            print("Entry exists already, skipping...")
            return False

    def check_db(self):
        result = (
            self.client.query.aggregate("DoMars16k")
            .with_fields("meta {count}")
            .do()
        )
        print(result)

    def get_descriptor(self, x):
        with torch.no_grad():
            descriptor = self.fe(x.to(self.device))
            descriptor = descriptor["layer4"].cpu().detach().numpy()
            descriptor = descriptor.reshape(-1)
            print("###########")
            print("DESC SUM")
            print(sum(descriptor))
            print("###########")
        return descriptor

    def get_certainty(self, x):
        certainty = self.model(x.to(self.device))
        certainty = torch.max(certainty).cpu().detach().numpy()
        print("###########")
        print("CERT")
        print(certainty)
        print("###########")
        return certainty

    @staticmethod
    def process_image(mrf_file: str):
        boxes = detect_regions(mrf_file)
        return boxes

    def determine_rois(self, img_file: str, mrf_file: str):
        boxes = self.process_image(mrf_file)
        img = skimage.io.imread(img_file)
        descriptors = []
        chosen_boxes = []
        for box in boxes:
            cutout = img[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]
            cutout = data_transform(Image.fromarray(cutout).convert("RGB"))
            cutout = cutout.unsqueeze(0)
            certainty = self.get_certainty(cutout)
            if certainty > CERT_THRESHOLD:
                descriptors.append(self.get_descriptor(cutout))
                chosen_boxes.append(box)
        return descriptors, chosen_boxes

    def store_for_ui(self, folder: str, results: list, query: str):
        # clean up old results
        print(results)
        images_path = HOME + "/segmentation/segmented/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # TODO: add query image in this folder as well
        print(query)
        file_format = query.split(".")[-1]
        print(file_format)
        shutil.copy(query, folder + f"query.{file_format}")
        counter = 1
        for result in results:
            all_imgs = os.listdir(images_path)
            box = result[1]
            box = eval(box)
            img_name = result[0].split("/")[-1]
            img = [
                i for i in all_imgs if re.findall(img_name, i) and re.findall("img", i)
            ][0]
            cutout = skimage.io.imread(images_path + img)
            cutout = cutout[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]
            skimage.io.imsave(folder + f"retrieval_{counter}.png", cutout)
            counter += 1

    @staticmethod
    def clear_queue():
        folder = HOME + "/query/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def descriptor_test(self, image_list: list):
        for file in image_list:
            ctx_image = data_transform(Image.open(file).convert("RGB"))
            x = ctx_image.unsqueeze(0)
            with torch.no_grad():
                descriptor = self.fe(x.to(self.device))
                descriptor = descriptor["layer4"].cpu().detach().numpy()
                descriptor = descriptor.reshape(-1)
                print(np.shape(descriptor))
                print(sum(descriptor))

    def descriptor_test2(self, image_list_mrf: list, image_list_img: list):
        model_stuff = [self.model, data_transform, self.device]
        for file_idx in range(len(image_list_img)):
            results, descriptors = process_image(image_list_mrf[file_idx], model_stuff)
            for result in results:
                img = skimage.io.imread(image_list_img[file_idx])[
                    int(result[0]) : int(result[1]), int(result[2]) : int(result[3])
                ]
                img = data_transform(Image.fromarray(img).convert("RGB"))
                test_img1 = img.unsqueeze(0)
                with torch.no_grad():
                    descriptor = self.fe(test_img1.to(self.device))
                    descriptor = descriptor["layer4"].cpu().detach().numpy()
                    descriptor = descriptor.reshape(-1)
                    print("#####################")
                    print("#####################")
                    print(np.shape(descriptor))
                    print(sum(descriptor))
                    print("#####################")
                    print("#####################")


if __name__ == "__main__":
    pipe = PipelineV2_2("http://localhost:8080")
    print("Instantiated pipeline")
    test_path = HOME + f"/segmentation/domars_benchmark/"
    all_imgs = os.listdir(test_path)
    all_imgs2 = [i for i in all_imgs if re.findall("img", i)]
    all_imgs = [i for i in all_imgs if re.findall("mrf", i)]

    all_imgs = [test_path + i for i in all_imgs][63:]
    all_imgs2 = [test_path + i for i in all_imgs2]
    all_imgs3 = [re.sub("img", "mrf", i) for i in all_imgs2]
    pipe.check_db()
    path = HOME + f"/codebase-v1/data/data/train"
    all_files = glob.glob(path+"/**/*.jpg",recursive=True)
    all_files = [i for i in all_files if re.findall("sample",i)]
    print(all_files)
    #pipe.build_db(all_imgs2, all_imgs3)
    #pipe.build_dm16k_db(all_files[0])
    for file in all_files:
        pipe.build_dm16k_db(file)
