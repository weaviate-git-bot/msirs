#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
import skimage
from skimage.color.colorconv import rgb2gray, gray2rgb
import torch
import re, os, time
from torchvision import transforms
#from process_image import detect_regions
from process_image import detect_regions
from domars_map import MarsModel
from pathlib import Path
import weaviate
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 78256587200
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_THRESHOLD_CERT = 11
ROI_THRESHOLD_CERT_LOW = 2
ROI_THRESHOLD_CERT_HIGH = 10
HOME = str(Path.home())
classes = {
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
    "fsf": (152, 223, 138),
    "sfe": (196, 156, 148),
    "fsg": (214, 39, 40),
    "fse": (44, 160, 44),
    "fss": (255, 152, 150),
    "cra": (255, 187, 120),
    "sfx": (227, 119, 194),
    "mix": (148, 103, 189),
    "rou": (140, 86, 74),
    "smo": (247, 182, 210),
    "tex": (127, 127, 127),
}


interesting_classes = [
    color_info["aec"],
    color_info["ael"],
    color_info["cli"],
    color_info["rid"],
    color_info["fsf"],
    color_info["sfe"],
    color_info["fsg"],
    color_info["fse"],
    color_info["cra"],
    color_info["sfx"],
]


def setup_model():
    network_name = "densenet161"
    USER = "pg2022"
    # USER = "dusc"
    torch.multiprocessing.set_sharing_strategy("file_system")

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
        "model": network_name,
        "num_classes": 15,
        "pretrained": False,
        "transfer_learning": False,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MarsModel(hyper_params)
    print(f"/home/{USER}/models")
    checkpoint = torch.load(
        f"/home/{USER}/models/" + network_name + ".pth",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint)
    model = model.to(device)

    model.eval()
    print("DONE LOADING MODEL")
    return model, data_transform, device


def trim_edges(img: np.ndarray) -> np.ndarray:
    size_x = np.shape(img)[0]
    size_y = np.shape(img)[1]

    img = img[50 : size_x - 50, 50 : size_y - 50]
    return img


def check_img(file, model, data_transform, device) -> float:
    cert = 0.0
    ctx_image = data_transform(Image.open(file).convert("RGB"))

    test_img1 = ctx_image.unsqueeze(0)
    vec_rep = model(test_img1.to(device))
    pred = torch.argmax(vec_rep, dim=1).cpu()
    cert = torch.max(vec_rep).cpu().detach().numpy()
    print(f"CERT: {cert}")
    # cert = float(vec_rep.cpu()[pred])
    return cert


def check_cutout(file, box, model, data_transform, device):
    cert = 0.0
    ctx_image = Image.open(file).convert("RGB")
    ctx_image = np.array(ctx_image)[
        int(box[0]) : int(box[1]), int(box[2]) : int(box[3])
    ]
    ctx_image = data_transform(Image.fromarray(ctx_image))
    test_img1 = ctx_image.unsqueeze(0)
    vec_rep = model(test_img1.to(device))
    pred = torch.argmax(vec_rep, dim=1).cpu().detach().numpy()
    cert = torch.max(vec_rep).cpu().detach().numpy()
    # cert = float(vec_rep.cpu()[pred])
    return cert, pred


def get_model_descriptor(img: np.ndarray, model, data_transform, device):
    # file has to be OG img
    # preprocess imag

    print(f"Image size in descriptor function: { np.shape(img) }")
    ctx_image = data_transform(Image.fromarray(img).convert("RGB"))

    img = ctx_image.unsqueeze(0)
    desc = model.fc_layer_output2(img.to(device))

    print(f"Size of the descriptor: {np.shape(desc)}")
    return desc


def initial_rois(file):
    boxes = detect_regions(file)
    return boxes


def final_results() -> list:
    results = []
    return results


def process_image(file: str, model_stuff):
    model = model_stuff[0]
    data_transform = model_stuff[1]
    device = model_stuff[2]

    rois = []
    descriptors = []

    og_file = re.sub("mrf", "img", file)
    og_img = skimage.io.imread(og_file)
    og_img = rgb2gray(og_img[:, :, :3])
    # img_cert = check_img(og_file, model, data_transform, device)
    # og_file_name = og_file.split("/")[-1]
    # TODO: kinda useless for large images, delete this
    img_cert = 0
    if img_cert > IMAGE_THRESHOLD_CERT:
        # set higher requirements for ROI cert
        ROI_THRESHOLD_CERT = ROI_THRESHOLD_CERT_HIGH
    else:
        ROI_THRESHOLD_CERT = ROI_THRESHOLD_CERT_LOW

    start = time.monotonic()
    boxes = initial_rois(file)
    print(f"Created {len(boxes)} boxes in {time.monotonic()-start}s")
    for c in boxes:
        cutout = og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])]
        # print("Done with cutout")
        # FIXME: implement size filter and size adjustment in the bounding box creation
        if np.shape(cutout)[0] >= 50 and np.shape(cutout)[1] >= 50:
            # run through network
            cert, pred = check_cutout(og_file, c, model, data_transform, device)
            print("Cert: ",cert)
            # plt.imshow(og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])])
            # plt.title(f"{ cert }")
            # plt.show()

            if cert > ROI_THRESHOLD_CERT:
                # save ROI
                rois.append(c)
                desc = get_model_descriptor(
                    og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])],
                    model,
                    data_transform,
                    device,
                )
                desc = desc
                descriptors.append(desc)

                plt.imsave(
                    f"extracted/{classes[pred[0]]}_{cert}.png",
                    gray2rgb(og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])]),
                )
            # print("Done with chunk.")
    return rois, descriptors


if __name__ == "__main__":
    model, data_transform, device = setup_model()
    model_stuff = [model, data_transform, device]
    database_dir = f"{HOME}/segmentation/segmented/"
    file_list = os.listdir(database_dir)
    file_list = [database_dir + i for i in file_list if re.findall("mrf", i)]
    file_list = [
        "/home/pg2022/segmentation/segmented/presi_segment_og_densenet1614_mrf.png"
    ]   
    print(file_list)
    for file in file_list:
        print(file)
        img = skimage.io.imread(file)
        results, descriptors = process_image(file, model_stuff)
        print(f"Found {len(results)} interesting results")
        print(results)
        counter = 0
        for result in results:
            plt.imsave(f"segmentation_test_results/{counter}.png",img[result[0]:result[1],result[2]:result[3]])
            counter += 1
        # TODO: save to db
