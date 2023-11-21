#!/usr/bin/env python3

# Credit to: http://github.com/thowilh/geomars & http://dx.doi.org/10.5281/zenodo.4291940

import torch
import numpy as np
import os, random

# import cv2
from PIL import Image
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor

user = "pg2022"

# from pytorch_lightning.metrics.functional import accuracy, precision_recall
from torchvision.models import (
    alexnet,
    vgg16_bn,
    resnet18,
    resnet34,
    resnet50,
    densenet121,
    densenet161,
)


class MarsModel(pl.LightningModule):
    def __init__(self, hyper_param):
        super().__init__()
        self.momentum = hyper_param["momentum"]
        self.optimizer = hyper_param["optimizer"]
        self.lr = hyper_param["learning_rate"]
        self.num_classes = hyper_param["num_classes"]

        if hyper_param["model"] == "resnet18":
            """
            Resnet18
            """
            self.net = resnet18(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)

            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "resnet34":
            """
            Resnet34
            """
            self.net = resnet34(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "resnet50":
            """
            Resnet50
            """
            self.net = resnet50(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "alexnet":
            """
            Alexnet
            """
            self.net = alexnet(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier[6].in_features
            self.net.classifier[6] = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "vgg16":
            """
            VGG16_bn
            """
            self.net = vgg16_bn(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier[6].in_features
            self.net.classifier[6] = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "densenet121":
            """
            Densenet-121
            """
            self.net = densenet121(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier.in_features
            self.net.classifier = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "densenet161":
            """
            Densenet-161
            """
            self.net = densenet161(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier.in_features
            self.net.classifier = nn.Linear(num_ftrs, hyper_param["num_classes"])

        else:
            print("Invalid model name, exiting...")
            exit()

    def forward(self, x):
        return self.net(x)

    def fc_layer_output(self, x):
        out = self.net.features.conv0.forward(x)
        out = self.net.features.norm0.forward(out)
        out = self.net.features.relu0.forward(out)
        out = self.net.features.pool0.forward(out)
        out = self.net.features.denseblock1.forward(out)
        out = self.net.features.transition1.forward(out)
        out = self.net.features.denseblock2.forward(out)
        out = self.net.features.transition2.forward(out)
        out = self.net.features.denseblock3.forward(out)
        out = self.net.features.transition3.forward(out)
        out = self.net.features.denseblock4.forward(out)
        # out = self.net.features.norm5.forward(out)
        out = out.cpu()
        out = out.detach().numpy()

        return out

    def fc_layer_output2(self, x):
        return_nodes = {"features.denseblock4.denselayer24.conv2": "layer4"}
        self.fe = create_feature_extractor(self.net, return_nodes=return_nodes)

        with torch.no_grad():
            descriptor = self.fe(x)
            descriptor = descriptor["layer4"].cpu().detach().numpy()
            descriptor = descriptor.reshape(-1)
            print(sum(descriptor))
            return descriptor


# network_name = "densenet161"

# classes = {
#     0: "aec",
#     1: "ael",
#     2: "cli",
#     3: "cra",
#     4: "fse",
#     5: "fsf",
#     6: "fsg",
#     7: "fss",
#     8: "mix",
#     9: "rid",
#     10: "rou",
#     11: "sfe",
#     12: "sfx",
#     13: "smo",
#     14: "tex",
# }

# data_transform = transforms.Compose(
#     [
#         transforms.Resize([224, 224]),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# hyper_params = {
#     "batch_size": 64,
#     "num_epochs": 15,
#     "learning_rate": 1e-2,
#     "optimizer": "sgd",
#     "momentum": 0.9,
#     "model": network_name,
#     "num_classes": 15,
#     "pretrained": True,
#     "transfer_learning": False,
# }

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = MarsModel(hyper_params)
# checkpoint = torch.load(
#     f"/home/{user}/models/" + network_name + ".pth", map_location=torch.device("cpu")
# )
# model.load_state_dict(checkpoint)

# model = model.to(device)

# model.eval()
# ctx_test = datasets.ImageFolder(root="data/data/test", transform=data_transform)
# test_loader = torch.utils.data.DataLoader(
#     ctx_test, batch_size=16, shuffle=True, num_workers=4
# )

# crop_img = Image.open("crop_test.png")
# crop_img = crop_img.convert("RGB")
# crop_img = data_transform(crop_img)
# crop_img = crop_img.unsqueeze(0)

# vec_rep = model(crop_img)
# pred = torch.argmax(vec_rep, dim=1).cpu()
# print(f"CROP IMAGE PRED:: {classes[int(pred)]}")

# # test_img1 = Image.open("ctx_crater.png")
# # test_img2 = Image.open("dune_test1.jpg")
# landform = "aec"
# img_choice = random.choice(os.listdir(f"data/data/test/{landform}"))
# test_img2 = Image.open(f"data/data/test/{landform}/{img_choice}")
# test_img2 = test_img2.convert("RGB")

# map_img_test = Image.open("ctx_crater.png")
# # x_h = np.floor(map_img_test.size[0] / 2)
# # y_h = np.floor(map_img_test.size[1] / 2)
# # map_test = map_img_test.crop((x_h, y_h, x_h, y_h))
# # map_test.show()
# window_size = 50
# spacing = 10
# num_of_pages = 5
# pred_window = np.zeros((map_img_test.size[0], map_img_test.size[1], num_of_pages))

# for x_i in np.arange(0, map_img_test.size[0] - window_size, spacing):
#     for y_i in np.arange(0, map_img_test.size[1] - window_size, spacing):
#         window = map_img_test.crop((x_i, y_i, x_i + window_size, y_i + window_size))
#         print(f"Checking {x_i}, {y_i}..")
#         test_img1 = data_transform(window)
#         test_img1 = test_img1.unsqueeze(0)
#         vec_rep = model(test_img1)
#         pred = torch.argmax(vec_rep, dim=1).cpu()
#         print(classes[int(pred)])
#         # window.show()
#         # plt.imshow(window)
#         # plt.show()
#         for page in range(num_of_pages):
#             if (
#                 int(
#                     np.sum(
#                         np.sum(
#                             np.sum(
#                                 pred_window[
#                                     x_i : x_i + window_size,
#                                     y_i : y_i + window_size,
#                                     page,
#                                 ]
#                             )
#                         )
#                     )
#                 )
#                 == 0
#             ):
#                 pred_window[x_i : x_i + window_size, y_i : y_i + window_size] = (
#                     int(pred) + 1
#                 ) * np.ones(
#                     np.shape(
#                         pred_window[x_i : x_i + window_size, y_i : y_i + window_size]
#                     )
#                 )

# print(np.sum(np.sum(pred_window)))

# final_preds = np.zeros((pred_window.shape[0], pred_window.shape[1]))
# for x in range(pred_window.shape[0]):
#     for y in range(pred_window.shape[1]):
#         count = 0
#         for page in range(num_of_pages):
#             if pred_window[x, y, page] > 0:
#                 count += 1

#         final_preds[x, y] = np.round(sum(pred_window[x, y, :]) / count)

# plt.imshow(final_preds)
# plt.show()
# # print(f"Input class: {landform}")
# # print(f"Prediction: {classes[int(pred1)]}")

# crop_img = Image.open("crop_test.png")
