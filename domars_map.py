# Create a map for a larger images in which different regions are colored by the predicted landform
#
# You need to download the train models here:
# Only requirement that troubled me was osgeo/gdal, got it to work with external packages on linux
# Make sure to change to correct user and path for the model (line 397)
#
# Choose a cutout of the chosen image in the cutout dict, the larger the cutout the longer the runtime
# Use can also add another image (.tiff format) by changing the CTX_stripe variable
#
#
#
#

from os import times_result
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import io
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import pickle
from tqdm import tqdm
from osgeo import gdal

# import gdal
from pathlib import Path
import numpy as np
from tqdm import tqdm
from numba import jit
import pytorch_lightning as pl
from torchvision.models import densenet161
from skimage.transform import resize
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

USER = "pg2022"

torch.multiprocessing.set_sharing_strategy("file_system")


@jit(nopython=True)
def mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size):
    for r in range(mrf.shape[0]):
        for c in range(mrf.shape[1]):
            pixel_propabilities = mrf_old[
                r,
                c,
            ]
            neighbor_cnt = 0

            m = np.zeros(15)

            n_row_start = max(0, r - neighborhood_size)
            n_row_end = min(mrf.shape[0], r + neighborhood_size + 1)

            n_col_start = max(0, c - neighborhood_size)
            n_col_end = min(mrf.shape[1], c + neighborhood_size + 1)

            for n_row in range(n_row_start, n_row_end):
                for n_col in range(n_col_start, n_col_end):
                    if n_row != r or n_col != c:  # skip self
                        m[mrf_old[n_row, n_col, :].argmax()] += 1
                        neighbor_cnt += 1

            gibs = np.exp(-mrf_gamma * (neighbor_cnt - m))
            mrf_probabilities = gibs * pixel_propabilities
            mrf_probabilities /= np.sum(mrf_probabilities)
            mrf[
                r,
                c,
            ] = mrf_probabilities

    return mrf


def MRF(original, mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11):
    mrf_old = np.array(original)
    mrf = np.zeros(np.shape(original))

    for i in tqdm(range(mrf_iterations)):
        mrf = mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size)
        mrf_old = mrf

    return mrf


def read_geotiff(path):
    # Directly read the tiff data skimage and gdal. Somehow dtype=uint8.
    # Import as_gray=False to avoid float64 conversion.
    img = io.imread(path, as_gray=False, plugin="gdal")
    cs = gdal.Open(path)

    return img, cs


class HIRISE_Image(Dataset):
    def __init__(self, path, window_size=200, transform=None, cutout=None, step_size=1):
        self.transform = transform
        self.path = path
        # self.image_full, self.cs = read_geotiff(path)
        # self.image_full = transform(Image.open(path))
        png_img = Image.open(path)
        rgb_img = png_img.convert("RGB")
        self.image_full = transform(rgb_img)
        self.image_full = self.image_full[0, :, :]
        self.cs = 0
        self.window_size = window_size

        # Crop image according to values in crop
        if cutout is not None:
            self.cutout(cutout)

        print(self.image_full.shape)

        # Get shapes of "new" full image
        self.image_size_full = np.shape(self.image_full)

        self.num_tiles_full = np.ceil(
            np.array(self.image_size_full) / self.window_size
        ).astype("int")
        print(f"{self.num_tiles_full = }")

        wd = self.image_size_full[0]
        hd = self.image_size_full[1]
        # create new image of desired size and color (blue) for padding
        print(window_size)
        print(self.num_tiles_full)
        ww, hh = window_size * self.num_tiles_full
        # hh = window_size * self.num_tiles_full[1]

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - hd) // 2

        # copy img image into center of result image
        self.padded_full = np.zeros(
            tuple((self.num_tiles_full * self.window_size).astype("int")),
            dtype=np.uint8,
        )
        self.padded_full[xx : xx + wd, yy : yy + hd] = self.image_full

        # self.padded_full[:self.image_size_full[0], :self.image_size_full[1]] = self.image_full

        step_size_full = step_size
        idx_tiles_full_a = np.rint(
            np.arange(0, self.num_tiles_full[0] * self.window_size, step_size_full)
        ).astype("int")
        idx_tiles_full_b = np.rint(
            np.arange(0, self.num_tiles_full[1] * self.window_size, step_size_full)
        ).astype("int")

        self.idx_tiles_full_a = idx_tiles_full_a[
            idx_tiles_full_a + self.window_size
            < self.num_tiles_full[0] * self.window_size
        ]
        self.idx_tiles_full_b = idx_tiles_full_b[
            idx_tiles_full_b + self.window_size
            < self.num_tiles_full[1] * self.window_size
        ]

        self.num_full = np.array(
            [self.idx_tiles_full_a.__len__(), self.idx_tiles_full_b.__len__()]
        )
        self.out_shape = (
            self.idx_tiles_full_a.__len__(),
            self.idx_tiles_full_b.__len__(),
        )

    def __len__(self):
        return np.prod(self.num_full)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print(idx)

        idx_in_res = idx
        idx_aa, idx_bb = np.unravel_index(idx_in_res, self.num_full)
        idx_a = self.idx_tiles_full_a[idx_aa]
        idx_b = self.idx_tiles_full_b[idx_bb]
        image = self.padded_full[
            idx_a : idx_a + self.window_size, idx_b : idx_b + self.window_size
        ]
        center_pixel = image[self.window_size // 2, self.window_size // 2]
        image = np.dstack([image] * 3)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, center_pixel

    def get_image(self):
        return self.image_full

    def cutout(self, crop):
        self.crop_image(crop)

    def crop_image(self, crop):
        self.image_full = self.image_full[crop[1] : crop[3], crop[0] : crop[2]]


class CTX_Image(Dataset):
    """CTX dataset."""

    def __init__(self, path, window_size=200, transform=None, cutout=None, step_size=1):
        self.transform = transform
        self.path = path
        self.image_full, self.cs = read_geotiff(path)
        # self.image_full = transform(Image.open(path))
        # self.image_full = self.image_full[0, :, :]
        self.cs = 0
        self.window_size = window_size

        # Crop image according to values in crop
        if cutout is not None:
            self.cutout(cutout)

        print(self.image_full.shape)

        # Get shapes of "new" full image
        self.image_size_full = np.shape(self.image_full)

        self.num_tiles_full = np.ceil(
            np.array(self.image_size_full) / self.window_size
        ).astype("int")
        print(f"{self.num_tiles_full = }")

        wd = self.image_size_full[0]
        hd = self.image_size_full[1]
        # create new image of desired size and color (blue) for padding
        print(window_size)
        print(self.num_tiles_full)
        ww, hh = window_size * self.num_tiles_full
        # hh = window_size * self.num_tiles_full[1]

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - hd) // 2

        # copy img image into center of result image
        self.padded_full = np.zeros(
            tuple((self.num_tiles_full * self.window_size).astype("int")),
            dtype=np.uint8,
        )
        self.padded_full[xx : xx + wd, yy : yy + hd] = self.image_full

        # self.padded_full[:self.image_size_full[0], :self.image_size_full[1]] = self.image_full

        step_size_full = step_size
        idx_tiles_full_a = np.rint(
            np.arange(0, self.num_tiles_full[0] * self.window_size, step_size_full)
        ).astype("int")
        idx_tiles_full_b = np.rint(
            np.arange(0, self.num_tiles_full[1] * self.window_size, step_size_full)
        ).astype("int")

        self.idx_tiles_full_a = idx_tiles_full_a[
            idx_tiles_full_a + self.window_size
            < self.num_tiles_full[0] * self.window_size
        ]
        self.idx_tiles_full_b = idx_tiles_full_b[
            idx_tiles_full_b + self.window_size
            < self.num_tiles_full[1] * self.window_size
        ]

        self.num_full = np.array(
            [self.idx_tiles_full_a.__len__(), self.idx_tiles_full_b.__len__()]
        )
        self.out_shape = (
            self.idx_tiles_full_a.__len__(),
            self.idx_tiles_full_b.__len__(),
        )

    def __len__(self):
        return np.prod(self.num_full)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print(idx)

        idx_in_res = idx
        idx_aa, idx_bb = np.unravel_index(idx_in_res, self.num_full)
        idx_a = self.idx_tiles_full_a[idx_aa]
        idx_b = self.idx_tiles_full_b[idx_bb]
        image = self.padded_full[
            idx_a : idx_a + self.window_size, idx_b : idx_b + self.window_size
        ]
        center_pixel = image[self.window_size // 2, self.window_size // 2]
        image = np.dstack([image] * 3)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, center_pixel

    def get_image(self):
        return self.image_full

    def cutout(self, crop):
        self.crop_image(crop)

    def crop_image(self, crop):
        self.image_full = self.image_full[crop[1] : crop[3], crop[0] : crop[2]]


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

        # return out.reshape(-1)
        return out[0, :, -1, -1].reshape(2208)

    def fc_layer_output2(self, x):
        return_nodes = {"features.denseblock4.denselayer24.conv2": "layer4"}
        self.fe = create_feature_extractor(self.net, return_nodes=return_nodes)

        with torch.no_grad():
            descriptor = self.fe(x)
            descriptor = descriptor["layer4"].cpu().detach().numpy()
            descriptor = descriptor.reshape(-1)
            print(f"SUM DESC: { sum(descriptor) }")
            return descriptor

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(torch.argmax(y_hat, dim=1), y, num_classes=self.num_classes)
        prec, recall = precision_recall(
            F.softmax(y_hat, dim=1), y, num_classes=self.num_classes, reduction="none"
        )
        return {
            "val_loss": loss,
            "val_acc": acc,
            "val_prec": prec,
            "val_recall": recall,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {
            "val_loss": avg_loss,
            "progress_bar": {"val_loss": avg_loss, "val_acc": avg_acc},
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        params_to_update = []
        print("Params to learn:")
        for name, param in self.net.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params_to_update, lr=self.lr, momentum=self.momentum
            )
        else:
            print("Invalid optimizer, exiting...")
            exit()

        return optimizer

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False


def segment_image(
    test_loader,
    model,
    device,
    hyper_params,
    step_size=1,
    img_size=200,
    img_name="CTX_stripe",
):
    predictions = []
    scores = []
    image_pred = []
    heatmap = []

    with tqdm(test_loader, desc="Segmenting", leave=False) as t:
        #    print(t)
        with torch.no_grad():
            for batch in t:
                x, center_pixels = batch
                y_hat = model(x.to(device))

                pred = torch.argmax(y_hat, dim=1).cpu()

                image_pred.append(center_pixels.numpy())
                predictions.append(pred.numpy())
                scores.append(F.softmax(y_hat, dim=1).detach().cpu().numpy())
                heatmap.append(torch.max(y_hat).cpu().detach().numpy())

    # TODO: interpolate data onto image of original size.
    predictions = np.concatenate(predictions, axis=0)
    scores = np.concatenate(scores, axis=0)
    scores = np.reshape(
        np.array(scores),
        (np.int(img_size / step_size), np.int(img_size / step_size))
        + (int(hyper_params["num_classes"]),),
    )
    image_pred = np.reshape(
        np.concatenate(image_pred, axis=0),
        (np.int(img_size / step_size), np.int(img_size / step_size)),
    )

    heatmap = np.reshape(
        np.concatenate(heatmap, axis=0),
        (np.int(img_size / step_size), np.int(img_size / step_size)),
    )
    image_pred = resize(image_pred, (img_size, img_size))
    heatmap = resize(heatmap, (img_size, img_size))
    heatmap *= 255.0 / heatmap.max().astype("uint8")
    jet_cm = plt.cm.get_cmap("jet")
    # TODO: make this dependable on step_size
    scores = resize(scores, (img_size, img_size) + (int(hyper_params["num_classes"]),))

    # Markov random field smoothing
    mrf_probabilities = MRF(scores.astype(np.float64))
    mrf_classes = np.argmax(mrf_probabilities, axis=2)

    # Create Colormap
    n = int(hyper_params["num_classes"])
    from_list = mpl.colors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.tab20(range(0, n)), n)

    # Saving Images
    # TODO: make this dependend on step size
    home_dir = str(Path.home())
    predictions = np.reshape(
        np.array(predictions),
        (np.int(img_size / step_size), np.int(img_size / step_size)),
    )

    plt.imsave(
        home_dir
        + "/segmentation/segmented/"
        + img_name
        + "_"
        + network_name
        + str(step_size)
        + "_map.png",
        resize(predictions, (img_size, img_size)),
        cmap=cm,
        vmin=0,
        vmax=int(hyper_params["num_classes"]),
    )
    plt.imsave(
        home_dir
        + "/segmentation/segmented/"
        + img_name
        + "_"
        + network_name
        + str(step_size)
        + "_img.png",
        np.dstack(
            [np.reshape(np.concatenate(image_pred, axis=0), (img_size, img_size))] * 3
        ),
    )
    plt.imsave(
        home_dir
        + "/segmentation/segmented/"
        + img_name
        + "_"
        + network_name
        + str(step_size)
        + "_mrf.png",
        mrf_classes,
        cmap=cm,
        vmin=0,
        vmax=int(hyper_params["num_classes"]),
    )
    plt.imsave(
        home_dir
        + "/segmentation/segmented/"
        + img_name
        + "_"
        + network_name
        + str(step_size)
        + "_heatmap.png",
        heatmap,
        cmap=jet_cm,
        vmin=0,
        vmax=int(hyper_params["num_classes"]),
    )


torch.backends.cudnn.benchmark = True

# CTX_stripe = "G14_023651_2056_XI_25N148W"
CTX_stripe = "D14_032794_1989_XN_18N282W"
path = CTX_stripe + ".tiff"
network_name = "densenet161"

cutouts = {
    # "D14_032794_1989_XN_18N282W": (1600, 11000, 7000, 14000),  # Jezero
    "D14_032794_1989_XN_18N282W": (600, 11000, 1000, 11400),  # Test
    "F13_040921_1983_XN_18N024W": (3000, 3000, 5800, 7000),  # Oxia Planum
    "G14_023651_2056_XI_25N148W": (1000, 1000, 2000, 2000),  # Lycus links
}

Sulci = {
    "D14_032794_1989_XN_18N282W": "https://image.mars.asu.edu/stream/D14_032794_1989_XN_18N282W.tiff?image=/mars/images/ctx/mrox_1861/prj_full/D14_032794_1989_XN_18N282W.tiff",  # Jezero
    "F13_040921_1983_XN_18N024W": "https://image.mars.asu.edu/stream/F13_040921_1983_XN_18N024W.tiff?image=/mars/images/ctx/mrox_2375/prj_full/F13_040921_1983_XN_18N024W.tiff",  # Oxia Planum
    "G14_023651_2056_XI_25N148W": "https://image.mars.asu.edu/stream/G14_023651_2056_XI_25N148W.tiff?image=/mars/images/ctx/mrox_1385/prj_full/G14_023651_2056_XI_25N148W.tiff",  # Lycus Sulci
}

# if not Path(path).exists():
#     # Download file
#     print("Dowloading...\n")
#     download_file(links[CTX_stripe], "data/raw/")
#     print("...Done")
#     1


# data_transform = transforms.Compose(
#    [
#        transforms.Resize([224, 224]),
#        transforms.ToTensor(),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#    ]
# )
#
# hyper_params = {
#    "batch_size": 64,
#    "num_epochs": 15,
#    "learning_rate": 1e-2,
#    "optimizer": "sgd",
#    "momentum": 0.9,
#    "model": network_name,
#    "num_classes": 15,
#    "pretrained": False,
#    "transfer_learning": False,
# }
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# model = MarsModel(hyper_params)
# print(f"/home/{USER}/models")
# checkpoint = torch.load(
#    f"/home/{USER}/models/" + network_name + ".pth", map_location=torch.device("cpu")
# )
# model.load_state_dict(checkpoint)
#
## path = f"/home/{USER}/codebase-v1/POIs/extracted_pois_ctx_crater.jpeg.png"
#
# ctx_image = CTX_Image(path=path, cutout=cutouts[CTX_stripe], transform=data_transform)
#
# ctx_image2 = CTX_Image(
#    path=path, cutout=cutouts[CTX_stripe], transform=data_transform, step_size=2
# )
#
# test_loader = DataLoader(
#    ctx_image, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
# )  # batch_size = 256, num_workers = 8
#
# test_loader2 = DataLoader(
#    ctx_image2, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
# )
#
## Put on GPU if available
# model = model.to(device)
#
## Set model to eval mode (turns off dropout and moving averages of batchnorm)
# model.eval()
#
# segment_image(test_loader, model, device, hyper_params, img_name=CTX_stripe)
#
# segment_image(
#    test_loader2, model, device, hyper_params, step_size=2, img_name=CTX_stripe
# )
