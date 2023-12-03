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
from PIL import Image
from scipy import spatial
from pathlib import Path

home_dir = str(Path.home())

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


class Pipeline:
    def __init__(self, db_adr: str):
        self.client = weaviate.Client(db_adr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        model = MarsModel(hyper_params)
        print(home_dir + "/models")
        checkpoint = torch.load(
            home_dir + "/models/" + hyper_params["model"] + ".pth",
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        self.model = model
        self.model.eval()

    def add_to_db(self, data_object: dict, vec: list):
        self.client.data_object.create(data_object, "SegmentedImg", vector=vec)

    def preliminary_search(self, vec: list, limit=10, with_distance=False):
        if not with_distance:
            vec_dict = {"vector": vec}
            result = (
                self.client.query.get("SegmentedImg", ["name", "region_descriptors"])
                .with_near_vector(vec_dict)
                .with_limit(limit)
                .do()
            )
        else:
            vec_dict = {"vector": vec}
            result = (
                self.client.query.get("SegmentedImg", ["name", "region_descriptors"])
                .with_near_vector(vec_dict)
                .with_additional(["distance"])
                .with_limit(limit)
                .do()
            )

        # res = [i["name"] for i in result["data"]["Get"][]]
        return result

    def query_image(self, file: str):
        # query an image, this is the main part of the pipeline
        # TODO: need to catch file extensions and name and such lol

        print(f"{file = }")
        # if the string "mrf" is not part of the image name, the image segmentation
        # this creates an segmented image and saves it onto the disk
        if not re.findall("mrf", file):
            self.preprocess_image(file)
            print("Created segmented image")
        # process image, the file name is changed to the segmented image because this is what the pipeline works with
        # TODO: make sure to change name here appropriately
        file = re.sub("og_img","mrf",file)
        print(f"{file =}")
        # region_info is not currently used, but may prove useful for debugging
        # info_dict stores info about the percentage of area covered by the different classes
        # q_cutouts is a list that contains cutout parts of the image that (hopefully) contain interesting landmarks
        info_dict, region_info, q_cutouts = self.process_image(file)
        # info_l is an array of length 15, in which every value correspondes to the percentage of area covered by the corresponding class
        info_l = [info_dict[i] for i in info_dict.keys()]

        # if the length of the cutouts list is less than one then no interesting landmarks were found.
        if len(q_cutouts) > 1:
            # the descriptor cnn is used to describe the cutouts provided by the segmentation
            q_descriptors = self.region_cnn_eval(q_cutouts)

            print("Processed image. Starting search...")
            # preliminary search
            results = self.preliminary_search(info_l)
            # only the values returned in the preliminary search are used
            names = [i["name"] for i in results["data"]["Get"]["SegmentedImg"]]
            descriptors = [eval(i["region_descriptors"]) for i in results["data"]["Get"]["SegmentedImg"]]

            # second stage of the the search
            mean_results, single_results = self.region_comparison(q_descriptors, names, descriptors)

            # check whether the image already exists in the database, if it doesnt exist, it is added to the database
            # !!no need to understand how this works!!
            vec = {"vector": info_l}
            db_check_results = self.client.query.get("SegmentedImg", ["region_descriptors"]).with_near_vector(vec).with_additional(["distance"]).with_limit(1).do()
            res = [i["_additional"]["distance"] for i in db_check_results["data"]["Get"]["SegmentedImg"]]
            if res[0] != 0:
                print("Adding to database...")
                data_object = {"name": file, "region_descriptors": str(q_descriptors)}
                # small helper function that adds the image to the database
                self.add_to_db(data_object, vec=info_l)

            return mean_results, single_results
        else:
            # no regions of interest. only do preliminary search.
            print("No interesting regions found, only doing preliminary search.")
            results = self.preliminary_search(info_l, with_distance=True)
            files = [i["name"] for i in results["data"]["Get"]["SegmentedImg"]]
            # methods expect 2 metrics to be returned, so im dividing the results to fake that
            files1 = files[:int(np.round(len(files)/2))]
            files2 = files[int(np.round(len(files)/2)):]
            # distance describes the distance from the vector used for the query, the smaller the distance the more similar the result.
            # the query return is in descending order, meaning the first x results are the best x results
            distances = [i["_additional"]["distance"] for i in results["data"]["Get"]["SegmentedImg"]]
            distances = [[i]*len(distances) for i in distances]
            distances1 = distances[:int(np.round(len(distances)/2))]
            distances2 = distances[int(np.round(len(distances)/2)):]

            # there are no descriptors but the format needs to match for other methods
            mean_results = [files1, distances1, []]
            single_results = [files2, distances2, []]

            return mean_results, single_results

    def build_db(self, files: list):
        # populates the database with a given list of image file names
        # FIXME: zero cutouts arent yet working!
        for file in files:
            info_dict, region_info, q_cutouts = self.process_image(file)
            if len(q_cutouts) > 1:
                info_l = [info_dict[i] for i in info_dict.keys()]
                q_descriptors = self.region_cnn_eval(q_cutouts)
                if self.check_for_existing_entries(info_l):
                    data_object = {
                        "name": file,
                        "region_descriptors": str(q_descriptors),
                    }
                    self.add_to_db(data_object, vec=info_l)
            else:
                # no cutouts, add empty descriptors
                info_l = [info_dict[i] for i in info_dict.keys()]
                if self.check_for_existing_entries(info_l):
                    data_object = {"name": file, "region_descriptors": str([])}
                    self.add_to_db(data_object, vec=info_l)

    def check_for_existing_entries(self, vector: list) -> bool:
        vec = {"vector": vector}
        results = (
            self.client.query.get("SegmentedImg", ["region_descriptors"])
            .with_near_vector(vec)
            .with_additional(["distance"])
            .with_limit(1)
            .do()
        )
        res = [
            i["_additional"]["distance"] for i in results["data"]["Get"]["SegmentedImg"]
        ]
        if res[0] != 0:
            print("New entry. Adding to database...")
            return True
        else:
            print("Entry exists already, skipping...")
            return False

    def check_db(self):
        # retrieves the number of images in the database and prints it to the console

        result = (
            self.client.query.aggregate("SegmentedImg").with_fields("meta {count}")
            # .with_where(where_filter)
            .do()
        )
        print(result)

    def region_cnn_eval(self, cutouts: list):
        # creates descriptors for an image given images of its intersting landmarks

        descriptors = []
        for cutout in cutouts:
            # image needs to be preprocessed
            ctx_image = data_transform(Image.fromarray(cutout).convert('RGB'))
            test_img1 = ctx_image.unsqueeze(0)
            descriptor = self.model.fc_layer_output(test_img1.to(self.device))
            descriptor = descriptor.tolist()
            descriptors.append(descriptor)

        return descriptors

    @staticmethod
    def process_image(file: str) -> tuple[dict, list, list]:
        # segments the image

        boxes = detect_regions(file)
        print("Found bounding boxes for query image")
        cutouts, region_info = extract_regions(
            file, boxes, interesting_classes, color_info
        )
        print("Extracted regions...")
        info_dict = post_process(file)
        # returns info about the area covered by different classes as well as images of the interesting regions
        return info_dict, region_info, cutouts

    def region_comparison(self, q_descriptors: list, db_files: list, db_descriptors: list, num_to_return=5):
        # compares the descriptors of query image with provided db_descriptors
        # similarity is currently measured via sum of squared differences -> less is better
        # currently uses 2 different metrics:
        # 1. Single best match: the image that has the single best similarity between any given descriptors
        # 2. Mean best match: the best mean similarity over all query descriptors

        best_per_image = []
        for file in range(len(db_files)):
            best_curr_sim = []
            best_curr_desc = []

            print(len(db_descriptors))
            print(len(q_descriptors))
            for q_desc in q_descriptors:
                c_best_ssd = -10e10
                c_best_desc = []
                for desc in db_descriptors[file]:
                    print(len(desc))
                    ssd = np.sum((np.array(q_desc) - np.array(desc)) ** 2)
                    if ssd > c_best_ssd:
                        c_best_ssd = ssd
                        c_best_desc = desc
                best_curr_sim.append(c_best_ssd)
                best_curr_desc.append(c_best_desc)
            # have best matches within current files descriptors for every q_descriptor
            best_per_image.append([best_curr_sim, best_curr_desc, db_files[file]])

        # sort similarity results in descending order, meaning the lowest sum of squared differences will be the first entry
        mean_results = []
        best_single_results = []
        files = []
        descriptors = []
        print(best_per_image)
        for db_img in range(len(best_per_image)):
            files.append(best_per_image[db_img][2])
            descriptors.extend(
                [
                    x
                    for _, x in sorted(
                        zip(best_per_image[db_img][0], best_per_image[db_img][1])
                    )
                ]
            )
            ordered_best_sim = sorted(best_per_image[db_img][0])
            print(f"{ordered_best_sim = }")

            # sort via mean
            mean_results.append(np.mean(ordered_best_sim))
            # sort via single best result
            best_single_results.append(ordered_best_sim[0])


        # sort files and descriptors by order of the best single and mean result respectively
        best_means_files = [x for _,x in sorted(zip(mean_results, files))]
        best_means_descriptors = [x for _,x in sorted(zip(mean_results, descriptors))]
        best_means = sorted(mean_results)
        best_single_files = [x for _, x in sorted(zip(best_single_results, files))]
        best_single_descriptors = [
            x for _, x in sorted(zip(best_single_results, descriptors))
        ]
        best_single = sorted(best_single_results)

        # lists with 3 entries:
        # 1. entry is a list containing the best matching files in descending order
        # 2. entry is a list containing the similarity scores
        # 3. entry is a list containing the descriptors of the corresponding files
        mean_return = [best_means_files[:num_to_return], best_means[:num_to_return], best_means_descriptors[:num_to_return]]
        single_return = [best_single_files[:num_to_return], best_single[:num_to_return], best_single_descriptors[:num_to_return]]

        return mean_return, single_return


    def visualize_results(self,query,best_mean, best_single):
        # this can be used to visualize results quickly on the local machine
        # !!you dont really need to understand this!!

        query = self.get_img_from_mrf(query)
        q_img = skimage.io.imread(query)
        num_to_show = len(best_mean)
        plt.figure()
        plt.subplot(1, 2 * num_to_show + 1, 1)
        title = query.split("/")[-1][:14]
        plt.title(f"Query: {title}")
        plt.imshow(q_img)

        for i in range(num_to_show):
            plt.subplot(1, 2 * num_to_show + 1, i + 2)
            r_img = skimage.io.imread(self.get_img_from_mrf(best_mean[0][i]))
            plt.imshow(r_img)
            title = best_mean[0][i].split("/")[-1][0:14]
            plt.title(f"{title}")

        for i in range(num_to_show):
            plt.subplot(1, 2 * num_to_show + 1, i + 2)
            r_img = skimage.io.imread(self.get_img_from_mrf(best_single[0][i]))
            plt.imshow(r_img)
            title = best_single[0][i].split("/")[-1][0:14]
            plt.title(f"{title}")
        plt.show()

    def preprocess_image(self, img: str):
        # segment a given image

        # TODO: some if else to make sure a proper name is chosen.
        file_name = img.split("/")[-1][:-24]
        step_size = 2
        plt.imsave(
            home_dir
            + "/segmentation/segmented/"
            + file_name
            + "_"
            + "densenet161"
            + str(step_size)
            + "_og_img.png",
            skimage.io.imread(img),
        )

        ctx_image = HIRISE_Image(
            path=img, transform=data_transform, step_size=step_size
        )

        test_loader = DataLoader(
            ctx_image, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
        )
        print("Segmenting image for real")
        segment_image(
            test_loader,
            self.model,
            self.device,
            hyper_params,
            step_size=step_size,
            img_name=file_name,
        )
        pass

    def store_for_ui(self, folder: str, mean_results: list, single_results: list, query: str):
        # clean up old results created by the webserver!
        # DONT USE THIS WHEN RUNNING THE SCRIPT LOCALLY

        images_path = home_dir+"/segmentation/segmented/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # what the fuck im i doing here huhhhhhhhhhhhhhhhh
        file = [i for i in os.listdir(images_path) if re.findall(img, images_path + i)][
            0
        ]
        if re.findall("mrf", file):
            file = re.sub("mrf", "og_img", file)
        shutil.copy(images_path + file, folder + f"query.png")

        i1 = [i for i in mean_results[0]]
        i2 = [i for i in single_results[0]]
        results = i1 + i2
        print(results)
        # look for file in storage folder
        counter = 1
        for result in results:
            # print("listdir",os.listdir(images_path)[0])
            # print("result",result)
            # print("combi",images_path+os.listdir(images_path)[0])
            file = [
                i
                for i in os.listdir(images_path)
                if re.findall(result, images_path + i)
            ]
            print(f"{result = }")
            print(file)
            file = file[0]
            file = self.get_img_from_mrf(file)
            shutil.copy(images_path + file, folder + f"retrieval_{counter}.png")
            counter += 1

    @staticmethod
    def convert_to_png(img: str):
        try:
            q_img = Image.open(img)
            png_str = img.split(".")[0] + ".png"
            q_img.save(png_str)
        except Exception:
            print("Not a valid image format lmao")

    @staticmethod
    def clear_queue():
        # cleans up the query folder
        # DONT USE THIS WHEN RUNNING THE SCRIPT LOCALLY

        folder = home_dir+"/query/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    @staticmethod
    def get_img_from_mrf(img: str):
        # changes the file name to the original image, given a segmented image
        img = re.sub("mrf", "og_img", img)
        return img

    @staticmethod
    def get_mrf_from_img(img: str):
        # changes the file name to get the original image, given a segmented image
        img = re.sub("og_img", "mrf", img)
        return img


if __name__ == "__main__":
    pipe = Pipeline("http://localhost:8080")
    print("Instantiated pipeline")
    home_dir = str(Path.home())
    test_path = home_dir + f"/segmentation/segmented/"
    all_imgs = os.listdir(test_path)
    all_imgs = [i for i in all_imgs if re.findall("mrf", i)]
    all_imgs = [test_path + i for i in all_imgs][63:]
    test_img = "/Users/dusc/Dropbox/marin.jpg"
    # img = all_imgs
    # all_imgs = [all_imgs]
    # print(all_imgs)
    pipe.check_db()
    # pipe.build_db(all_imgs)
    # img = random.choice(all_imgs)
    # img = re.sub("mrf", "og_img", img)
    query_path = home_dir + "/query/"
    img = query_path + os.listdir(query_path)[0]
    img_name = img.split("/")[-1:][0]
    img = test_path + img_name
    print(img)

    v, f = pipe.query_image(img)
    pipe.clear_queue()
    # pipe.visualize_results(img, v, f)

    pipe.store_for_ui(home_dir + f"/server-test/", v, f, img)
