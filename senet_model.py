import numpy as np
import skimage
import tensorflow as tf
import scipy
import json

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

# network only accepts images of size 224x224x3
IMAGE_SIZE = 224


class SegmentChunks(tf.keras.utils.Sequence):
    def __init__(self, img: np.array, window_size: int = 224, step_size: int = 2):
        self.image_full = img
        self.cs = 0
        self.window_size = window_size

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

    # FIXME: TEST THIS!!!!!!!!!!!
    # TODO: need to resize image_batches to form (N,224,244,3)!
    def __getitem__(self, idx):
        images = []
        centers = []

        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        for my_idx in range(low, high):
            idx_aa, idx_bb = np.unravel_index(my_idx, self.num_full)
            idx_a = self.idx_tiles_full_a[idx_aa]
            idx_b = self.idx_tiles_full_b[idx_bb]
            image = self.padded_full[
                idx_a : idx_a + self.window_size, idx_b : idx_b + self.window_size
            ]
            centers.append(image[self.window_size // 2, self.window_size // 2])
            images.append(np.dstack([image] * 3))

        return np.array(images), np.array(centers)


class SENet:
    def __init__(self, model_path: str) -> None:
        self.model = tf.keras.models.load_model(model_path)

    @staticmethod
    def prep_image(img: np.array) -> np.array:
        """
        Prepares a greyscale image (1 channel) for use with the network
        """
        image = skimage.color.gray2rgb(img)
        image = skimage.transform.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True
        )
        image = np.resize(image, (1, 224, 224, 3))
        return image

    def predict(self, img: np.array) -> str:
        """
        Returns string of predicted class for given input image
        """
        image = self.prep_image(img)
        predictions = self.model.predict(image)
        prediction = predictions[0]
        return CATEGORIES[np.argmax(prediction)]

    def get_descriptor(self, img: np.array) -> np.array:
        """
        Returns 512 dimensional descriptor of image, by using output of second to last model layer
        """
        extractor = tf.keras.Model(
            inputs=self.model.inputs, outputs=self.model.layers[610].output
        )
        image = self.prep_image(img)
        feature = extractor(image)
        return feature.numpy().reshape((-1,))

    # FIXME: Test this
    def segment_image(
        self,
        img: np.array,
        window_size: int = 200,
        step_size: int = 4,
        batch_size: int = 64,
        workers: int = 4,
    ) -> None:
        segmenter_sequence = SegmentChunks(
            img=img, window_size=window_size, step_size=step_size
        )
        self.predictions = self.model.predict(
            segmenter_sequence,
            batch_size=batch_size,
            workers=workers,
            use_multiprocessing=True,
        )

    async def vectorize(self, img: str):
        """
        Vectorize function that is used by weaviate for use as vectorizer
        """
        img = json.loads(img)
        return self.get_descriptor(np.array(img)).tolist()


if __name__ == "__main__":
    # small example showcasing class

    model = SENet("fullAdaptedSENetNetmodel.keras")

    img_paths = [
        "/Users/dusc/codebase-v1/data/data/test/ael/B08_012727_1742_XN_05S348W_CX1593_CY12594.jpg",
        "/Users/dusc/codebase-v1/data/data/test/cra/B07_012260_1447_XI_35S194W_CX4750_CY4036.jpg",
        "/Users/dusc/codebase-v1/data/data/test/ael/P06_003352_1763_XN_03S345W_CX440_CY3513.jpg",
        "/Users/dusc/codebase-v1/data/data/test/cra/K01_053719_1938_XI_13N232W_CX1714_CY6640.jpg",
    ]

    features = []

    for i in range(len(img_paths)):
        features.append(model.get_descriptor(skimage.io.imread(img_paths[i])))
        print(model.predict(skimage.io.imread(img_paths[i])))

    dis01 = scipy.spatial.distance.cosine(features[0], features[1])
    dis02 = scipy.spatial.distance.cosine(features[0], features[2])
    dis03 = scipy.spatial.distance.cosine(features[0], features[3])
    dis13 = scipy.spatial.distance.cosine(features[1], features[3])
    dis12 = scipy.spatial.distance.cosine(features[1], features[2])
    dis10 = scipy.spatial.distance.cosine(features[1], features[0])

    print(f"{dis01 = }")
    print(f"{dis02 = }")
    print(f"{dis03 = }")
    print(f"{dis10 = }")
    print(f"{dis12 = }")
    print(f"{dis13 = }")
