import numpy as np
import skimage
import tensorflow as tf
import scipy

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

    # FIXME: Check if this works yo!!
    def create_data_batch(
        self, img: np.array, window_size: int, step_size: int
    ) -> np.array:
        image_full = img
        cs = 0
        window_size = window_size

        print(image_full.shape)

        # Get shapes of "new" full image
        image_size_full = np.shape(image_full)

        num_tiles_full = np.ceil(np.array(image_size_full) / window_size).astype("int")
        print(f"{ num_tiles_full = }")

        wd = image_size_full[0]
        hd = image_size_full[1]
        # create new image of desired size and color (blue) for padding
        print(window_size)
        print(num_tiles_full)
        ww, hh = window_size * num_tiles_full
        # hh = window_size *  num_tiles_full[1]

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - hd) // 2

        # copy img image into center of result image
        padded_full = np.zeros(
            tuple((num_tiles_full * window_size).astype("int")),
            dtype=np.uint8,
        )
        padded_full[xx : xx + wd, yy : yy + hd] = image_full

        #  padded_full[: image_size_full[0], : image_size_full[1]] =  image_full

        step_size_full = step_size
        idx_tiles_full_a = np.rint(
            np.arange(0, num_tiles_full[0] * window_size, step_size_full)
        ).astype("int")
        idx_tiles_full_b = np.rint(
            np.arange(0, num_tiles_full[1] * window_size, step_size_full)
        ).astype("int")

        idx_tiles_full_a = idx_tiles_full_a[
            idx_tiles_full_a + window_size < num_tiles_full[0] * window_size
        ]
        idx_tiles_full_b = idx_tiles_full_b[
            idx_tiles_full_b + window_size < num_tiles_full[1] * window_size
        ]

        num_full = np.array([idx_tiles_full_a.__len__(), idx_tiles_full_b.__len__()])
        out_shape = (
            idx_tiles_full_a.__len__(),
            idx_tiles_full_b.__len__(),
        )

        # TODO: define indices
        indices = []
        images = []
        for idx in indices:
            idx_aa, idx_bb = np.unravel_index(idx, num_full)
            idx_a = idx_tiles_full_a[idx_aa]
            idx_b = idx_tiles_full_b[idx_bb]
            image = padded_full[
                idx_a : idx_a + window_size, idx_b : idx_b + window_size
            ]
            # TODO: do i need this??
            # center_pixel = image[window_size // 2, window_size // 2]
            images.append(np.dstack([image] * 3))

        return np.array(images)

    async def vectorize(self, img: np.array):
        """
        Vectorize function that is used by weaviate for use as vectorizer
        """
        return self.get_descriptor(img)


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
