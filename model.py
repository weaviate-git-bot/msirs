#!/usr/bin/env python3
import numpy as np
import skimage
import tensorflow as tf
import scipy, json

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
