#!/usr/bin/env python3

import weaviate
import base64
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import random
import re
import skimage.io
from process_image2 import process_image

USER = "dusc"
client = weaviate.Client("http://localhost:8080")


def create_db(client, imgs, class_name):
    for img in imgs:
        rois, descs = process_image(img)
        for i in range(len(rois)):
            data = {"SourceName": img, "Coordinates": rois[i]}
            client.data_object.create(data, class_name, vector=descs[i])
    return client


def query_img(client, img_desc, class_name, num_to_retrieve=10):
    result = (
        client.query.get(class_name, ["SourceName", "Coordinates"])
        .with_near_vector(img_desc)
        .with_additional(["distance"])
        .with_limit(num_to_retrieve)
        .do()
    )
    res = [
        [i["SourceName"], i["Coordinates"]] for i in result["data"]["Get"][class_name]
    ]
    return res


schema = {
    "classes": [
        {
            "class": "DoMars16k",
            "vectorizer": "none",
            "properties": [
                {"name": "SourceName", "dataType": ["text"]},
            ],
        }
    ]
}

if __name__ == "__main__":
    client = weaviate.Client("http://localhost:8080")
    print(client.query.aggregate("RegionOfInterest").with_fields("meta {count}").do())
    print(client.query.aggregate("DoMars16k").with_fields("meta {count}").do())
    print(client.query.aggregate("DoMars").with_fields("meta {count}").do())
    #client.schema.delete_class("RegionOfInterest")
    #client.schema.create(schema)
    print(client.schema.get())
    data = {"SourceName": "Test1", "Coordinates": "1111"}
    vec = {"vector": [1, 1, 1, 1, 2]}
    vec2 = [1, 2, 3, 4, 5]
    vec3 = [1, 3, 3, 4, 5]
    # client.data_object.create(data, "RegionOfInterest",vector=vec2)


#    result = (
#        client.query
#        .get("RegionOfInterest", ["sourceName"])
#        .with_near_vector(vec)
#        .with_additional(["distance"])
#        .with_limit(5)
#        .do()
#    )
#    print(result)
#    res = [i["_additional"]["distance"] for i in result["data"]["Get"]["RegionOfInterest"]]
#    print(res)
