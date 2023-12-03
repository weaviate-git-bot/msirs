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
from process_image import post_process

USER = "dusc"
client = weaviate.Client("http://localhost:8080")

def create_db(client,imgs, class_name):
    for img in imgs:
        info = post_process(img)
        info_l = [info[i] for i in info.keys()]
        data = {"name": img}
        client.data_object.create(data, class_name, vector=info_l)
    return client

def query_img(client, img, class_name):
    info = post_process(img)
    info_l = {"vector": [info[i] for i in info.keys()]}
    result = (
        client.query
        .get(class_name, ["name"])
        .with_near_vector(info_l)
        .with_additional(["distance"])
        .with_limit(5)
        .do()
    )
    res = [i["name"] for i in result["data"]["Get"][class_name]]
    return res

def plot_results(query,res):
    for i in range(len(res)):
        img_raw = skimage.io.imread(res[i])[:,:,:3]
        skimage.io.imsave(f"res{i}.jpg", img_raw)

    q_img = skimage.io.imread(query)[:,:,:3]
    plt.figure()
    plt.subplot(1,6,1)
    plt.imshow(q_img)
    for i in range(len(res)):
        plt.subplot(1,6,i+2)
        r_img = skimage.io.imread(f"res{i}.jpg")
        plt.imshow(r_img)
    plt.show()

schema2 = {
    'classes': [ {
        'class': 'Region',
        'vectorizer': 'none',
        'properties': [
            {
            'name': 'descriptor',
            'dataType': ['number[]']
            }
            ]
        }]
    }
schema = {
    'classes': [ {
        'class': 'SegmentedImg',
        'vectorizer': 'none',
        'properties': [
            {
            'name': 'name',
            'dataType': ["text"]
            },
            {
                'name': 'region_descriptors',
                'dataType': ['text']
            }
            ]
        }]
    }




# client.schema.delete_class("SegmentedImg")
# client.schema.create(schema)

# print(client.schema.get())
# some_objects = client.data_object.get()
# print(some_objects)

# test_path = f"/Users/{USER}/segmentation/segmented/"
# all_imgs = os.listdir(test_path)
# all_imgs = [i for i in all_imgs if re.findall("mrf", i)]
# all_imgs = [test_path+i for i in all_imgs]

# test = all_imgs[0]
# region_descriptor = [[1.0, 2.0],[2.0, 1.0]]
# for d in region_descriptor:
#     data = {"descriptor": d}
#     client.data_object.create(data, "Region", vector=[1])

# vec = {"vector": [1]}
# result = (
#     client.query
#     .get("Region", ["descriptor"])
#     .with_near_vector(vec)
#     # .with_additional(["distance"])
#     .with_limit(5)
#     .do()
# )

# print(result)

# res = [i for i in result["data"]["Get"]["Region"]]
# img_data_obj = {"name": test, "region_descriptors": str([[0,1,2],[1,2,3]])}

# client.data_object.create(img_data_obj, "SegmentedImg", vector=[5])

vec = {"vector": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
result = (
    client.query
    .get("SegmentedImg", ["region_descriptors"])
    .with_near_vector(vec)
    .with_additional(["distance"])
    .with_limit(5)
    .do()
)
res = [i["_additional"]["distance"] for i in result["data"]["Get"]["SegmentedImg"]]
print(res)
# res = [i["region_descriptors"] for i in result["data"]["Get"]["SegmentedImg"]]
# print(eval(res[0]))



# img = test_path+random.choice(os.listdir(test_path))
# where_filter = {
#     "path": ["name"],
#     "operator": "Equal",
#     "valueText": img,
# }

# result = (
#     client.query
#     .aggregate("SegmentedImg")
#     .with_fields("meta {count}")
#     .with_where(where_filter)
#     .do()
# )
# print(result)
# client = create_db(client, all_imgs, "SegmentedImg")
# img = test_path+random.choice(os.listdir(test_path))
# res = query_img(client, img, "SegmentedImg")
# plot_results(img, res)
