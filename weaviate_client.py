#!/usr/bin/env python3
import weaviate
import numpy as np
import json

# TODO: add uuid or smth
SCHEMA = {
    "classes": [
        {
            "class": "Test",
            "vectorizer": "img2vec-neural",
            "vectorIndexType": "hnsw",
            "moduleConfig": {"img2vec-neural": {"imageFields": ["image"]}},
            "properties": [
                {"name": "image", "dataType": ["string"]},
                {"name": "source", "dataType": ["string"]},
                {"name": "meta_data", "dataType": ["string"]},
            ],
        }
    ],
}


class WeaviateClient:
    def __init__(self, db_adr: str, schema: str = "") -> None:
        SCHEMA = {
            "classes": [
                {
                    "class": "Test",
                    "vectorizer": "img2vec-neural",
                    "vectorIndexType": "hnsw",
                    "moduleConfig": {"img2vec-neural": {"imageFields": ["image"]}},
                    "properties": [
                        {"name": "image", "dataType": ["string"]},
                        {"name": "source", "dataType": ["string"]},
                        {"name": "meta_data", "dataType": ["string"]},
                    ],
                }
            ],
        }

        self.client = weaviate.Client(db_adr)

        # add schema, allow for passing of custom schema for better development and testing
        if schema == "":
            self.schema = "Test"
        else:
            self.schema = schema

    def add_to_db(self, img: np.ndarray) -> None:
        # TODO: make sure image is in correct format!!
        data = {"image": str(img.tolist())}
        # TODO: add meta data here somehow
        self.create_entry(data)

    def create_entry(self, data_object: dict) -> None:
        does_exist = self.check_for_duplicate_entries(data_object)
        if not does_exist:
            self.client.data_object.create(data_object, self.schema)
        else:
            print("Object already exists, skipping addition")

    def check_for_duplicate_entries(self, object_to_check: dict) -> bool:
        # TODO: rewrite this using UUIDs
        does_exist = True
        return does_exist

    def query_image(self, img_data: np.ndarray, num_to_retrieve=10) -> dict:
        # TODO: make sure this works
        vector = img_data.tolist()

        result = (
            self.client.query.get(self.schema, ["source", "meta_data"])
            .with_near_vector(
                {
                    "vector": vector,
                }
            )
            .with_additional(["distance"])
            .with_limit(num_to_retrieve)
            .do()
        )

        images = [i["source"] for i in result["data"]["Get"][self.schema]]
        distances = [i["_additional"] for i in result["data"]["Get"][self.schema]]
        meta_data = [
            json.loads(i["meta_data"]) for i in result["data"]["Get"][self.schema]
        ]
        response = {"images": images, "distances": distances, "meta_data": meta_data}
        return response

    def check_db(self) -> None:
        result = (
            self.client.query.aggregate(self.schema).with_fields("meta {count}").do()
        )
        print(result)
