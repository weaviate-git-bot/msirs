#!/usr/bin/env python3
import weaviate
import numpy as np

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
            ],
        }
    ],
}


class WeaviateClient:
    def __init__(self, db_adr: str, schema: dict = SCHEMA) -> None:
        self.client = weaviate.Client(db_adr)
        self.schema = schema

    def add_to_db(self, img: np.array) -> None:
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
