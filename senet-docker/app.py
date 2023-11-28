"""
This app is instantiated in the docker container and provides the necessary apis for the vectorization that will automatically be used by weaviate
"""

from fastapi import FastAPI, Response, status
from senet_model import SENet
import os
import numpy as np
from pydantic import BaseModel


class VectorInput(BaseModel):
    image: str


app = FastAPI()

model_path = "fullAdaptedSENetNetmodel.keras"
vec = SENet(model_path)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    return {"Wow": "wow"}


@app.post("/vectors")
async def read_item(item: VectorInput, response: Response):
    try:
        vector = await vec.vectorize(item.image)
        return {"text": "success??", "vector": vector}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
