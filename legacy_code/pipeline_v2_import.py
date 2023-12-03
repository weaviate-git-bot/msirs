#!/usr/bin/env python3

from pipeline_v2 import PipelineV2
from pathlib import Path
from PIL import Image
from torchvision import transforms
import os, re, sys

HOME = str(Path.home())


data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":
    img = sys.argv[1]
    pipe = PipelineV2("http://localhost:8080")
    print("Instantiated pipeline")
    pipe.check_db()
    
    img = data_transform(Image.fromarray(img).convert("RGB"))
    img = img.unsqueeze(0)
    descriptor = pipe.get_descriptor(img)
    data = {"SourceName": file_name}
    pipe.add_to_db(data, descriptor)
