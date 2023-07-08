import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from toolbox.efficientnet_model import phi_values
from toolbox.prediction import predict
from toolbox.train_model import load_dummy_data
from typing import List

app = FastAPI()


class InferenceDataLoader(Dataset):
    def __init__(self, files: List[UploadFile]):
        self.files = files
        self.resolution = phi_values[os.getenv('ENET_VERSION')][1] # get res from phi_values
        self.transforms = transforms.Compose([transforms.Resize((self.resolution, self.resolution)),
                                              transforms.ToTensor()]) 
        self.classes =  ('plane', 'car', 'bird', 'cat', 'deer',
                         'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx].file)
        return self.transforms(img)[None, :, :, :] # convert single to batch


@app.get("/healthcheck")
def get_check_service():
    return {"response": "I am alive!"}


@app.get("/predict")
def get_predict(files: List[UploadFile]):
    dataloader = InferenceDataLoader(files)
    probs, decoded_preds = predict(dataloader)
    results = {}
    for i, (prob, dpred) in enumerate(zip(probs, decoded_preds)):
        id = f"image{i}"
        results[id] = (dpred, float(torch.max(prob)*100))
        print("\tpredicted: {} ({:.2f}%)".format(*results[id]))
    
    return JSONResponse({"predictions": results}, 200)