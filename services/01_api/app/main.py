import torch
import os
from torch.utils.data import Dataset
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
        self.images = [filename.file.read() for filename in files]
        self.resolution = phi_values[os.getenv('ENET_VERSION')][1] # get res from phi_values
        self.transform = transforms.Resize(self.resolution)
        self.classes =  ('plane', 'car', 'bird', 'cat', 'deer',
                         'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_bytes = self.images[idx]
        tensor = torch.frombuffer(image_bytes, dtype=torch.int)
        return self.transform(tensor)


@app.get("/healthcheck")
def get_check_service():
    return {"response": "I am alive!"}


@app.get("/predict")
def get_predict(files: List[UploadFile]):
    dataloader = InferenceDataLoader(files)
    probs, decoded_preds, decoded_labels = predict(dataloader)

    results = {}
    for i, (prob, dpred, dlabel) in enumerate(zip(probs, decoded_preds, decoded_labels)):
        id = f"image{i}"
        results[id] = (dpred, float(torch.max(prob)*100), dlabel)
        print("\tpredicted: {} ({:.2f}%); true: {}".format(*results[id]))
    
    return JSONResponse({"predictions": results}, 200)