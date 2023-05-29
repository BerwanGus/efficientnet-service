from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ....efficientnet_model import EfficientNet

import torch

app = FastAPI()

@app.get("/check_service")
def check_service():
    return {"response": "I am alive!"}

@app.get("/get_model")
def get_model(version: str = 'b0'):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_classes = 10

    model = EfficientNet(version, num_classes).to(device)
    return JSONResponse({'model': model}, 200)
    
@app.get("/get_model/{model_path}")
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = torch.load(model, map_location=torch.device(device))
    return JSONResponse({'model': model}, 200)