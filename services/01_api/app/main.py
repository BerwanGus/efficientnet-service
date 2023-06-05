import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from toolbox.efficientnet_model import EfficientNet
from toolbox.prediction import predict


app = FastAPI()

@app.get("/healthcheck")
def get_check_service():
    return {"response": "I am alive!"}


@app.get("/predict")
def get_predict():
    rand_index = torch.randint(low=0, high=int(10000/4), size=(1,))
    probs, decoded_preds, decoded_labels = predict(rand_index)

    results = {}
    for i, (prob, dpred, dlabel) in enumerate(zip(probs, decoded_preds, decoded_labels)):
        id = f"batch_{rand_index}_image{i}"
        results[id] = (dpred, float(torch.max(prob)*100), dlabel)
        print("\tpredicted: {} ({:.2f}%); true: {}".format(*results[id]))
    
    return JSONResponse({"predictions": results}, 200)