import os
import torch
from toolbox.efficientnet_model import EfficientNet
from toolbox.train_model import DeviceDataLoader


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_enet(version, path):
    num_classes = 10

    model = EfficientNet(version, num_classes).to(device)
    if path:
        model_state = torch.load(path, map_location=torch.device(device))
        model.load_state_dict(model_state)
    
    return model


def decode_classes(preds, classes):
    if isinstance(preds, (tuple, list, torch.Tensor)):
        return [classes[int(x)] for x in preds]
    else:
        return classes[int(preds)]


def predict(inference_dataloader):
    model = load_enet(os.getenv("ENET_VERSION"), os.getenv("MODEL_PATH"))
    model.training = False

    classes = inference_dataloader.classes
    inference_dataloader = DeviceDataLoader(inference_dataloader, device)

    for i, inputs in enumerate(inference_dataloader):
        logits = model(inputs)
        probs = torch.nn.Softmax(1)(logits)
        preds = torch.argmax(probs, dim=1)
    
    decoded_preds = decode_classes(preds, classes)

    return (probs, decoded_preds)
        

if __name__=='__main__':
    if os.getenv("DATA_PATH") is None:
        os.environ["DATA_PATH"] = '../data'
        os.environ["MODEL_PATH"] = "../models/model_20230529_050355_4"
        os.environ["ENET_VERSION"] = 'b0'
    predict()

