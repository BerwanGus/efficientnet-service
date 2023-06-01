import os
import torch
from efficientnet_model import EfficientNet
from train_model import load_dummy_data


def load_unet(version, path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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


def predict():
    # model = load_unet(os.getenv('UNET_VERSION'), os.getenv('MODEL_PATH'))
    model = load_unet(os.getenv("UNET_VERSION"), os.getenv("MODEL_PATH"))
    model.training = False
    
    training_loader, validation_loader, classes = load_dummy_data()

    rand_index = torch.randint(low=0, high=len(validation_loader), size=(1,))
    batch = validation_loader[rand_index]
    inputs, labels = batch
    logits = model(inputs)
    probs = torch.nn.Softmax(1)(logits)
    preds = torch.argmax(probs, dim=1)
    
    decoded_preds = decode_classes(preds, classes)
    decoded_labels = decode_classes(labels, classes)

    print("batch: {}".format(int(rand_index)))
    for prob, dpred, dlabel in zip(probs, decoded_preds, decoded_labels):
        print("\tpredicted: {} ({:.2f}%); true: {}".format(dpred,
                                                           torch.max(prob) * 100,
                                                           dlabel))
        

if __name__=='__main__':
    if os.getenv("DATA_PATH") is None:
        os.environ["DATA_PATH"] = '../data'
        os.environ["MODEL_PATH"] = "../models/model_20230529_050355_4"
        os.environ["UNET_VERSION"] = 'b0'
    predict()

