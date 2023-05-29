from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_model import EfficientNet


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            if isinstance(batch, (list, tuple)):
                yield [data.to(self.device) for data in batch]
            else:
                yield batch.to(self.device)

    def __len__(self):
        return len(self.dl)


def load_dummy_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    training_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    training_loader = DataLoader(training_set, batch_size=4, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=4, shuffle=True)
    
    training_loader = DeviceDataLoader(training_loader, device)
    validation_loader = DeviceDataLoader(validation_loader, device)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("Training set has {} examples".format(len(training_set)))
    print("Valiation set has {} examples".format(len(validation_set)))

    return (training_loader, validation_loader, classes)


def train_step(data_loader, model, optimizer, loss_fn):
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(data_loader):
        inputs, labels = data

        optimizer.zero_grad()

        predictions = model(inputs)

        loss = loss_fn(predictions, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    
    return last_loss


def train_model(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

    training_loader, validation_loader, classes = load_dummy_data()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_vloss = 1_000_000.

    epochs = 5
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch + 1))

        model.train(True)
        avg_loss = train_step(training_loader,
                              model=model,
                              optimizer=optimizer,
                              loss_fn=loss_fn)
        
        running_vloss = 0
        model.train(False)
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vpredictions = model(vinputs)
            vloss = loss_fn(vpredictions, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print('Loss train {:.6f} \
               valid {:.6f}'.format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    version = "b0"
    num_classes = 10
    model = EfficientNet(version, num_classes).to(device)
    train_model(model)

    # testing
    # _, vdataloader, _ = load_dummy_data()
    # x, y = list(vdataloader)[0]
    # preds = model(x)
    # torch.nn.Softmax(dim=1)(preds).argmax(axis=1), y



