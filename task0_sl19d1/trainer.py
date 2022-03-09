import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from create_dataset import CustomTrainDataset
from NN import NeuralNetwork

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred[...,0], y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred[...,0], y).item()
            correct += (pred.argmax(1) == y).type(torch.double).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def train(annotations_file, model_name):
    full_dataset = CustomTrainDataset(annotations_file=annotations_file)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # fix random seed for reproducability 
    g = torch.Generator()
    g.manual_seed(0)

    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=g)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, worker_init_fn=seed_worker,
    generator=g)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, worker_init_fn=seed_worker,
    generator=g)

    # Display 
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # print(f"Data: {train_features[0]}")
    # print(f"Label: {train_labels[0]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    torch.manual_seed(0)
    model = NeuralNetwork().to(device)
    model.double()

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # logits = model(train_features)
    # print(f"Predicted class: {logits}")

    learning_rate = 1e-4
    batch_size = 64
    epochs = 50

    #TODO continue tutorial here https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    # loss_fn = nn.CrossEntropyLoss() #TODO replace this with the RMS loss
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)

    # print model information
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    torch.save(model, model_name)
    print("Done training!")
    return model

