import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from network import Ranking

data = TensorDataset()

def preprocess_learn(x):
    data = {}
    data["user_id"] = torch.LongTensor(x[:, 0])
    data["target_item_id"] = torch.LongTensor(x[:, 1])
    data["target_group_id"] = torch.LongTensor(x[:, 2])
    return data

    
def train():
    model = Ranking()
    loss_fn = nn.CrossEntropyLoss()
    dataloader = DataLoader(data, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(10):
        train_loss = 0.
        test_loss = 0.
        for steps, inputs in enumerate(dataloader):
            x, y = inputs
            optimizer.zero_grad()
            pred = model.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 5 == 0:
            with torch.no_grad():
                for inputs in dataloader:
                    x, y = inputs
                    outputs = model.forward(x)
                    test_loss += outputs.item()

                print(
                    "epoch {}: test_loss: {:.4f}, \t test_loss: {:.4f}".format(
                        epoch, train_loss / (steps + 1), test_loss / (steps + 1)
                    )
                )