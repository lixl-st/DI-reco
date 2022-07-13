import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from network import Ranking
from data import RecoData, collate_fn


USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_CUDA else "cpu"


def main():
    model = Ranking()
    if USE_CUDA:
        model.cuda()
    loss_fn = nn.BCELoss()
    train_data = RecoData(mode="train")
    test_data = RecoData(mode="test")
    train_dataloader = DataLoader(train_data, batch_size=64, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=64, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100):
        train_loss = 0.
        for steps, inputs in enumerate(train_dataloader):
            x, y = inputs
            if USE_CUDA:
                for k in x.keys():
                    x[k].to(DEVICE)
                y.to(DEVICE)
            optimizer.zero_grad()
            pred = model.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 5 == 0:
            test_loss = 0.
            pred_list = []
            y_list = []
            with torch.no_grad():
                for inputs in test_dataloader:
                    x, y = inputs
                    if USE_CUDA:
                        for k, in x.keys():
                            x[k].to(DEVICE)
                        y.to(DEVICE)
                    pred = model.forward(x)
                    loss = loss_fn(pred, y)
                    test_loss += loss.item()
                    if USE_CUDA:
                        y.cpu()
                        pred.cpu()
                    y_list.extend(y)
                    pred_list.extend(pred)
            auc = roc_auc_score(y_list, pred_list)
            print(
                "[epoch {}] train_loss: {:.4f}, \t test_loss: {:.4f}, \t auc: {:4f}".format(
                    epoch, train_loss / (steps + 1), test_loss / (steps + 1), auc
                )
            )


if __name__ == "__main__":
    main()