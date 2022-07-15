import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from network import Ranking
from data import RecoData, collate_fn


def train(train_df, eval_df, epochs, batch_size=64):
    print("start processing train data")
    train_data = RecoData(train_df)
    print("start processing eval data")
    eval_data = RecoData(eval_df)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collate_fn)

    model = Ranking()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("train start")

    train_loss_list = []
    eval_loss_list = []
    auc_list = []
    for epoch in range(epochs):
        train_loss = 0.
        for train_steps, inputs in enumerate(train_dataloader):
            x, y = inputs
            optimizer.zero_grad()
            pred = model.forward(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_steps % 500 == 499:
                eval_loss = 0.
                pred_list = []
                y_list = []
                with torch.no_grad():
                    for eval_step, inputs in enumerate(eval_dataloader):
                        x, y = inputs
                        pred = model.forward(x)
                        loss = loss_fn(pred, y)
                        eval_loss += loss.item()
                        y_list.extend(y)
                        pred_list.extend(pred)
                auc = roc_auc_score(y_list, pred_list)
                print(
                    "[epoch {:2d} batch {:4d}] train_loss: {:.4f}, \t eval_loss: {:.4f}, \t auc: {:4f}".format(
                        epoch, train_steps + 1, train_loss / 500, eval_loss / (eval_step + 1), auc
                    )
                )
                train_loss_list.append(train_loss / 500)
                eval_loss_list.append(eval_loss / (eval_step + 1))
                auc_list.append(auc)
                train_loss = 0.

    torch.save(model.state_dict(), "./model.pkl")
    df = pd.DataFrame()
    df["train_loss"] = train_loss_list
    df["eval_loss"] = eval_loss_list
    df["auc"] = auc_list
    df["step"] = df.index
    return df


def test(test_df):
    model = Ranking()
    model.load_state_dict(torch.load("./model.pkl"))
    test_data = RecoData(test_df)
    test_dataloader = DataLoader(test_data, batch_size=64, collate_fn=collate_fn)
    pred = []
    with torch.no_grad():
        for inputs in test_dataloader:
            x, _ = inputs
            p = model.forward(x)
            pred.extend(p.tolist())
    test_df["pred"] = pred
    return test_df


if __name__ == "__main__":
    df = pd.read_csv("./data/taobao_small.csv")
    train_df = df[df["timestamp"] < 1512144000]
    test_df = df[df["timestamp"] >= 1512144000]
    train(train_df, test_df, epochs=5)