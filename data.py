from re import L
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def reward_mapping(ss: str, mode):
    if ss == "pv":
        return 0
    elif ss == "buy":
        return 1
    else:
        if mode == "train":
            return 0.01
        else:
            return 0


class RecoData(Dataset):
    def __init__(self, path="/Users/lixuelin/code/demo/poc/data.csv", max_len=100000, mode="train") -> None:
        super().__init__()
        df = pd.read_csv(path, nrows=max_len)
        if mode == "train":
            df = df[df["timestamp"] < 1512144000]
        else:
            df = df[df["timestamp"] >= 1512144000]
        df = df[df["item_list"].apply(lambda x: len(eval(x))) >= 3]
        self.user_id = [torch.tensor(v, dtype=torch.long) for v in df["user"].values]
        self.target_item_id = [torch.tensor(v, dtype=torch.long) for v in df["item"]]
        self.target_category_id = [torch.tensor(v, dtype=torch.long) for v in df["category"]]
        self.history_item_id = [torch.tensor(eval(s), dtype=torch.long) for s in df["item_list"]]
        self.history_category_id = [torch.tensor(eval(s), dtype=torch.long) for s in df["cate_list"]]
        self.label = [torch.tensor(v, dtype=torch.float) for v in df["behavior"].apply(reward_mapping)]
        del df
    
    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self, index):
        return self.user_id[index], self.target_item_id[index], self.target_category_id[index],\
            self.history_item_id[index], self.history_category_id[index], self.label[index]


def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    # batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    # data_length = [len(xi[0]) for xi in batch_data]
    # sent_seq = [xi[0] for xi in batch_data]
    # label = [xi[2] for xi in batch_data]
    # padded_sent_seq = nn.utils.rnn.pad_sequence(sent_seq, batch_first=True, padding_value=0)

    user_id = torch.tensor([data[0] for data in batch_data])
    target_item_id = torch.tensor([data[1] for data in batch_data])
    target_category_id = torch.tensor([data[2] for data in batch_data])
    history_item_id = nn.utils.rnn.pad_sequence([data[3] for data in batch_data], batch_first=True, padding_value=0)
    history_category_id = nn.utils.rnn.pad_sequence([data[4] for data in batch_data], batch_first=True, padding_value=0)
    seq_length = torch.tensor([len(data[3]) for data in batch_data])
    x = {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "target_category_id": target_category_id,
        "history_item_id": history_item_id,
        "history_category_id": history_category_id,
        "seq_length": seq_length,
    }
    y = torch.tensor([data[5] for data in batch_data])
    # y = torch.cat([data[5].unsqueeze(0) for data in batch_data], axis=0)
    return x, y
