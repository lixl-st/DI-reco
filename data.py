import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def reward_mapping(ss: str):
    if ss == "pv":
        return 0
    elif ss == "buy":
        return 1
    else:
        return 0

def history_action_mapping(ss: str):
    mapping_dict = {
        "pv": 0,
        "cart": 1,
        "fav": 2,
        "buy": 3,
        "click": 2,
    }
    return mapping_dict[ss]


class RecoData(Dataset):
    def __init__(self, df) -> None:
        super().__init__()
        self.user_id = [torch.tensor(v, dtype=torch.long) for v in df["user"].values]
        self.target_item_id = [torch.tensor(v, dtype=torch.long) for v in df["item"]]
        self.target_category_id = [torch.tensor(v, dtype=torch.long) for v in df["category"]]
        self.history_item_id = [torch.tensor(json.loads(v), dtype=torch.long) for v in df["item_list"]]
        self.history_category_id = [torch.tensor(json.loads(v), dtype=torch.long) for v in df["cate_list"]]
        self.history_action_id = [torch.tensor(v, dtype=torch.long) for v in df["behavior_list"].apply(lambda x: [history_action_mapping(i) for i in eval(x)])]
        self.label = [torch.tensor(reward_mapping(v), dtype=torch.float) for v in df["behavior"]]
        del df
    
    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self, index):
        return self.user_id[index], self.target_item_id[index], self.target_category_id[index],\
            self.history_item_id[index], self.history_category_id[index], self.history_action_id[index], self.label[index]


def collate_fn(batch_data):

    user_id = torch.tensor([data[0] for data in batch_data])
    target_item_id = torch.tensor([data[1] for data in batch_data])
    target_category_id = torch.tensor([data[2] for data in batch_data])
    history_item_id = nn.utils.rnn.pad_sequence([data[3] for data in batch_data], batch_first=True, padding_value=0)
    seq_length = torch.tensor([len(data[3]) for data in batch_data])
    history_category_id = nn.utils.rnn.pad_sequence([data[4] for data in batch_data], batch_first=True, padding_value=0)
    history_action_id = nn.utils.rnn.pad_sequence([data[5] for data in batch_data], batch_first=True, padding_value=0)
    x = {
        "user_id": user_id,
        "target_item_id": target_item_id,
        "target_category_id": target_category_id,
        "history_item_id": history_item_id,
        "seq_length": seq_length,
        "history_category_id": history_category_id,
        "history_action_id": history_action_id,
    }
    y = torch.tensor([data[6] for data in batch_data])
    return x, y
