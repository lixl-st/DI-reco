from turtle import forward
import torch
import torch.nn as nn


class Ranking(nn.Module):
    def __init__(self, max_user=10000, max_item=10000, max_category=10):
        self.user_embedding = nn.Embedding(max_user, 16, freeze=False)
        self.item_embedding = nn.Embedding(max_item, 12, freeze=False)
        self.category_embedding = nn.Embedding(max_category, 4, freeze=False)
        self.encoder1 = nn.GRU(input_size=16, hidden_size=16, num_layers=1, batch_first=True)
        self.attention = nn.MultiheadAttention(16, 1, batch_first=True, need_weights=False)
        self.encoder2 = nn.GRU(input_size=16, hidden_size=16, num_layers=1, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(48),
            nn.Linear(48, 16),
            nn.ReLU(16),
            nn.BatchNorm1d(16),
            nn.Linear(16, 2),
            nn.Softmax(2)
        )

    def forward(self, input):
        # user profile
        user_emb = self.user_embedding(input["user_id"])
        # history
        history_item_emb = self.item_embedding(input["history_item_id"])
        history_category_emb = self.category_embedding(input["history_category_id"])
        history_emb = torch.cat([history_item_emb, history_category_emb], axis=-1)
        history_emb = nn.utils.rnn.pack_padded_sequence(history_emb, 50, batch_first=True)
        # target item
        target_item_emb = self.item_embedding(input["target_item_id"])
        target_category_emb = self.category_embedding(input["target_category_id"])
        target_emb = torch.cat([target_item_emb, target_category_emb], axis=-1)

        history_encoding, _ = self.encoder1(history_emb)
        attention = self.attention(history_encoding, target_emb, target_emb) * history_encoding
        # history_encoding, _ = self.encoder2(attention)
        avg = attention.sum(axis=1)
        all_encoding = torch.cat([avg, target_emb, user_emb], axis=1)
        out = self.fc_layers(all_encoding)
        return out


