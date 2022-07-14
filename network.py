import torch
import torch.nn as nn


class Ranking(nn.Module):
    def __init__(self, max_user=1000000, max_item=10000, max_category=1000, max_action_type=4):
        super().__init__()
        self.user_embedding = nn.Embedding(max_user, 8)
        self.item_embedding = nn.Embedding(max_item, 8)
        self.category_embedding = nn.Embedding(max_category, 4)
        self.action_embedding = nn.Embedding(max_action_type, 4)
        self.encoder = nn.GRU(input_size=16, hidden_size=12, num_layers=1, batch_first=True)
        self.attention = nn.MultiheadAttention(12, 1, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(32),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # user profile
        user_emb = self.user_embedding(input["user_id"])
        # history
        history_item_emb = self.item_embedding(input["history_item_id"])
        history_category_emb = self.category_embedding(input["history_category_id"])
        history_action_embedding = self.action_embedding(input["history_action_id"])
        history_emb = torch.cat([history_item_emb, history_category_emb, history_action_embedding], axis=-1)
        history_emb = nn.utils.rnn.pack_padded_sequence(history_emb, input["seq_length"], batch_first=True, enforce_sorted=False)

        # target item
        target_item_emb = self.item_embedding(input["target_item_id"])
        target_category_emb = self.category_embedding(input["target_category_id"])
        target_emb = torch.cat([target_item_emb, target_category_emb], axis=-1)

        history_encoding, _ = self.encoder(history_emb)
        history_encoding, _ = nn.utils.rnn.pad_packed_sequence(history_encoding, batch_first=True)
        kv = target_emb.unsqueeze(1)
        attention, _ = self.attention(history_encoding, kv, kv)
        avg = attention.mean(axis=1)
        all_encoding = torch.cat([avg, target_emb, user_emb], axis=1)
        out = self.fc_layers(all_encoding)
        return out.squeeze()


