import torch.nn as nn
import torch
import torch.nn.functional as F

class change_attention_module(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        self.embed = nn.Sequential(
            nn.Conv1d(self.input_dim, self.emb_dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.emb_dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.att = nn.Conv1d(self.emb_dim, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(self.input_dim // 2, 6)

    def forward(self, ref_feats, chg_feats):
        # batch_size, _, H, W = input_1.size()
        input_diff = chg_feats - ref_feats
        input_before = torch.cat([ref_feats, input_diff], 1)
        input_after = torch.cat([chg_feats, input_diff], 1)
        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = F.sigmoid(self.att(embed_before))
        att_weight_after = F.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(ref_feats)
        attended_1 = (ref_feats * att_1_expand).sum(2)  # (batch, input_dim)
        att_2_expand = att_weight_after.expand_as(chg_feats)
        attended_2 = (chg_feats * att_2_expand).sum(2)  # (batch, input_dim)
        input_attended = attended_2 - attended_1
        pred = self.fc1(input_attended)

        return pred, att_weight_before, att_weight_after, attended_1, attended_2, input_attended
