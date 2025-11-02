import torch
import torch.nn as nn
import torch.nn.functional as F

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, dropout=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_proj = nn.Linear(v_dim, h_out * h_dim)
        self.q_proj = nn.Linear(q_dim, h_out * h_dim)
        self.gate = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.post = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim)
        )
        self._init_weights()
    def _init_weights(self):
        for layer in [self.v_proj, self.q_proj]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.out_features == 1:
                    nn.init.constant_(m.bias, -5.0)
                else:
                    nn.init.constant_(m.bias, 0.0)
        for m in self.post:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    def forward(self, v, q, q_skip, force_g_zero: bool = False):
        if v.dim() == 3: v = v.squeeze(1)
        if q.dim() == 3: q = q.squeeze(1)
        B = v.size(0)
        v_h = self.v_proj(v).view(B, self.h_out, self.h_dim)
        q_h = self.q_proj(q).view(B, self.h_out, self.h_dim)
        h_bilin = v_h * q_h
        scores = h_bilin.sum(dim=-1)
        att = F.softmax(scores, dim=-1).unsqueeze(-1)
        fused_bilin = (att * h_bilin).sum(dim=1)
        v_pool = v_h.mean(dim=1)
        q_pool = q_h.mean(dim=1)
        g = self.gate(torch.cat([v_pool, q_pool], dim=-1))
        if force_g_zero:
            g = torch.zeros_like(g)
        fused = q_skip + g * fused_bilin
        fused = self.dropout(fused)
        fused = self.post(fused)
        return fused, att.squeeze(-1), g.mean().item()

class dvib_ban(nn.Module):
    def __init__(self, trad_input_dim, esm_input_dim,
                 reduce_dim=256, hidden_size=256,
                 dropout=0.5, ban_heads=1, ban_dropout=0.1):
        super().__init__()
        self.reduce_dim = reduce_dim
        self.hidden_size = hidden_size
        self.trad_fc = nn.Sequential(
            nn.Linear(trad_input_dim, reduce_dim),
            nn.BatchNorm1d(reduce_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.esm_fc = nn.Sequential(
            nn.Linear(esm_input_dim, reduce_dim),
            nn.BatchNorm1d(reduce_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.q_skip_proj = (nn.Identity() if hidden_size == reduce_dim
                            else nn.Linear(reduce_dim, hidden_size))
        if hidden_size != reduce_dim:
            nn.init.xavier_normal_(self.q_skip_proj.weight)
            nn.init.constant_(self.q_skip_proj.bias, 0.0)
        self.ban = BANLayer(
            v_dim=REDUCE_DIM, q_dim=REDUCE_DIM,
            h_dim=hidden_size, h_out=ban_heads, dropout=ban_dropout
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 2)
        )
    def forward(self, trad_feat, esm_feat, force_g_zero: bool = False):
        trad_r = self.trad_fc(trad_feat).unsqueeze(1)
        esm_r  = self.esm_fc(esm_feat).unsqueeze(1)
        q_skip = esm_r.squeeze(1)
        q_skip_h = self.q_skip_proj(q_skip)
        fused, att, g_mean = self.ban(trad_r, esm_r, q_skip=q_skip_h, force_g_zero=force_g_zero)
        out = self.decoder(fused)
        return out, att, g_mean