import torch
import torch.nn as nn

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out):
        super().__init__()
        self.h_dim = h_dim
        self.h_out = h_out
        
        self.v_proj = nn.Sequential(
            nn.Linear(v_dim, h_dim*2),
            nn.GELU(),
            nn.Linear(h_dim*2, h_dim)
        )
        self.q_proj = nn.Sequential(
            nn.Linear(q_dim, h_dim*2),
            nn.GELU(),
            nn.Linear(h_dim*2, h_dim)
        )
        
        self.attention = nn.MultiheadAttention(h_dim, h_out, batch_first=True)
        self.layer_norm = nn.LayerNorm(h_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(h_dim*2, h_dim),
            nn.GELU(),
            nn.Dropout()
        )

    def forward(self, v, q):
        v_proj = self.v_proj(v.squeeze(1))
        q_proj = self.q_proj(q.squeeze(1))
        
        att_output, att_weights = self.attention(
            q_proj.unsqueeze(1), 
            v_proj.unsqueeze(1), 
            v_proj.unsqueeze(1)
        )
        
        att_output = self.layer_norm(q_proj + att_output.squeeze(1))
        
        fused = self.fusion(torch.cat([v_proj, att_output], dim=-1))
        return fused, att_weights


class dvib_ban(nn.Module):
    def __init__(self, 
                 trad_input_dim,
                 esm_input_dim,
                 reduce_dim,
                 hidden_size,  
                 dropout,
                 ban_heads):
        super(dvib_ban, self).__init__()
        
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
        
        self.ban = BANLayer(
            v_dim=reduce_dim,
            q_dim=reduce_dim,
            h_dim=hidden_size,  
            h_out=ban_heads
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size*2, 2)
        )


    def forward(self, trad_feat, esm_feat):
        # Feature reduction
        trad_reduced = self.trad_fc(trad_feat).unsqueeze(1)  
        esm_reduced = self.esm_fc(esm_feat).unsqueeze(1)    
        
        # BAN fusion
        fused, att = self.ban(trad_reduced, esm_reduced)
        
        # Decoding
        outputs = self.decoder(fused)
        return outputs, att