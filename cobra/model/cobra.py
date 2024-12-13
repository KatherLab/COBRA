import torch
import torch.nn as nn
import sys

from cobra.utils.mamba2 import Mamba2Enc
from cobra.utils.abmil import BatchedABMIL
import torch.nn.functional as F
from einops import rearrange
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024,dropout=0.25):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, embed_dim),
             nn.Dropout(dropout) if dropout else nn.Identity(),
             nn.SiLU(),
             nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x) 

class Cobra(nn.Module):
    def __init__(self,embed_dim=768, mamba_layers=2,dropout=0.25,num_heads=8):
        super().__init__()
        
        self.embed = nn.ModuleDict({"768":Embed(768,embed_dim),
                                   "1024":Embed(1024,embed_dim),
                                   "1280":Embed(1280,embed_dim),
                                    "1536":Embed(1536,embed_dim),})
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mamba_enc = Mamba2Enc(embed_dim,embed_dim,n_classes=embed_dim,layer=mamba_layers,dropout=dropout)
        
        self.num_heads = num_heads
        self.attn = nn.ModuleList([BatchedABMIL(input_dim=int(embed_dim/num_heads),hidden_dim=int(embed_dim/num_heads),
                                                dropout=dropout,n_classes=1) for _ in range(self.num_heads)])
        
    def forward(self, x, multi_fm_mode=False, fm_idx=None, get_attention=False):
        if multi_fm_mode:
            fm_embs = torch.concat([self.embed[str(xi.shape[-1])](xi) for xi in x],dim=0)
            assert fm_embs.shape[-1]==self.embed_dim, fm_embs.shape
            assert len(fm_embs.shape)==3, fm_embs.shape
            assert fm_embs.shape[0]==len(x), fm_embs.shape
            logits = torch.mean(fm_embs,dim=0) 
        else:
            logits = self.embed[str(x.shape[-1])](x)

        h = self.norm(self.mamba_enc(logits))
        
        if self.num_heads > 1:
            h_ = rearrange(h, 'b t (e c) -> b t e c',c=self.num_heads)

            attention = []
            for i, attn_net in enumerate(self.attn):
                _, processed_attention = attn_net(h_[:, :, :, i], return_raw_attention = True)
                attention.append(processed_attention)
                
            A = torch.stack(attention, dim=-1)

            A = rearrange(A, 'b t e c -> b t (e c)',c=self.num_heads).mean(-1).unsqueeze(-1)
            A = torch.transpose(A,2,1)
            A = F.softmax(A, dim=-1) 
        else: 
            A = self.attn[0](h)
        
        if multi_fm_mode:
            if fm_idx:
                feats = torch.bmm(A,x[fm_idx]).squeeze(0).squeeze(0)
            else:
                feats=[]
                for i,xi in enumerate(x):
                    feats.append(torch.bmm(A,xi).squeeze(0).squeeze(0))
                    assert len(feats[i].shape)==1 and feats[i].shape[0]==xi.shape[-1], feats[i].shape
        else:
            feats = torch.bmm(A,x).squeeze(1)
        
        if get_attention:
            return feats, A
        return feats
