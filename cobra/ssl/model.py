<<<<<<< HEAD
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from cobra.utils.mamba2 import Mamba2Enc
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
from cobra.utils.mamba import Mamba2Enc
>>>>>>> d43e2a91628f072f319617895e62ce10bdf241fa
from cobra.utils.abmil import BatchedABMIL
from einops import rearrange 


<<<<<<< HEAD

=======
>>>>>>> d43e2a91628f072f319617895e62ce10bdf241fa
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
<<<<<<< HEAD
    def __init__(self,embed_dim, c_dim, input_dims=[384,512,1024,1280,1536], num_heads=8,layer=2,dropout=0.25,att_dim=256,d_state=64):
        super().__init__()
        
        # self.embed = nn.ModuleDict({ "384":Embed(384,embed_dim),
        #                               "512":Embed(512,embed_dim),
        #                             #"768":Embed(768,embed_dim),
        #                            "1024":Embed(1024,embed_dim),
        #                            "1280":Embed(1280,embed_dim),
        #                             "1536":Embed(1536,embed_dim)})

        self.embed = nn.ModuleDict({str(d):Embed(d,embed_dim) for d in input_dims})
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mamba_enc = Mamba2Enc(embed_dim,embed_dim,n_classes=embed_dim,layer=layer,dropout=dropout,d_state=d_state)
=======
    def __init__(self,embed_dim, c_dim, num_heads=8,layer=2,dropout=0.25):
        super().__init__()
        
        self.embed = nn.ModuleDict({"768":Embed(768,embed_dim),
                                   "1024":Embed(1024,embed_dim),
                                   "1280":Embed(1280,embed_dim),
                                    "1536":Embed(1536,embed_dim)})
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mamba_enc = Mamba2Enc(embed_dim,embed_dim,n_classes=embed_dim,layer=layer,dropout=dropout)
>>>>>>> d43e2a91628f072f319617895e62ce10bdf241fa
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,4*embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(4*embed_dim,c_dim),
            nn.BatchNorm1d(c_dim),
        )
   
        self.num_heads = num_heads
<<<<<<< HEAD
        self.attn = nn.ModuleList([BatchedABMIL(input_dim=int(embed_dim/num_heads),hidden_dim=att_dim,
                                        dropout=dropout,n_classes=1) for _ in range(self.num_heads)]) #,hidden_dim=int(embed_dim/num_heads)
=======
        self.attn = nn.ModuleList([BatchedABMIL(input_dim=int(embed_dim/num_heads),hidden_dim=int(embed_dim/num_heads),dropout=dropout,n_classes=1) for _ in range(self.num_heads)])
>>>>>>> d43e2a91628f072f319617895e62ce10bdf241fa

    def forward(self, x, lens=None):

        if lens is not None:
            assert len(x)==len(lens)
            logits = torch.concat([self.embed[str(lens[i].item())](x[i,:,:lens[i].item()]).unsqueeze(0) for i in range(len(x))],dim=0)
        else:
            logits = x

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
        
        h = torch.bmm(A,h).squeeze(1)
        feats = self.proj(h)
        
        assert len(feats.shape)==2, feats.shape
        return feats
    
class MoCo(nn.Module): # adapted from https://github.com/facebookresearch/moco-v3
<<<<<<< HEAD
    def __init__(self,embed_dim, c_dim, input_dims=[384,512,1024,1280,1536], num_heads=8, nr_mamba_layers=2, gpu_id=0, T=0.2,dropout=0.25,
                 att_dim=256,d_state=64):
        super().__init__()

        self.T = T
        self.base_enc = Cobra(embed_dim,c_dim,input_dims,num_heads,layer=nr_mamba_layers,dropout=dropout,
                              att_dim=att_dim,d_state=d_state)
        self.momentum_enc = Cobra(embed_dim,c_dim,input_dims,num_heads,layer=nr_mamba_layers,dropout=None,
                                  att_dim=att_dim,d_state=d_state)
=======
    def __init__(self,embed_dim, c_dim, num_heads=8, nr_mamba_layers=2, gpu_id=0, T=0.2,dropout=0.25):
        super().__init__()

        self.T = T
        self.base_enc = Cobra(embed_dim,c_dim,num_heads,layer=nr_mamba_layers,dropout=dropout)
        self.momentum_enc = Cobra(embed_dim,c_dim,num_heads,layer=nr_mamba_layers,dropout=None)
>>>>>>> d43e2a91628f072f319617895e62ce10bdf241fa
        self.predictor = nn.Sequential(
            nn.LayerNorm(c_dim),
            nn.Linear(c_dim,2*c_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2*c_dim,c_dim),
            nn.BatchNorm1d(c_dim),
        ).cuda(gpu_id)

        for param_b, param_m in zip(self.base_enc.parameters(), self.momentum_enc.parameters()):
            param_m.data.copy_(param_b.data)  
            param_m.requires_grad = False 

    @torch.no_grad()
    def _update_momentum_encoder(self, m=0.99):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_enc.parameters(), self.momentum_enc.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)


    def forward(self, x1, x2, sizes_1=None, sizes_2=None,m=0.99):

        x1_enc = self.base_enc(x1,sizes_1)
        x2_enc = self.base_enc(x2,sizes_2)
        q1 = self.predictor(x1_enc)
        q2 = self.predictor(x2_enc)
       
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m=m)

            k1 = self.momentum_enc(x1,sizes_1)
            k2 = self.momentum_enc(x2,sizes_2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    def contrastive_loss(self, q, k):
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output