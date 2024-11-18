import os
import torch
import torch.nn as nn
import h5py
import pandas as pd
from tqdm import tqdm
import warnings
from tqdm import tqdm 
warnings.simplefilter(action='ignore', category=FutureWarning)

import yaml
from jinja2 import Environment, FileSystemLoader
from CobraCode.utils.abmil import BatchedABMIL
from CobraCode.utils.mamba import Mamba2Enc
import torch.nn.functional as F
from einops import rearrange

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
    def __init__(self,embed_dim, c_dim,num_heads=8,layer=2):
        super().__init__()
        self.embed = nn.ModuleDict({"768":Embed(768,embed_dim),
                                    "1024":Embed(1024,embed_dim),
                                  "1280":Embed(1280,embed_dim),
                                  "1536":Embed(1536,embed_dim),})
        self.norm = nn.LayerNorm(embed_dim)
        self.mamba_enc = Mamba2Enc(embed_dim,embed_dim,n_classes=embed_dim,layer=layer)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,4*embed_dim),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(4*embed_dim,c_dim),
            nn.BatchNorm1d(c_dim),
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.ModuleList([BatchedABMIL(input_dim=int(embed_dim/num_heads),
                                                hidden_dim=int(embed_dim/num_heads),dropout=0.25,n_classes=1) for _ in range(self.num_heads)])
    
    def forward(self, xs,get_tile_embs=False):
        fm_embs = torch.concat([self.embed[str(x.shape[-1])](x) for x in xs],dim=0)
        assert fm_embs.shape[-1]==self.embed_dim, fm_embs.shape
        assert len(fm_embs.shape)==3, fm_embs.shape
        assert fm_embs.shape[0]==len(xs), fm_embs.shape
        embs = torch.mean(fm_embs,dim=0)
        assert embs.shape[-1]==self.embed_dim, embs.shape
        
        h = self.norm(self.mamba_enc(embs))
        
        if self.num_heads > 1:
            h_ = rearrange(h, 'b t (e c) -> b t e c',c=self.num_heads)

            attention = []
            for i, attn_net in enumerate(self.attn):
                _, processed_attention = attn_net(h_[:, :, :, i], return_raw_attention = True)
                attention.append(processed_attention)
            A = torch.stack(attention, dim=-1)
            A = rearrange(A, 'b t e c -> b t (e c)',c=self.num_heads).mean(-1).unsqueeze(-1)
            A = torch.transpose(A,2,1)
            A = F.softmax(A, dim=-1) ,att_dim=att_dim
        else: 
            A = self.attn[0](h)
        
        feats = []
        for i,x in enumerate(xs):
            feats.append(self.proj(torch.bmm(A,x).squeeze(0).squeeze(0)))
            assert len(feats[i].shape)==1 and feats[i].shape[0]==x.shape[-1], feats[i].shape   
        return feats

def get_feats(group,feat_dir,device):
    all_feats_list = []
            
    for _, row in group.iterrows():
        slide_filename = row['FILENAME']
        h5_path = os.path.join(feat_dir, slide_filename)

        if not os.path.exists(h5_path):
            continue

        with h5py.File(h5_path, 'r') as f:
            feats = f['feats'][:]

        feats = torch.tensor(feats).to(torch.float32).to(device)
        all_feats_list.append(feats)
    if all_feats_list:
        all_feats_cat = torch.cat(all_feats_list, dim=0)
        
        return all_feats_cat.unsqueeze(0)
    else:
        return "None"
    
def postprocess(output_dir, feat_dirs,device,models,cohort,model_name):
    slide_table_path = f"/path/to/slide_tables/slide_table_{cohort}.csv"
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby('PATIENT')
    patient_dict = {}
    
    if not os.path.exists(os.path.join(output_dir, f"{model_name}-{models[0]}",f"{cohort}.h5")):
    
        for patient_id, group in tqdm(patient_groups,leave=False):
            fm_list = [get_feats(group,f,device) for f in feat_dirs]
            
            if not ("None" in fm_list):
                with torch.inference_mode():
                    slide_feats = model(fm_list)

                patient_dict[patient_id] = {
                    'feats': slide_feats.detach().squeeze().cpu().numpy(),
                    'extractor': f"{model_name}"
                }

        output_path_m = os.path.join(output_dir, f"{model_name}",f"{cohort}.h5")
        os.makedirs(os.path.dirname(output_path_m), exist_ok=True)
        with h5py.File(output_path_m, 'w') as f:
            for patient_id, data in patient_dict.items():
                f.create_dataset(f"{patient_id}", data=data['feats'])
                f.attrs['extractor'] = data['extractor']

        tqdm.write(f"Finished cohort {cohort}")
    else:
        tqdm.write(f"Skipping, already exists...")

enc = "momentum_enc"
get_weighting = False

config_paths = ["/path/to/config/config.yml"]

ckpt_dirs = ["/patch/to/chkpt/cobra.pth.tar"]

feat_dirs = ["/path/to/patch/embs/features"]
model_names = [f'cobra']

for (feat_dir,config_path,ckpt_dir,model_name) in tqdm(zip(feat_dirs,config_paths,
                                                           ckpt_dirs,model_names),leave=False):

    with open(config_path, 'r') as f:
        cfg_data = yaml.safe_load(f)

    template_env = Environment(loader=FileSystemLoader(searchpath='./'))
    template = template_env.from_string(str(cfg_data))

    cfg_data = yaml.safe_load(template.render(**cfg_data))

    device = "cuda"
    output_directory = f"/path/to/output/features/{model_name}"   
    embed_dim = cfg_data["model"]["dim"]
    c_dim = cfg_data["model"]["l_dim"]
    nr_heads = cfg_data["model"]["nr_heads"]
    

    chkpt = torch.load(ckpt_dir,map_location=device)
    if "state_dict" in list(chkpt.keys()):
        chkpt = chkpt["state_dict"]
    base_enc = {k.split(f"{enc}.")[-1]:v for k,v in chkpt.items() if enc in k}
    nr_mamba_layers = cfg_data["model"]["nr_mamba_layers"]
    model = Cobra(embed_dim,c_dim,layer=nr_mamba_layers,nr_heads=nr_heads)
    msg = model.load_state_dict(base_enc,strict=False)
    print(msg)
    model.proj = nn.Identity()

    if torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    print(f"{model_name} slide encoder successfully initialized...\n")

    models = cfg_data["general"]["fms"]

    for cohort in tqdm(os.listdir(feat_dir)):
        if "cptac" in cohort or "tcga" in cohort:
            feat_dirs = [os.path.join(feat_dir, cohort, fm) for fm in models]
            postprocess(
                output_dir=output_directory,
                feat_dirs=feat_dirs,
                device=device,
                models=models,
                cohort=cohort,
                model_name=model_name,
                dim=embed_dim,
                get_weighted_avg=get_weighting,)
            
    print(f"Done preprocessing {model_name}")
