import os
import torch
import h5py
from tqdm import tqdm
from glob import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from cobra.utils.load_cobra import get_cobra
from cobra.model.cobra import Cobra
import argparse
import yaml
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd

# def get_tile_embs(h5_path,device):
    
#     with h5py.File(h5_path, 'r') as f:
#         feats = f['feats'][:]
    
#     feats = torch.tensor(feats).to(torch.float32).to(device)
#     return feats.unsqueeze(0)
    
def get_slide_embs(model,output_dir,feat_dir,output_file="cobra-feats.h5",model_name="COBRAII",slide_table_path=None,device="cuda"):
    
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby('PATIENT')
    slide_dict = {}
    
    #tile_emb_paths = glob(f"{feat_dir}/*.h5")
    for patient_id, group in tqdm(patient_groups,leave=False):
        all_feats_list = []
        
        for _, row in group.iterrows():
            slide_filename = row['FILENAME']
            h5_path = os.path.join(feat_dir, slide_filename)
            if not os.path.exists(h5_path):
                tqdm.write(f"File {h5_path} does not exist, skipping")
                continue
            with h5py.File(h5_path, 'r') as f:
                feats = f['feats'][:]
                #extractor = f.attrs['extractor']

            feats = torch.tensor(feats).to(torch.float32).to(device)
            feat_dim = feats.shape[-1]
            all_feats_list.append(feats)
        
        if all_feats_list:
            # Concatenate all features for this patient along the second dimension
            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)
        
        
        #tile_embs = get_tile_embs(tile_emb_path,device)
        #slide_name = Path(tile_emb_path).stem
        with torch.inference_mode():
            assert all_feats_cat.ndim == 3, f"Expected 3D tensor, got {all_feats_cat.ndim}"
            slide_feats = model(all_feats_cat)
            slide_dict[patient_id] = {
                'feats': slide_feats.detach().squeeze().cpu().numpy(),
                'extractor': f"{model_name}-V2"
            }
    output_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        for patient_id, data in slide_dict.items():
            f.create_dataset(f"{patient_id}", data=data['feats'])
            f.attrs['extractor'] = data['extractor']
        tqdm.write(f"Finished extraction, saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract slide embeddings using COBRA model")
    parser.add_argument('-d','--download_model', action='store_false', help='Flag to download model weights')
    parser.add_argument('-w','--checkpoint_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('-c','--config', type=str, default=None, help='Path to model config')
    parser.add_argument('-o','--output_dir', type=str, required=True, help='Directory to save extracted features')
    parser.add_argument('-f','--feat_dir', type=str, required=True, help='Directory containing tile feature files')
    parser.add_argument('-m','--model_name', type=str, required=False,default="COBRAII", help='model_name')
    parser.add_argument('-e','--h5_name', type=str, required=False,default="cobra_feats.h5", help='File name')
    parser.add_argument('-s','--slide_table', type=str, required=True, help='slide table path')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.download_model:
        model = get_cobra(download_weights=args.download_model, checkpoint_path=args.checkpoint_path)
    elif args.checkpoint_path is not None and args.config is not None:
        try:
            with open(args.config, "r") as f:
                cfg_data = yaml.safe_load(f)
        except FileNotFoundError:
            with open(os.path.join(os.path.dirname(args.config),f"{os.path.basename(args.config).split('-')[0]}.yml"), "r") as f:
                cfg_data = yaml.safe_load(f)
        template_env = Environment(loader=FileSystemLoader(searchpath="./"))
        template = template_env.from_string(str(cfg_data))
        cfg = yaml.safe_load(template.render(**cfg_data))
        model = Cobra(embed_dim=cfg["model"]['dim'], layers=cfg["model"]['nr_mamba_layers'], 
                      dropout=cfg["model"]['dropout'], num_heads=cfg["model"]['nr_heads'],
                      att_dim=cfg["model"]['att_dim'],d_state=cfg["model"]['d_state'])
        try:
            chkpt = torch.load(args.checkpoint_path, map_location=device)
        except FileNotFoundError:
            chkpt = torch.load(args.checkpoint_path.replace("II",""), map_location=device)
        if "state_dict" in list(chkpt.keys()):
            chkpt = chkpt["state_dict"]
        momentum_enc = {k.split("momentum_enc.")[-1]:v for k,v in chkpt.items() if "momentum_enc" in k and "momentum_enc.proj" not in k}
        model.load_state_dict(momentum_enc)
    model = model.to(device)
    model.eval()
    
    get_slide_embs(model, args.output_dir, args.feat_dir,args.h5_name,args.model_name, args.slide_table, device)
    
    
if __name__ == "__main__":
    main()