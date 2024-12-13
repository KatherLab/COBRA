import os
import torch
import h5py
from tqdm import tqdm
from glob import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from cobra.utils.load_cobra import get_cobra
import argparse
from pathlib import Path

def get_tile_embs(h5_path,device):
    
    with h5py.File(h5_path, 'r') as f:
        feats = f['feats'][:]
    
    feats = torch.tensor(feats).to(torch.float32).to(device)
    return feats.unsqueeze(0)
    
def get_slide_embs(model,output_dir,feat_dir,device="cuda"):
    
    slide_dict = {}
    
    tile_emb_paths = glob(f"{feat_dir}/*.h5")
    for tile_emb_path in tqdm(tile_emb_paths):
        tile_embs = get_tile_embs(tile_emb_path,device)
        slide_name = Path(tile_emb_path).stem
        with torch.inference_mode():
            assert tile_embs.ndim == 3, f"Expected 3D tensor, got {tile_embs.ndim}"
            slide_feats = model(tile_embs)
            slide_dict[slide_name] = {
                'feats': slide_feats.detach().squeeze().cpu().numpy(),
                'extractor': f"Cobra"
            }
    output_file = os.path.join(output_dir, "cobra-feats.h5")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        for slide_name, data in slide_dict.items():
            f.create_dataset(f"{slide_name}", data=data['feats'])
            f.attrs['extractor'] = data['extractor']
        tqdm.write(f"Finished extraction, saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract slide embeddings using COBRA model")
    parser.add_argument('--download_model', action='store_false', help='Flag to download model weights')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted features')
    parser.add_argument('--feat_dir', type=str, required=True, help='Directory containing tile feature files')
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_cobra(download_weights=args.download_model, checkpoint_path=args.checkpoint_path)

    model = model.to(device)
    model.eval()
    
    get_slide_embs(model, args.output_dir, args.feat_dir, device)
    
    
if __name__ == "__main__":
    main()