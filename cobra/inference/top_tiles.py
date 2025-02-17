import torch 
import pandas as pd
import os
from os.path import exists
from tqdm import tqdm 
import argparse
import h5py
import json

from cobra.utils.load_cobra import get_cobraII
import zipfile
#from cobra.inference.extract_feats import load_patch_feats

def load_patch_feats(h5_path,device):
    if not os.path.exists(h5_path):
        tqdm.write(f"File {h5_path} does not exist, skipping")
        return None
    with h5py.File(h5_path, "r") as f:
        feats = f["feats"][:]
        feats = torch.tensor(feats).to(device)
        coords = torch.tensor(f["coords"][:]).to(device)
        #coords = torch.tensor(coords).to(device)
    return feats, coords

def get_cache_dict(file_dict, indices):
    cache_dict = {}
    for i, idx in enumerate(indices):
        for key, value in file_dict.items():
            if idx < value:
                cache_dict[i] = key
                break
    return cache_dict

def get_cache(coords,tile_dir,output_dir,pat,file_dict,indices):
    zip_file = None
    cache_dict = get_cache_dict(file_dict, indices)
    for i,coord in enumerate(coords):
        for file in os.listdir(tile_dir):
            if file.split(".")[0] == cache_dict[i] and file.endswith(".zip"):
                zip_file = os.path.join(tile_dir, file)
                break

        if zip_file is None:
            tqdm.write(f"No zip file found for patient {pat}, skipping")
            return

        # Create the output directory if it doesn't exist
        pat_output_dir = os.path.join(output_dir, pat)
        os.makedirs(pat_output_dir, exist_ok=True)

        # Extract the required images from the zip file
        with zipfile.ZipFile(zip_file, 'r') as z:
            # for j, coord in enumerate(coords):
            x, y = coord.tolist()
            img_name = f"tile_({x}, {y}).jpg"
            if img_name in z.namelist():
                img_data = z.read(img_name)
                output_img_name = f"{i+1}_{cache_dict[i]}_{img_name}"
                output_img_path = os.path.join(pat_output_dir, output_img_name)
                with open(output_img_path, 'wb') as img_file:
                    img_file.write(img_data)
            else:
                tqdm.write(f"Image {img_name} not found in zip file {zip_file}")
    
def get_top_k(model, feats, coords, k=8):
    with torch.inference_mode():
        A = model(feats.to(torch.float32),get_attention=True)
        top_k_indices = torch.topk(A, k, dim=-1).indices # (1,1,top_k)
        top_k_A = A.gather(-1, top_k_indices)
        top_k_coords = coords.gather(1, top_k_indices.squeeze(0).unsqueeze(-1).repeat(1,1,2))
    
    return top_k_A.squeeze(0), top_k_coords.squeeze(0), top_k_indices.squeeze(1).squeeze(0)  

def get_top_tiles(model,patch_feat_dir, 
                  tile_dir, 
                  slide_table, 
                  weighting_fm, 
                  agregation_fm, 
                  output_dir,
                  k=8,
                  microns=224,
                  device="cuda"):

    patient_groups = slide_table.groupby("PATIENT")

    if not exists(output_dir):
        os.makedirs(output_dir)

    for pat, group in tqdm(patient_groups):
        all_feats_list = []
        all_coords_list = []  
        file_dict = {}      
        feat_length = 0
        for _, row in group.iterrows():
            slide_filename = row["FILENAME"]
            
            h5_path_w = os.path.join(patch_feat_dir, slide_filename)
            feats_w, coords_w = load_patch_feats(h5_path_w, device)  
            feat_length += feats_w.shape[1]
            file_dict[slide_filename.split(".")[0]] = feat_length 
            all_feats_list.append(feats_w)
            all_coords_list.append(coords_w)

        if all_feats_list:
            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)
            all_coords_cat = torch.cat(all_coords_list, dim=0).unsqueeze(0)
        
            weights, coords, indices = get_top_k(model, all_feats_cat, all_coords_cat, k=k)
            get_cache(coords,tile_dir,output_dir,pat,file_dict,indices)
            metadata = {
                "weights": weights.cpu().tolist(),
                "coordinates": coords.cpu().tolist(),
                "weighting_fm": weighting_fm,
                "aggregation_fm": agregation_fm,
                "microns_per_patch": microns
            }
            output_path = os.path.join(output_dir, f"{pat}_metadata.json")
            with open(output_path, "w") as f:
                json.dump(metadata, f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract slide/pat embeddings using COBRA model"
    )
    parser.add_argument(
        "-p",
        "--patch_encoder",
        type=str,
        required=False,
        default="Virchow2",
        help="patch encoder name",
    )
    parser.add_argument(
        "-a",
        "--patch_encoder_a",
        type=str,
        required=False,
        default="ConchV1-5",
        help="patch encoder name used for aggregation",
    )
    parser.add_argument(
        "-s", "--slide_table", type=str, 
        default="/p/scratch/mfmpm/tim/data/Engelmann/metadata/slide_table_final_edited.csv",
        help="slide table path"
    )
    parser.add_argument(
        "-f",
        "--feat_dir",
        type=str,
        default="/p/scratch/mfmpm/tim/data/Engelmann/features-virchow2-5x/virchow2-9286a880",
        required=False,
        help="Directory containing tile feature files",
    )
    parser.add_argument(
        "-t",
        "--tile_dir",
        type=str,
        required=False,
        default="/p/scratch/mfmpm/tim/data/Engelmann/cache-conchv1_5-10x",
        help="Directory containing cached zip files filled with jpg tiles",
    )
    parser.add_argument(
        "-w",
        "--checkpoint_path",
        type=str,
        default="/p/scratch/mfmpm/tim/code/COBRA/weights/cobraII.pth.tar",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "-r",
        "--microns",
        type=int,
        required=False,
        default=448,
        help="microns per patch used for extraction",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        default="/p/scratch/mfmpm/tim/data/Engelmann/cobra-hybrid-top-tiles",
        help="Directory to save top tiles in",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"


    slide_table = pd.read_csv(args.slide_table)
    

    model = get_cobraII(download_weights=(not exists(args.checkpoint_path)), 
                        checkpoint_path=args.checkpoint_path)
    model.eval()
    model.to(device)

    get_top_tiles(model, args.feat_dir, args.tile_dir, slide_table, 
                  args.patch_encoder, args.patch_encoder_a, args.output_dir, 
                  microns=args.microns, device=device)


if __name__ == "__main__":
    main()
