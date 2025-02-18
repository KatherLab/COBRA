import torch 
import pandas as pd
import os
from os.path import exists
from tqdm import tqdm 
import argparse
import h5py
import json
import numpy as np

from cobra.model.cobra import Cobra
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

def match_coords(coords,features, namelist):
    int_coords = coords.cpu().numpy().astype(int)
    float_cache_coords = np.array([[float(f.split("_(")[1].split(")")[0].split(", ")[0]),
                    float(f.split("_(")[1].split(")")[0].split(", ")[1])] for f in namelist if f.endswith(".jpg")])
    int_cache_coords = float_cache_coords.astype(int)
    #assert len(int_cache_coords) == len(coords)
    sorted_coords_idx = np.lexsort((int_coords[:, 1], int_coords[:, 0]))
    sorted_cache_coords_idx = np.lexsort((int_cache_coords[:, 1], int_cache_coords[:, 0]))

    sorted_coords = int_coords[sorted_coords_idx]
    sorted_feat_float_coords = coords[sorted_coords_idx]
    sorted_cache_coords = int_cache_coords[sorted_cache_coords_idx]
    sorted_cache_float_coords = float_cache_coords[sorted_cache_coords_idx]
    sorted_feats = features[sorted_coords_idx]

    idx_coords = np.array([i for i, coord in enumerate(sorted_coords) if any(np.all(coord == sorted_cache_coords, axis=1))])
    idx_cache = np.array([i for i, coord in enumerate(sorted_cache_coords) if any(np.all(coord == sorted_coords, axis=1))])
    #breakpoint()
    matched_cache_coords = sorted_cache_float_coords[idx_cache]
    assert len(matched_cache_coords)>0
    #breakpoint()
    return matched_cache_coords, sorted_feats[idx_coords], sorted_feat_float_coords[idx_coords]

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
            #x, y = coord.tolist()
            #x_matched, y_matched = match_coords(coord.cpu().numpy().astype(int).tolist(), z.namelist())
            img_name = f"tile_({coord[0]}, {coord[1]}).jpg"
            if img_name in z.namelist():
                img_data = z.read(img_name)
                output_img_name = f"{i+1}_{cache_dict[i]}_{img_name}"
                output_img_path = os.path.join(pat_output_dir, output_img_name)
                with open(output_img_path, 'wb') as img_file:
                    img_file.write(img_data)
            else:
                tqdm.write(f"Image {img_name} not found in zip file {zip_file}")
    
def get_top_k(model, feats, coords, cache_coords, k=8):
    with torch.inference_mode():
        A = model(feats.to(torch.float32),get_attention=True)
        top_k_indices = torch.topk(A, k, dim=-1).indices # (1,1,top_k)
        top_k_A = A.gather(-1, top_k_indices)
        top_k_coords = coords.to(torch.float32).gather(0, top_k_indices.squeeze(1).squeeze(0).unsqueeze(-1).repeat(1,2))
        top_K_cache_coords = cache_coords.gather(0, top_k_indices.squeeze(1).squeeze(0).unsqueeze(-1).repeat(1,2))
    
    return top_k_A.squeeze(0), top_k_coords.squeeze(0), top_K_cache_coords.squeeze(0), top_k_indices.squeeze(1).squeeze(0)

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
        cache_coords = []
        file_dict = {}      
        feat_length = 0
        for _, row in group.iterrows():
            slide_filename = row["FILENAME"]
            
            h5_path_w = os.path.join(patch_feat_dir, slide_filename)
            feats_w, coords_w = load_patch_feats(h5_path_w, device)  
            feat_length += feats_w.shape[0]
            file_dict[slide_filename.split(".")[0]] = feat_length 
            #all_feats_list.append(feats_w)
            #all_coords_list.append(coords_w)

            # Match coordinates with the coords of the jpgs from the zip file
            zip_file = None
            for file in os.listdir(tile_dir):
                if file.split(".")[0] == slide_filename.split(".")[0] and file.endswith(".zip"):
                    zip_file = os.path.join(tile_dir, file)
                    break

            if zip_file is None:
                tqdm.write(f"No zip file found for slide {slide_filename}, skipping")
                continue

            with zipfile.ZipFile(zip_file, 'r') as z:
                # matched_coords = []
                # for coord in coords_w:
                #     try:
                #         x_matched, y_matched = match_coords(coord.cpu().numpy().astype(int).tolist(), z.namelist())
                #         matched_coords.append([x_matched, y_matched])
                #     except IndexError:
                #         tqdm.write(f"Coordinate {coord} not found in zip file {zip_file}, skipping")
                #         continue
                matched_coords, feats_w, coords_w = match_coords(coords_w,feats_w, z.namelist())
            all_feats_list.append(feats_w)
            all_coords_list.append(coords_w)
            cache_coords.append(torch.tensor(matched_coords).to(device))
                #matched_coords = torch.tensor(matched_coords).to(device)
                #all_coords_list[-1] = matched_coords


        if all_feats_list:
            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)
            all_coords_cat = torch.cat(all_coords_list, dim=0)
            cache_coords = torch.cat(cache_coords, dim=0)

            weights, coords, cache_coords, indices = get_top_k(model, all_feats_cat, all_coords_cat, cache_coords, k=k)
            get_cache(cache_coords,tile_dir,output_dir,pat,file_dict,indices)
            metadata = {
                "weights": weights.cpu().tolist(),
                "total_patches": all_feats_cat.shape[1],
                "uniform_threshold": 1./all_feats_cat.shape[1],
                "feat_coords": coords.cpu().tolist(),
                "coordinates": cache_coords.cpu().tolist(),
                "weighting_fm": weighting_fm,
                "aggregation_fm": agregation_fm,
                "microns_per_patch": microns
            }
            output_path = os.path.join(output_dir,pat, f"{pat}_metadata.json")
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
        #default="Virchow2",
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
        #default="/p/scratch/mfmpm/tim/data/Engelmann/features-virchow2-10x-2/virchow2-9286a880",
        required=False,
        help="Directory containing tile feature files",
    )
    parser.add_argument(
        "-t",
        "--tile_dir",
        type=str,
        required=False,
        default="/p/scratch/mfmpm/tim/data/Engelmann/cache-conchv1_5-10x",
        #default="/p/scratch/mfmpm/tim/data/Engelmann/cahce-v2-10x",
        help="Directory containing cached zip files filled with jpg tiles",
    )
    parser.add_argument(
        "-w",
        "--checkpoint_path",
        type=str,
        default="/p/scratch/mfmpm/tim/code/COBRA/weights/cobraII.pth.tar",
        #default="/p/scratch/mfmpm/tim/data/Engelmann/v2_20x_cobra_chkpt_split-3/model.ckpt",
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
    # model = Cobra(layers=1,input_dims=[512,1024,1280,1536],
    #               num_heads=4,dropout=0.2,att_dim=256,d_state=128)
    # checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    # cobra_weights = {k.split("cobra.")[-1]:v for k,v in checkpoint["state_dict"].items() if "cobra" in k}
    # #breakpoint()
    # model.load_state_dict(cobra_weights)
    model.eval()
    model.to(device)

    get_top_tiles(model, args.feat_dir, args.tile_dir, slide_table, 
                  args.patch_encoder, args.patch_encoder_a, args.output_dir, 
                  microns=args.microns, device=device)


if __name__ == "__main__":
    main()
