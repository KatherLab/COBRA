import os
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from glob import glob
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from cobra.utils.load_cobra import get_cobra, get_cobraII
from cobra.model.cobra import Cobra
import argparse

from pathlib import Path
import pandas as pd
import numpy as np
import json

warnings.simplefilter(action="ignore", category=FutureWarning)

def load_patch_feats(h5_path,device):
    if not os.path.exists(h5_path):
        tqdm.write(f"File {h5_path} does not exist, skipping")
        return None
    with h5py.File(h5_path, "r") as f:
        feats = f["feats"][:]
        feats = torch.tensor(feats).to(device)
        coords = np.array(f["coords"][:])
        #coords = torch.tensor(coords).to(device)
    return feats, coords

def match_coords(feats_w,feats_a,coords_w,coords_a):

    # Sort coordinates and features based on sorted indices
    sorted_indices_w = np.lexsort((coords_w[:, 1], coords_w[:, 0]))
    sorted_indices_a = np.lexsort((coords_a[:, 1], coords_a[:, 0]))

    coords_w = coords_w[sorted_indices_w]
    coords_a = coords_a[sorted_indices_a]
    feats_w = feats_w[sorted_indices_w]
    feats_a = feats_a[sorted_indices_a]

    # Find matching indices after sorting
    idx_w = np.array([i for i, coord in enumerate(coords_w) if any(np.all(coord == coords_a, axis=1))])
    idx_a = np.array([i for i, coord in enumerate(coords_a) if any(np.all(coord == coords_w, axis=1))])
    assert len(idx_w) == len(idx_a), "Lengths do not match"
    assert len(idx_w) > 0, "No matching coordinates found"
    return feats_w[idx_w], feats_a[idx_a]

def get_cobra_feats(model,patch_feats_w,patch_feats_a,top_k=None):
    with torch.inference_mode():
        A = model(patch_feats_w,get_attention=True)
        # A.shape: (1,1,num_patches)
        if top_k:
            if A.size(-1) < top_k:
                top_k = A.size(-1)
            top_k_indices = torch.topk(A, top_k, dim=-1).indices # (1,1,top_k)
            top_k_A = A.gather(-1, top_k_indices) # (1,1,top_k)
            top_k_x = patch_feats_a.gather(1, top_k_indices.squeeze(0).unsqueeze(-1).expand(-1, -1, patch_feats_a.size(-1)))
            # top_k_x.shape: (1,top_k,feat_dim)
            cobra_feats = torch.bmm(F.softmax(top_k_A,dim=-1), top_k_x).squeeze(1)
            # cobra_feats.shape: (1,feat_dim)
        else:
            cobra_feats = torch.bmm(A, patch_feats_a).squeeze(1)
    return cobra_feats.squeeze(0)
    
def get_pat_embs(
    model,
    output_dir,
    feat_dir_w,
    feat_dir_a=None,
    output_file="cobra-feats.h5",
    model_name="COBRAII",
    slide_table_path=None,
    device="cuda",
    dtype=torch.float32,
    top_k=None,
    weighting_fm="Virchow2",
    aggregation_fm="Virchow2",
    microns=224,
):
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby("PATIENT")
    pat_dict = {}

    output_file = os.path.join(output_dir, output_file)

    if os.path.exists(output_file) and os.path.getsize(output_file) > 800:
        tqdm.write(f"Output file {output_file} already exists, skipping")
        return

    for patient_id, group in tqdm(patient_groups, leave=False):
        all_feats_list_w = []
        all_feats_list_a = []

        for _, row in group.iterrows():
            slide_filename = row["FILENAME"]
            h5_path_w = os.path.join(feat_dir_w, slide_filename)
            feats_w, coords_w = load_patch_feats(h5_path_w, device)
            if feats_w is None:
                continue
            if feat_dir_a:
                h5_path_a = os.path.join(feat_dir_a, slide_filename)
                feats_a, coords_a = load_patch_feats(h5_path_a, device)
            else:
                feats_a = feats_w
                coords_a = coords_w
            if feats_a is None:
                continue
            
            feats_w,feats_a = match_coords(feats_w,feats_a,coords_w,coords_a)

            all_feats_list_w.append(feats_w)
            all_feats_list_a.append(feats_a)

        if all_feats_list_w:
            all_feats_cat_w = torch.cat(all_feats_list_w, dim=0).unsqueeze(0)
            all_feats_cat_a = torch.cat(all_feats_list_a, dim=0).unsqueeze(0)
            #with torch.inference_mode():
            assert all_feats_cat_w.ndim == 3, (
                f"Expected 3D tensor, got {all_feats_cat_w.ndim}"
            )
            assert all_feats_cat_a.ndim == 3, (
                f"Expected 3D tensor, got {all_feats_cat_a.ndim}"
            )
            assert all_feats_cat_w.shape[1] == all_feats_cat_a.shape[1], (
                f"Expected same number of tiles, got {all_feats_cat_w.shape[1]} and {all_feats_cat_a.shape[1]}"
            )
            patient_feats = get_cobra_feats(model,all_feats_cat_w.to(dtype),all_feats_cat_a.to(dtype),top_k=top_k)
            pat_dict[patient_id] = {
                "feats": patient_feats.to(torch.float32)
                .detach()
                .squeeze()
                .cpu()
                .numpy(),
            }
        else:
            tqdm.write(f"No features found for patient {patient_id}, skipping")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for patient_id, data in pat_dict.items():
            f.create_dataset(f"{patient_id}", data=data["feats"])
        f.attrs["extractor"] = model_name
        f.attrs["top_k"] = top_k if top_k else "None"
        f.attrs["dtype"] = str(dtype)
        f.attrs["weighting_FM"] = weighting_fm
        f.attrs["aggregation_FM"] = aggregation_fm
        f.attrs["microns"] = microns



    tqdm.write(f"Finished extraction, saved to {output_file}")
    metadata = {
        "extractor": model_name,
        "top_k": top_k if top_k else "None",
        "dtype": str(dtype),
        "weighting_FM": weighting_fm,
        "aggregation_FM": aggregation_fm,
        "microns": microns,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as json_file:
        json.dump(metadata, json_file, indent=4)

def get_slide_embs(
    model,
    output_dir,
    feat_dir_w,
    feat_dir_a=None,
    output_file="cobra-feats.h5",
    model_name="COBRAII",
    device="cuda",
    dtype=torch.float32,
    top_k=None,
    weighting_fm="Virchow2",
    aggregation_fm="Virchow2",
    microns=224,
):
    slide_dict = {}

    tile_emb_paths_w = glob(f"{feat_dir_w}/**/*.h5", recursive=True)
    if feat_dir_a is not None:
        tile_emb_paths_a = glob(f"{feat_dir_a}/**/*.h5", recursive=True)
    else:
        tile_emb_paths_a = glob(f"{feat_dir_w}/**/*.h5", recursive=True)
    assert len(tile_emb_paths_w) == len(tile_emb_paths_a), (
        f"Expected same number of files, got {len(tile_emb_paths_w)} and {len(tile_emb_paths_a)}"
    )
    for tile_emb_path_w,tile_emb_path_a in zip(tqdm(tile_emb_paths_w), tile_emb_paths_a):
        slide_name = Path(tile_emb_path_w).stem
        feats_w = load_patch_feats(tile_emb_path_w, device)
        if feats_w is None:
            continue
        if feat_dir_a:
            tile_emb_path_a = os.path.join(feat_dir_a, f"{slide_name}.h5")
            feats_a = load_patch_feats(tile_emb_path_a, device)
        else:
            feats_a = feats_w
        if feats_a is None:
            continue
        tile_embs_w = feats_w[0].unsqueeze(0)
        tile_embs_a = feats_a[0].unsqueeze(0)

        assert tile_embs_w.ndim == 3, f"Expected 3D tensor, got {tile_embs_w.ndim}"
        assert tile_embs_a.ndim == 3, f"Expected 3D tensor, got {tile_embs_a.ndim}"
        assert tile_embs_w.shape[1] == tile_embs_a.shape[1], (
            f"Expected same number of tiles, got {tile_embs_w.shape[1]} and {tile_embs_a.shape[1]}"
        )

        slide_feats = get_cobra_feats(model, tile_embs_w.to(dtype), tile_embs_a.to(dtype), top_k=top_k)
        slide_dict[slide_name] = {
            "feats": slide_feats.to(torch.float32).detach().cpu().numpy(),
            "extractor": model_name,
        }

    output_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for slide_name, data in slide_dict.items():
            f.create_dataset(f"{slide_name}", data=data["feats"])
        f.attrs["extractor"] = model_name
        f.attrs["top_k"] = top_k if top_k else "None"
        f.attrs["dtype"] = str(dtype)
        f.attrs["weighting_FM"] = weighting_fm
        f.attrs["aggregation_FM"] = aggregation_fm
        f.attrs["microns"] = microns
    
    tqdm.write(f"Finished extraction, saved to {output_file}")
    metadata = {
        "extractor": model_name,
        "top_k": top_k if top_k else "None",
        "dtype": str(dtype),
        "weighting_FM": weighting_fm,
        "aggregation_FM": aggregation_fm,
        "microns": microns,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as json_file:
        json.dump(metadata, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Extract slide/pat embeddings using COBRA model"
    )
    parser.add_argument(
        "-d",
        "--download_model",
        action="store_true",
        help="Flag to download model weights",
    )
    parser.add_argument(
        "-w",
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        required=False,
        default=None,
        help="Top k tiles to use for slide/patient embedding",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "-f",
        "--feat_dir",
        type=str,
        required=True,
        help="Directory containing tile feature files",
    )
    parser.add_argument(
        "-g",
        "--feat_dir_a",
        type=str,
        required=False,
        default=None,
        help="Directory containing tile feature files",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default="COBRAII",
        help="model_name",
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
        default="Virchow2",
        help="patch encoder name used for aggregation",
    )
    parser.add_argument(
        "-e",
        "--h5_name",
        type=str,
        required=False,
        default="cobra_feats.h5",
        help="File name",
    )
    parser.add_argument(
        "-r",
        "--microns",
        type=int,
        required=False,
        default=224,
        help="microns per patch used for extraction",
    )
    parser.add_argument(
        "-s", "--slide_table", type=str, required=False, help="slide table path"
    )
    parser.add_argument(
        "-u", "--use_cobraI", action="store_true", help="whether to use cobra I or II"
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cobra_func = get_cobra if args.use_cobraI else get_cobraII
    if args.download_model:
            model = cobra_func(
                download_weights=args.download_model,
            )
    else:
        if args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                model = cobra_func(
                    checkpoint_path=args.checkpoint_path,
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint file {args.checkpoint_path} not found"
                )
        else:
            raise ValueError(
                "Please provide either a checkpoint path or set the download_model flag"
            )
    model = model.to(device)
    model.eval()

    if torch.cuda.get_device_capability()[0] < 8:
        print(
            f"\033[93mCOBRA (Mamba2) is designed to run on GPUs with compute capability 8.0 or higher!! "
            f"Your GPU has compute capability {torch.cuda.get_device_capability()[0]}. "
            f"We are forced to switch to mixed FP16 precision. This may lead to numerical instability and reduced performance!!\033[0m"
        )
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    if args.slide_table:
        get_pat_embs(
            model,
            args.output_dir,
            args.feat_dir,
            args.feat_dir_a,
            args.h5_name,
            args.model_name,
            args.slide_table,
            device,
            dtype=dtype,
            top_k=args.top_k,
            weighting_fm=args.patch_encoder,
            aggregation_fm=args.patch_encoder_a,
            microns=args.microns,
        )
    else:
        get_slide_embs(
            model,
            args.output_dir,
            args.feat_dir,
            args.feat_dir_a,
            args.h5_name,
            args.model_name,
            device=device,
            dtype=dtype,
            top_k=args.top_k,
            weighting_fm=args.patch_encoder,
            aggregation_fm=args.patch_encoder_a,
            microns=args.microns,
        )
        
if __name__ == "__main__":
    main()
