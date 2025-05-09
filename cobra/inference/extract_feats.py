import os
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm
from glob import glob
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from cobra.utils.load_cobra import get_cobra, get_cobraII
import argparse

from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml

def load_patch_feats(h5_path, device):
    """
    Load patch features from an HDF5 file.
    Args:
        h5_path (str): Path to the HDF5 file containing patch features.
        device (str): Device to load the features onto (e.g., "cuda" or "cpu").
    Returns:                                
        feats (torch.Tensor): Loaded patch features as a PyTorch tensor.
        coords (np.ndarray): Coordinates associated with the patch features.
    """
    if not os.path.exists(h5_path):
        tqdm.write(f"File {h5_path} does not exist, skipping")
        return None
    with h5py.File(h5_path, "r") as f:
        feats = f["feats"][:]
        feats = torch.tensor(feats).to(device)
        coords = np.array(f["coords"][:])
    return feats, coords

def match_coords(feats_w, feats_a, coords_w, coords_a):
    """
    Match and extract features whose corresponding coordinates are identical in two sets.

    It uses np.intersect1d to compute the intersection (in sorted order)
    of the coordinate arrays, and returns the features accordingly.

    Parameters:
        feats_w (np.ndarray): Feature array for weighted patches.
        feats_a (np.ndarray): Feature array for auxiliary patches.
        coords_w (np.ndarray): Coordinates for feats_w with shape (N, D).
        coords_a (np.ndarray): Coordinates for feats_a with shape (M, D).

    Returns:
        tuple: (matched_feats_w, matched_feats_a) where the i-th entry in both arrays corresponds 
               to the same common coordinate.

    Raises:
        ValueError: If no common coordinates are found.
    """
    # Create structured views so entire rows can be compared as single elements.
    dt = np.dtype((np.void, coords_w.dtype.itemsize * coords_w.shape[1]))
    coords_w_view = np.ascontiguousarray(coords_w).view(dt).ravel()
    coords_a_view = np.ascontiguousarray(coords_a).view(dt).ravel()

    common, idx_w, idx_a = np.intersect1d(coords_w_view, coords_a_view, return_indices=True)
    if len(common) == 0:
        raise ValueError("No matching coordinates found")
    
    return feats_w[idx_w], feats_a[idx_a]

def get_cobra_feats(model, patch_feats_w, patch_feats_a, top_k=None):
    """
    Compute COBRA features by aggregating patch features using attention scores from the model.
    This function takes patch features and processes them with the given model to extract
    attention scores. If a top_k value is provided, it selects the top_k patches based on
    the attention scores, applies a softmax weighting over these scores, and computes a weighted
    sum of the corresponding patch features. Otherwise, it directly aggregates all patch
    features with the raw attention scores.
    Parameters:
        model (torch.nn.Module): The neural network model used to compute attention scores.
                                 It should accept patch_feats_w as input and support a "get_attention"
                                 keyword argument.
        patch_feats_w (torch.Tensor): The input patch features that are processed by the model to obtain
                                      attention scores.
        patch_feats_a (torch.Tensor): The patch features used for the final feature aggregation.
        top_k (int, optional): The number of top patches (based on attention score) to use for feature
                               aggregation.
    Returns:
        torch.Tensor: The aggregated COBRA features as a 1D tensor.
    Notes:
        - The function runs within torch.inference_mode() to disable gradient calculations.
        - When top_k is used, the function ensures that top_k does not exceed the total number of patches.
        - The attention weights are normalized using softmax before being used for aggregation.
    """
    with torch.inference_mode():
        A = model(patch_feats_w, get_attention=True)
        # A.shape: (1,1,num_patches)
        if top_k:
            if A.size(-1) < top_k:
                top_k = A.size(-1)
            top_k_indices = torch.topk(A, top_k, dim=-1).indices  # (1,1,top_k)
            top_k_A = A.gather(-1, top_k_indices)  # (1,1,top_k)
            top_k_x = patch_feats_a.gather(1, top_k_indices.squeeze(0).unsqueeze(-1).expand(-1, -1, patch_feats_a.size(-1)))
            # top_k_x.shape: (1,top_k,feat_dim)
            cobra_feats = torch.bmm(F.softmax(top_k_A, dim=-1), top_k_x).squeeze(1)
        else:
            cobra_feats = torch.bmm(A, patch_feats_a).squeeze(1)
    return cobra_feats.squeeze(0)

def save_chunk(
    data_dict,
    chunk_index,
    output_dir,
    output_file,
    model_name,
    top_k,
    dtype,
    weighting_fm,
    aggregation_fm,
    microns,
):
    """Helper: save one chunk of features to <output_file>_chunk<chunk_index>.h5"""
    if chunk_index > 0:
        if chunk_index == 1:
            os.rename(
                os.path.join(output_dir, output_file),
                os.path.join(output_dir, f"{output_file.split('.h5')[0]}_0.h5"),
            )
        chunk_file = os.path.join(
            output_dir, f"{output_file.split('.h5')[0]}_{chunk_index}.h5"
        )
    else:
        chunk_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
    with h5py.File(chunk_file, "w") as f:
        for key, data in data_dict.items():
            f.create_dataset(key, data=data["feats"])
        f.attrs.update({
            "extractor": model_name,
            "top_k": top_k if top_k else "None",
            "dtype": str(dtype),
            "weighting_FM": weighting_fm,
            "aggregation_FM": aggregation_fm,
            "microns": microns,
        })
    tqdm.write(f"Saved chunk {chunk_index} to {chunk_file}")

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
    chunk_size=None  # Enable chunking if provided
):
    """
    Extract patient-level features from slide-level feature files and save them into HDF5 files.
    Supports saving in chunks if `chunk_size` is provided.
    Uses match_coords only when auxiliary features are provided and weighting_fm != aggregation_fm.
    """
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby("PATIENT")
    pat_dict = {}

    # Determine if we need to run match_coords.
    do_match = (feat_dir_a is not None) and (weighting_fm != aggregation_fm)
    if do_match:
        print("Using match_coords for patient-level extraction (weighting_fm != aggregation_fm).")
    else:
        print("Skipping match_coords for patient-level extraction (using identical features or no auxiliary features).")

    chunk_index = 0
    chunk_patients = []

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
                feats_a, coords_a = feats_w, coords_w

            if feats_a is None:
                continue

            if do_match:
                try:
                    feats_w, feats_a = match_coords(feats_w, feats_a, coords_w, coords_a)
                except ValueError as e:
                    tqdm.write(f"Patient {patient_id}, slide {slide_filename}: {str(e)}")
                    continue

            all_feats_list_w.append(feats_w)
            all_feats_list_a.append(feats_a)

        if all_feats_list_w:
            all_feats_cat_w = torch.cat(all_feats_list_w, dim=0).unsqueeze(0)
            all_feats_cat_a = torch.cat(all_feats_list_a, dim=0).unsqueeze(0)
            assert all_feats_cat_w.ndim == 3, f"Expected 3D tensor, got {all_feats_cat_w.ndim}"
            assert all_feats_cat_a.ndim == 3, f"Expected 3D tensor, got {all_feats_cat_a.ndim}"
            assert all_feats_cat_w.shape[1] == all_feats_cat_a.shape[1], (
                f"Expected same number of tiles, got {all_feats_cat_w.shape[1]} and {all_feats_cat_a.shape[1]}"
            )
            patient_feats = get_cobra_feats(model, all_feats_cat_w.to(dtype), all_feats_cat_a.to(dtype), top_k=top_k)
            pat_dict[patient_id] = {
                "feats": patient_feats.to(torch.float32).detach().squeeze().cpu().numpy(),
            }
            chunk_patients.append(patient_id)

        if chunk_size and len(chunk_patients) >= chunk_size:
            save_chunk(
                pat_dict,
                chunk_index,
                output_dir,
                output_file,
                model_name,
                top_k,
                dtype,
                weighting_fm,
                aggregation_fm,
                microns,
            )
            pat_dict.clear()
            chunk_patients.clear()
            chunk_index += 1

    if pat_dict:  # final leftover
        save_chunk(
            pat_dict,
            chunk_index,
            output_dir,
            output_file,
            model_name,
            top_k,
            dtype,
            weighting_fm,
            aggregation_fm,
            microns,
        )

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
    chunk_size=None  # Enable chunking if provided
):
    """
    Generates slide-level features from tile embeddings and saves them in HDF5 files (in chunks if chunk_size is provided) 
    along with a metadata JSON file.
    Uses match_coords only when auxiliary features are provided and weighting_fm != aggregation_fm.
    """
    slide_dict = {}
    chunk_index = 0
    chunk_slides = []

    tile_emb_paths_w = glob(f"{feat_dir_w}/**/*.h5", recursive=True)
    if feat_dir_a is not None:
        tile_emb_paths_a = glob(f"{feat_dir_a}/**/*.h5", recursive=True)
    else:
        tile_emb_paths_a = tile_emb_paths_w

    assert len(tile_emb_paths_w) == len(tile_emb_paths_a), (
        f"Expected same number of files, got {len(tile_emb_paths_w)} and {len(tile_emb_paths_a)}"
    )

    # Determine if we need to run match_coords.
    do_match = (feat_dir_a is not None) and (weighting_fm != aggregation_fm)
    if do_match:
        print("Using match_coords for slide-level extraction (weighting_fm != aggregation_fm).")
    else:
        print("Skipping match_coords for slide-level extraction (using identical features or no auxiliary features).")

    for tile_emb_path_w, tile_emb_path_a in zip(tqdm(tile_emb_paths_w), tile_emb_paths_a):
        slide_name = Path(tile_emb_path_w).stem
        feats_w, coords_w = load_patch_feats(tile_emb_path_w, device)
        if feats_w is None:
            continue
        if feat_dir_a:
            # Assume auxiliary file has the same base name in feat_dir_a.
            tile_emb_path_a = os.path.join(feat_dir_a, f"{slide_name}.h5")
            feats_a, coords_a = load_patch_feats(tile_emb_path_a, device)
        else:
            feats_a, coords_a = feats_w, coords_w
        if feats_a is None:
            continue

        if do_match:
            try:
                feats_w, feats_a = match_coords(feats_w, feats_a, coords_w, coords_a)
            except ValueError as e:
                tqdm.write(f"Slide {slide_name}: {str(e)}")
                continue

        # For slide embeddings, we assume the entire tensor is used.
        tile_embs_w = feats_w.unsqueeze(0)
        tile_embs_a = feats_a.unsqueeze(0)
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
        chunk_slides.append(slide_name)

        if chunk_size and len(chunk_slides) >= chunk_size:
            save_chunk(
                slide_dict,
                chunk_index,
                output_dir,
                output_file,
                model_name,
                top_k,
                dtype,
                weighting_fm,
                aggregation_fm,
                microns,
            )
            slide_dict.clear()
            chunk_slides.clear()
            chunk_index += 1

    # Save any remaining slides (if not an exact multiple of chunk_size)
    if slide_dict:
        if chunk_size:
            save_chunk(
                slide_dict,
                chunk_index,
                output_dir,
                output_file,
                model_name,
                top_k,
                dtype,
                weighting_fm,
                aggregation_fm,
                microns,
            )
            tqdm.write(f"Finished extraction, saved chunk {chunk_index} to {output_dir}")
        else:
            output_path = os.path.join(output_dir, output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with h5py.File(output_path, "w") as f:
                for slide_name, data in slide_dict.items():
                    f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["extractor"] = model_name
                f.attrs["top_k"] = top_k if top_k else "None"
                f.attrs["dtype"] = str(dtype)
                f.attrs["weighting_FM"] = weighting_fm
                f.attrs["aggregation_FM"] = aggregation_fm
                f.attrs["microns"] = microns
            tqdm.write(f"Finished extraction, saved to {output_path}")

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
    """
    Main function for extracting slide or patient embeddings using the COBRA model.

    This function parses command-line arguments (and optionally a YAML configuration file) to set up the
    extraction parameters. It supports two modes:
     - Patient-level embeddings (if a slide table is provided)
     - Slide-level embeddings (if not)

    Command-line arguments include:
        -c, --config (str): Optional path to a YAML configuration file.
        -d, --download_model: Flag to download model weights.
        -w, --checkpoint_path (str): Path to the model checkpoint.
        -k, --top_k (int): Top k tiles to use for embedding aggregation.
        -o, --output_dir (str): Directory to save extracted features.
        -f, --feat_dir (str): Directory containing primary tile feature files.
        -g, --feat_dir_a (str): Directory containing auxiliary tile feature files.
        -m, --model_name (str): Model name (default: "COBRAII").
        -p, --patch_encoder (str): Patch encoder name (default: "Virchow2").
        -a, --patch_encoder_a (str): Patch encoder name used for aggregation (default: "Virchow2").
        -e, --h5_name (str): Output HDF5 file name (default: "cobra_feats.h5").
        -r, --microns (int): Microns per patch (default: 224).
        -s, --slide_table (str): Slide table path for patient-level extraction.
        -u, --use_cobraI: Flag to use the COBRA I model (default uses COBRAII).
        -z, --chunk_size (int): Maximum number of slides or patients per HDF5 file (optional).
    """
    parser = argparse.ArgumentParser(
        description="Extract slide/patient embeddings using COBRA model"
    )
    parser.add_argument("-c", "--config", type=str,
                        help="Path to a YAML configuration file", default=None)
    parser.add_argument("-d", "--download_model", action="store_true",
                        help="Flag to download model weights")
    parser.add_argument("-w", "--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("-k", "--top_k", type=int, required=False, default=None,
                        help="Top k tiles to use for slide/patient embedding")
    parser.add_argument("-o", "--output_dir", type=str, required=False,
                        help="Directory to save extracted features")
    parser.add_argument("-f", "--feat_dir", type=str, required=False,
                        help="Directory containing tile feature files")
    parser.add_argument("-g", "--feat_dir_a", type=str, required=False, default=None,
                        help="Directory containing tile feature files for aggregation")
    parser.add_argument("-m", "--model_name", type=str, required=False, default="COBRAII",
                        help="Model name")
    parser.add_argument("-p", "--patch_encoder", type=str, required=False, default="Virchow2",
                        help="Patch encoder name")
    parser.add_argument("-a", "--patch_encoder_a", type=str, required=False, default="Virchow2",
                        help="Patch encoder name used for aggregation")
    parser.add_argument("-e", "--h5_name", type=str, required=False, default="cobra_feats.h5",
                        help="Output HDF5 file name")
    parser.add_argument("-r", "--microns", type=int, required=False, default=224,
                        help="Microns per patch used for extraction")
    parser.add_argument("-s", "--slide_table", type=str, required=False,
                        help="Slide table path (for patient-level extraction)")
    parser.add_argument("-u", "--use_cobraI", action="store_true",
                        help="Whether to use COBRA I (if not set, use COBRAII)")
    parser.add_argument("-z", "--chunk_size", type=int, required=False, default=None,
                        help="Maximum number of slides or patients per HDF5 file (for chunked saving)")
    args = parser.parse_args()
    
    # If a config file is provided, load parameters from the config file.
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = config.get("extract_feats", {})
        args.download_model = config.get("download_model", args.download_model)
        args.checkpoint_path = config.get("checkpoint_path", args.checkpoint_path)
        args.top_k = config.get("top_k", args.top_k)
        args.output_dir = config.get("output_dir", args.output_dir)
        args.feat_dir = config.get("feat_dir", args.feat_dir)
        args.feat_dir_a = config.get("feat_dir_a", args.feat_dir_a)
        args.model_name = config.get("model_name", args.model_name)
        args.patch_encoder = config.get("patch_encoder", args.patch_encoder)
        args.patch_encoder_a = config.get("patch_encoder_a", args.patch_encoder_a)
        args.h5_name = config.get("h5_name", args.h5_name)
        args.microns = config.get("microns", args.microns)
        args.slide_table = config.get("slide_table", args.slide_table)
        args.use_cobraI = config.get("use_cobraI", args.use_cobraI)
        args.chunk_size = config.get("chunk_size", args.chunk_size)
    
    print(f"Using configuration: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cobra_func = get_cobra if args.use_cobraI else get_cobraII
    if args.checkpoint_path:
        model = cobra_func(
                download_weights=(not os.path.exists(args.checkpoint_path)),
                checkpoint_path=args.checkpoint_path,
            )
    else:
        print("No checkpoint path provided. Downloading model weights...")
        model = cobra_func(
            download_weights=True,
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
        # Patient-level embeddings
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
            chunk_size=args.chunk_size,      
        )
    else:
        # Slide-level embeddings
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
            chunk_size=args.chunk_size,     
        )
        
if __name__ == "__main__":
    main()
