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

def load_patch_feats(h5_path,device):
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

def match_coords(feats_w,feats_a,coords_w,coords_a):
    """
    Match and extract features whose corresponding coordinates are identical in two sets.

    This function sorts the given coordinate arrays (and their associated feature arrays) in lexicographical order.
    It then finds the indices where coordinates in one set also exist in the other, and returns the corresponding
    features from both sets. Ensure that both arrays have matching features for the overlapping coordinates; otherwise,
    an AssertionError is raised.

    Parameters:
        feats_w (np.ndarray): Array of features corresponding to the first set of coordinates.
        feats_a (np.ndarray): Array of features corresponding to the second set of coordinates.
        coords_w (np.ndarray): Array of coordinates for feats_w with shape (N, 2) where each row is [x, y].
        coords_a (np.ndarray): Array of coordinates for feats_a with shape (M, 2) where each row is [x, y].

    Returns:
        tuple: A tuple (matched_feats_w, matched_feats_a) where:
            matched_feats_w (np.ndarray): Features from the first set corresponding to coordinates found in both arrays.
            matched_feats_a (np.ndarray): Features from the second set corresponding to coordinates found in both arrays.

    Raises:
        AssertionError: If no matching coordinates are found or if the number of matches in both sets does not match.
    """
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
                               aggregation. If specified, only the top_k patches are aggregated; if not,
                               all patches are used.
    Returns:
        torch.Tensor: The aggregated COBRA features as a 1D tensor (after squeezing the unnecessary dimensions).
    Notes:
        - The function runs within torch.inference_mode() to disable gradient calculations.
        - When top_k is used, the function ensures that top_k does not exceed the total number of patches.
        - The attention weights are normalized using softmax before being used for aggregation.
    """
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
    """
    Extract patient-level features from slide-level feature files and save them into an HDF5 file.
    This function reads a slide table CSV file that contains slide information grouped by patient.
    For each patient, it iterates over all associated slide entries, loads the corresponding features
    from the specified directories (one for weighting features and optionally one for an alternative set),
    matches coordinates between the two sets of features, and aggregates them. The extracted and
    aggregated features are then passed through a provided model to obtain patient-level embeddings.
    Finally, the function saves the computed features and metadata (including extractor settings and parameters)
    to an HDF5 file and a JSON metadata file in the output directory.
    Note:
        - The function skips processing if the output file already exists and its size is greater than 800 bytes.
        - If feature extraction for a particular slide fails (i.e., features are None), that slide is skipped.
        - The function asserts that the concatenated feature tensors have the expected dimensions before proceeding.
    Parameters:
        model: The model used to compute the final patient-level features.
        output_dir (str): Directory where the output files (HDF5 and metadata JSON) will be saved.
        feat_dir_w (str): Directory containing the weighting feature H5 files for each slide.
        feat_dir_a (str, optional): Directory containing the alternative feature H5 files. If None, weighting features are used.
        output_file (str, optional): Name of the output HDF5 file. Default is "cobra-feats.h5".
        model_name (str, optional): Name of the model/extractor used. Default is "COBRAII".
        slide_table_path (str): Path to the CSV file containing slide information, including patient IDs and filenames.
        device (str, optional): Device to use for computation (e.g., "cuda" or "cpu"). Default is "cuda".
        dtype (torch.dtype, optional): Torch data type to which features are cast. Default is torch.float32.
        top_k (int, optional): If provided, only the top_k features will be selected during feature aggregation.
        weighting_fm (str, optional): Descriptor for the weighting feature method used. Default is "Virchow2".
        aggregation_fm (str, optional): Descriptor for the aggregation feature method used. Default is "Virchow2".
        microns (int, optional): Scale parameter indicating the micron size of the features. Default is 224.
    Behavior:
        - Loads the slide table and groups slides by patient.
        - Iterates over each patient, loading and optionally matching features from the provided directories.
        - Aggregates slide features into a 3D tensor and computes patient-level features using the model.
        - Saves the results in a specified HDF5 file under the given output directory.
        - Writes metadata about the extraction process to a metadata.json file.
    Returns:
        None
    """
    
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
    """
    Generates slide-level features from tile embeddings and saves them to an HDF5 file along with metadata.
    This function processes two sets of tile embeddings stored in separate directories (or a single directory if an alternative is not provided),
    computes slide-level features by aggregating weighted and auxiliary tile-level embeddings using a given model, and saves the results in an HDF5
    file along with a corresponding metadata JSON file.
    Parameters:
        model: 
            The feature extraction model used to aggregate tile embeddings into slide-level features.
        output_dir (str):
            The directory where the output files (HDF5 file and metadata JSON) will be saved.
        feat_dir_w (str):
            The directory containing HDF5 files with the primary (weighted) tile embeddings.
        feat_dir_a (str, optional):
            The directory containing HDF5 files with auxiliary tile embeddings. If not provided, tile_emb_paths_a will be fetched from feat_dir_w.
        output_file (str, optional):
            The name of the output HDF5 file where aggregated slide features will be saved. Default is "cobra-feats.h5".
        model_name (str, optional):
            A string identifier for the model/extractor used; saved in metadata and dataset attributes. Default is "COBRAII".
        device (str, optional):
            The computational device (e.g., "cuda" or "cpu") on which tensors should be loaded and processed. Note that mamba requires a NVIDIA gpu to function.
        dtype (torch.dtype, optional):
            The data type to which the tensor embeddings are cast prior to feature aggregation. Default is torch.float32.
        top_k (int, optional):
            An optional parameter to select the top-k features during aggregation. When None, no top-k filtering is applied.
        weighting_fm (str, optional):
            A string representing the weighting function or method used in the feature calculation. Default is "Virchow2".
        aggregation_fm (str, optional):
            A string representing the aggregation method used to compute slide-level features from tile embeddings. Default is "Virchow2".
        microns (int, optional):
            An integer parameter (e.g., representing physical scale/size) that is stored in the metadata. Default is 224.
    Behavior:
        - Searches for HDF5 files in the provided directories using a recursive glob pattern.
        - Loads tile-level features from each HDF5 file using an external function (load_patch_feats).
        - Ensures that corresponding weighted and auxiliary files contain the same number of tiles.
        - Applies the model to compute slide-level features by aggregating tile embeddings (using get_cobra_feats).
        - Creates and writes a dataset for each slide to the output HDF5 file, with attributes recording extraction details.
        - Saves a metadata JSON file containing extractor and processing parameters.
        - Outputs progress using tqdm.
    Raises:
        AssertionError:
            If the number of files in feat_dir_w and feat_dir_a (if provided) do not match, or if the shapes of the embeddings do not conform to expectations.
    """

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
    """
    Main function for extracting slide or patient embeddings using the COBRA model.
    This function parses command-line arguments, optionally loads configuration parameters
    from a YAML file, and sets up the model based on the provided arguments. It supports
    two modes of embedding extraction:
        - Patient-level embeddings: If a slide table is provided via the '--slide_table' argument.
        - Slide-level embeddings: If no slide table is provided.
    The function determines whether to download model weights or load a checkpoint based on
    the provided flags, selects the appropriate COBRA model function (COBRA I or COBRAII),
    and configures the model's device and data type (adjusting to mixed FP16 precision if the 
    GPU's compute capability is less than 8.0).
    Arguments:
            -c, --config (str): Optional path to a YAML configuration file to override command-line arguments.
            -d, --download_model: Flag indicating whether to download model weights.
            -w, --checkpoint_path (str): Path to the model checkpoint file.
            -k, --top_k (int): Optional top k tiles to use for slide/patient embedding.
            -o, --output_dir (str): Directory to save the extracted features (required).
            -f, --feat_dir (str): Directory containing tile feature files (required).
            -g, --feat_dir_a (str): Optional directory containing tile feature files for aggregation.
            -m, --model_name (str): Model name (default: "COBRAII").
            -p, --patch_encoder (str): Patch encoder name (default: "Virchow2").
            -a, --patch_encoder_a (str): Patch encoder name used for aggregation (default: "Virchow2").
            -e, --h5_name (str): Output HDF5 file name (default: "cobra_feats.h5").
            -r, --microns (int): Microns per patch used for extraction (default: 224).
            -s, --slide_table (str): Optional slide table path for patient-level extraction.
            -u, --use_cobraI: Flag to use the COBRA I model; if not set, COBRAII is used.
    Raises:
            FileNotFoundError: If a checkpoint path is provided but the file does not exist.
            ValueError: If neither a checkpoint path is provided nor the download_model flag is set.
    Returns:
            None
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
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save extracted features")
    parser.add_argument("-f", "--feat_dir", type=str, required=True,
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

    args = parser.parse_args()
    
    # If a config file is provided, load parameters from the config file
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
        # patient level embeddings
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
        # slide level embeddings
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
