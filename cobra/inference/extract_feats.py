import os
import torch
import h5py
from tqdm import tqdm
from glob import glob
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from cobra.utils.load_cobra import get_cobra, get_cobraII
from cobra.model.cobra import Cobra
import argparse
import yaml
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd
warnings.simplefilter(action="ignore", category=FutureWarning)

def get_cobra_feats(model,patch_feats,dtype,top_k=None):
    with torch.inference_mode():
        A = model(patch_feats.to(dtype),get_attention=True)
        if top_k:
            top_k_indices = torch.topk(A, top_k, dim=1).indices
            top_k_A = A.gather(1, top_k_indices)
            top_k_x = patch_feats.gather(1, top_k_indices.expand(-1, -1, patch_feats.size(-1)))
            cobra_feats = torch.bmm(top_k_A, top_k_x).squeeze(1)
        else:
            cobra_feats = torch.bmm(A, patch_feats).squeeze(1)
    return cobra_feats
    
def get_pat_embs(
    model,
    output_dir,
    feat_dir,
    output_file="cobra-feats.h5",
    model_name="COBRAII",
    slide_table_path=None,
    device="cuda",
    dtype=torch.float32,
    top_k=None,
    pe_name="Virchow2",
):
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby("PATIENT")
    pat_dict = {}

    output_file = os.path.join(output_dir, output_file)

    if os.path.exists(output_file):
        tqdm.write(f"Output file {output_file} already exists, skipping")
        return

    for patient_id, group in tqdm(patient_groups, leave=False):
        all_feats_list = []

        for _, row in group.iterrows():
            slide_filename = row["FILENAME"]
            h5_path = os.path.join(feat_dir, slide_filename)
            if not os.path.exists(h5_path):
                tqdm.write(f"File {h5_path} does not exist, skipping")
                continue
            with h5py.File(h5_path, "r") as f:
                feats = f["feats"][:]

            feats = torch.tensor(feats).to(device)
            all_feats_list.append(feats)

        if all_feats_list:
            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)
            #with torch.inference_mode():
            assert all_feats_cat.ndim == 3, (
                f"Expected 3D tensor, got {all_feats_cat.ndim}"
            )
            patient_feats = get_cobra_feats(model,all_feats_cat.to(dtype),dtype,top_k=top_k)
            pat_dict[patient_id] = {
                "feats": patient_feats.to(torch.float32)
                .detach()
                .squeeze()
                .cpu()
                .numpy(),
                "extractor": f"{model_name}-{pe_name}",
            }
        else:
            tqdm.write(f"No features found for patient {patient_id}, skipping")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for patient_id, data in slide_dict.items():
            f.create_dataset(f"{patient_id}", data=data["feats"])
        f.attrs["extractor"] = data["extractor"]
        f.attrs["top_k"] = top_k
        f.attrs["dtype"] = str(dtype)
        tqdm.write(f"Finished extraction, saved to {output_file}")


def get_slide_embs(
    model,
    output_dir,
    feat_dir,
    output_file="cobra-feats.h5",
    model_name="COBRAII",
    device="cuda",
    dtype=torch.float32,
    top_k=None,
    pe_name="Virchow2",
):
    slide_dict = {}

    tile_emb_paths = glob(f"{feat_dir}/*.h5")
    for tile_emb_path in tqdm(tile_emb_paths):
        with h5py.File(h5_path, "r") as f:
                feats = f["feats"][:]

        tile_embs = torch.tensor(feats).to(device).unsqueeze(0)
        slide_name = Path(tile_emb_path).stem
        #with torch.inference_mode():
        assert tile_embs.ndim == 3, f"Expected 3D tensor, got {tile_embs.ndim}"
        slide_feats = get_cobra_feats(model, tile_embs.to(dtype), dtype, top_k=top_k)
        slide_dict[slide_name] = {
            "feats": slide_feats.to(torch.float32).detach().squeeze().cpu().numpy(),
            "extractor": f"{model_name}-{pe_name}",
        }
    output_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for slide_name, data in slide_dict.items():
            f.create_dataset(f"{slide_name}", data=data["feats"])
        f.attrs["extractor"] = data["extractor"]
        f.attrs["top_k"] = top_k
        f.attrs["dtype"] = str(dtype)
        tqdm.write(f"Finished extraction, saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract slide embeddings using COBRA model"
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
        "-e",
        "--h5_name",
        type=str,
        required=False,
        default="cobra_feats.h5",
        help="File name",
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
            args.h5_name,
            args.model_name,
            args.slide_table,
            device,
            dtype=dtype,
        )
    else:
        get_slide_embs(
            model,
            args.output_dir,
            args.feat_dir,
            args.h5_name,
            args.model_name,
            device=device,
            dtype=dtype,
        )
        
if __name__ == "__main__":
    main()
