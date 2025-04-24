import torch 
import pandas as pd
import os
from os.path import exists
from tqdm import tqdm 
import argparse
import h5py
import numpy as np
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import openslide
import yaml

from cobra.utils.load_cobra import get_cobraII
from cobra.utils.get_mpp import get_slide_mpp_

def get_slide_thumbnail(slide, heatmap_shape, heat_map_scale_factor=8):
    """
    Generate a thumbnail of the slide with the specified heatmap shape and scale factor.

    Args:
        slide (OpenSlide object): The whole slide image object.
        heatmap_shape (tuple): Shape of the heatmap.
        heat_map_scale_factor (int): Scale factor for the thumbnail.

    Returns:
        np.ndarray: Thumbnail image as a NumPy array.
    """
    thumb = slide.get_thumbnail(heatmap_shape * heat_map_scale_factor)
    thumb = np.array(thumb).transpose(1, 0, 2)  
    return thumb


def load_patch_features(feat_path, device="cuda"):
    """
    Load patch features and coordinates from the specified HDF5 file.

    Args:
        feat_path (str): Path to the HDF5 file containing patch features.
        device (str): Device to load the features onto (e.g., "cuda" or "cpu").

    Returns:
        tuple: A tuple containing patch features (torch.Tensor) and coordinates (torch.Tensor).
    """
    with h5py.File(feat_path, "r") as f:
        feats = torch.tensor(f["feats"][:]).to(device)
        coords = torch.tensor(f["coords"][:]).to(device)
    return feats, coords
    

def create_heatmap(model, slide_name, wsi_path, feat_path, output_dir, microns_per_patch=112,
     patch_size=224, scale_factor=8, device="cuda" , stamp_v=1):
    """
    Create a heatmap for the given slide using the specified model and save it to the output directory.

    Args:
        model (torch.nn.Module): The trained COBRA model.
        slide_name (str): Name of the slide.
        wsi_path (str): Path to the whole slide image (WSI) file.
        feat_path (str): Path to the HDF5 file containing patch features.
        output_dir (str): Directory to save the generated heatmap.
        microns_per_patch (int): Microns per patch used for extraction.
        patch_size (int): Size of each patch in pixels.
        scale_factor (int): Scale factor for resizing the heatmap.
        device (str): Device to perform computations on (e.g., "cuda" or "cpu").
    """
    # Load patch features
    feats, coords = load_patch_features(feat_path, device=device)
    patch_feat_mpp = (microns_per_patch / patch_size)
    with torch.inference_mode():
        attention = model(feats.to(torch.float32), get_attention=True).squeeze().cpu().numpy()
    if stamp_v==2:
        coords = np.floor(coords.cpu().numpy() / patch_feat_mpp).astype(np.int32)
    else:
        coords = coords.cpu().numpy().astype(np.int32)
    xs = np.unique(sorted(coords[:, 0]))
    stride = min(xs[1:] - xs[:-1])

    coords_norm = coords // stride
    
    slide = openslide.open_slide(wsi_path)
    mpp = get_slide_mpp_(slide, default_mpp=None)
    dims_um = np.ceil(np.array(slide.dimensions) * mpp / (patch_feat_mpp * patch_size)).astype(np.int32)
    if not np.all(coords_norm.max(0) <= dims_um):
        tqdm.write(f"Warning: Coordinates exceed slide dimensions. Trying to flip axes...")
        coords_norm = coords_norm[:, ::-1]
    im = np.zeros((dims_um[0], dims_um[1]), dtype=np.float32)

    for att, pos in zip(attention / attention.max(), coords_norm, strict=True):
        im[*pos] = att
    foreground = im > 0
    im = plt.get_cmap("viridis")(im)
    im[..., -1] = foreground

    heatmap_im = Image.fromarray(np.uint8(im * 255)).resize(
        np.array(im.shape[:2][::-1]) * 8, Image.Resampling.NEAREST
    )
    slide_im = Image.fromarray(get_slide_thumbnail(slide, np.array(im.shape[:2]), heat_map_scale_factor=scale_factor))

    # Convert heatmap and slide images to NumPy arrays
    heatmap_array = np.array(heatmap_im)
    slide_array = np.array(slide_im)

    # Dynamically adjust figure size based on aspect ratios
    slide_aspect_ratio = slide_array.shape[1] / slide_array.shape[0]
    heatmap_aspect_ratio = heatmap_array.shape[1] / heatmap_array.shape[0]
    width_ratios = [slide_aspect_ratio, heatmap_aspect_ratio]
    fig_width = 10
    fig_height = fig_width / (sum(width_ratios) / len(width_ratios)) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': width_ratios})

    # Display slide image
    ax1.imshow(slide_array)
    ax1.axis('off')

    # Add scale bar annotation
    scale_bar_length = 2000 / microns_per_patch * scale_factor # 2000 microns = 2 mm
    scale_bar_text = '2 mm'
    ax1.annotate('', xy=(10, slide_array.shape[0] - 20), xytext=(10 + scale_bar_length, slide_array.shape[0] - 20),
                 arrowprops=dict(arrowstyle='-', color='black', lw=2))
    ax1.text(10 + scale_bar_length / 2, slide_array.shape[0] - 30, scale_bar_text, color='black', ha='center')

    # Display heatmap image with colorbar
    cax = ax2.imshow(heatmap_array, cmap='viridis')
    ax2.axis('off')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cax, cax=cbar_ax, orientation='vertical')

    # Save the combined figure as a PDF
    print(f"Saving heatmap and slide images for {slide_name}...")
    plt.savefig(os.path.join(output_dir, f"{slide_name}.pdf"), dpi=300)

    

def main(device="cuda"):
    """
    Main function to generate heatmaps for whole slide images using the COBRA model.

    Args:
        device (str): Device to perform computations on (e.g., "cuda" or "cpu").
    """
    parser = argparse.ArgumentParser(
        description="Generate heatmaps for whole slide images using the COBRA model."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to configuration YAML file", default=None)
    # Commandline arguments (will be overridden if --config is provided)
    parser.add_argument("-f", "--feat_dir", type=str, default="/path/to/features", 
                        help="Directory containing tile feature files.")
    parser.add_argument("-s", "--wsi_dir", type=str, default="/path/to/wsi", 
                        help="Directory containing WSI files.")
    parser.add_argument("-w", "--checkpoint_path", type=str, default="/path/to/checkpoint.pth.tar", 
                        help="Path to the model checkpoint.")
    parser.add_argument("-r", "--microns", type=int, default=112, 
                        help="Microns per patch used for extraction.")
    parser.add_argument("-p", "--patch_size", type=int, default=224, 
                        help="Patch size used for extraction.")
    parser.add_argument("-o", "--output_dir", type=str, default="/path/to/output", 
                        help="Directory to save the generated heatmaps.")
    parser.add_argument("-v", "--stamp_version", type=int, default=2, 
                        help="Stamp version that was used for extraction.")
    
    args = parser.parse_args()
    
    # If a config file is provided, load and override the defaults
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        config = config.get("heatmap", {})
        args.feat_dir = config.get("feat_dir", args.feat_dir)
        args.wsi_dir = config.get("wsi_dir", args.wsi_dir)
        args.checkpoint_path = config.get("checkpoint_path", args.checkpoint_path)
        args.microns = config.get("microns", args.microns)
        args.patch_size = config.get("patch_size", args.patch_size)
        args.output_dir = config.get("output_dir", args.output_dir)
        args.stamp_version = config.get("stamp_version", args.stamp_version)
    
    print(f"Using configuration: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_cobraII(download_weights=(not exists(args.checkpoint_path)), checkpoint_path=args.checkpoint_path)
    model.eval()
    model.to(device)
    for wsi in tqdm([f for f in os.listdir(args.wsi_dir) if os.path.isfile(os.path.join(args.wsi_dir, f))]):
        wsi_path = os.path.join(args.wsi_dir, wsi)
        slide_name = os.path.splitext(wsi)[0]
        feat_path = os.path.join(args.feat_dir, slide_name + ".h5")
        if not exists(feat_path):
            tqdm.write(f"Feature file {feat_path} does not exist. Skipping...")
            continue
        if not exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        create_heatmap(model, slide_name, wsi_path, feat_path, args.output_dir,
                       microns_per_patch=args.microns,
                       patch_size=args.patch_size,
                       scale_factor=8,
                       device=device,
                       stamp_v=args.stamp_version)

if __name__ == "__main__":
    main()