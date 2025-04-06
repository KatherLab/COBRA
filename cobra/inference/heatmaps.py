import torch 
import pandas as pd
import os
from os.path import exists
from tqdm import tqdm 
import argparse
import h5py
import json
import numpy as np
import zipfile
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import openslide

from cobra.utils.load_cobra import get_cobraII
from cobra.utils.get_mpp import get_slide_mpp_

#TODO: too much space around the subimages

def get_slide_thumbnail(slide,dims_um,attention_shape,heat_map_scale_factor=8):
    
    #print(f"slide dimensions: {slide.dimensions}, mpp: {mpp}, dims_um: {dims_um}")
    thumb = slide.get_thumbnail(tuple(dims_um)*heat_map_scale_factor) 
    thumb = np.array(thumb).transpose(1, 0, 2)  
    if (np.array(attention_shape)*heat_map_scale_factor <= np.array(thumb.shape[:2])).all():
        res_thumb = thumb[: attention_shape[0] * heat_map_scale_factor,: attention_shape[1] * heat_map_scale_factor]
    else:
        print(f"Attention shape (scaled): {np.array(attention_shape) * heat_map_scale_factor}, Thumbnail shape: {thumb.shape[:2]}")
        res_thumb = np.zeros((max(attention_shape[0] * heat_map_scale_factor, thumb.shape[0]), max(attention_shape[1] * heat_map_scale_factor, thumb.shape[1]), 3), dtype=np.uint8)
        res_thumb[:thumb.shape[0], :thumb.shape[1]] = thumb

    # Dynamically adjust res_thumb to match the expected shape
    expected_shape = np.array(attention_shape) * heat_map_scale_factor
    res_thumb = res_thumb[:expected_shape[0], :expected_shape[1]]

    assert (np.array(res_thumb.shape[:2]) == expected_shape).all(), f"Thumbnail shape {res_thumb.shape} does not match attention shape {expected_shape}"
    return res_thumb


def load_patch_features(feat_path,device="cuda"):
    """
    Load patch features from the specified directory.
    """

    with h5py.File(feat_path, "r") as f:
        feats = torch.tensor(f["feats"][:]).to(device)
        coords = torch.tensor(f["coords"][:]).to(device)
    return feats, coords
    

def create_heatmap(model, slide_name, wsi_path, feat_path, output_dir, microns_per_patch=112,patch_size=224,scale_factor=8, device="cuda"):
    """
    Create a heatmap for the given slide using the specified model and save it to the output directory.
    """

    # Load patch features
    feats, coords = load_patch_features(feat_path,device=device)

    with torch.inference_mode():
        attention = model(feats.to(torch.float32), get_attention=True).squeeze().cpu().numpy()
    coords = np.round(coords.cpu().numpy()/(microns_per_patch / patch_size)).astype(np.int32)
    xs = np.unique(sorted(coords[:, 0]))
    stride = min(xs[1:] - xs[:-1])

    coords_norm = coords // stride
    patch_feat_mpp=(microns_per_patch/patch_size)
    slide = openslide.open_slide(wsi_path)
    mpp = get_slide_mpp_(slide, default_mpp=None)
    dims_um = np.round(np.array(slide.dimensions)*mpp/(patch_feat_mpp * patch_size)).astype(np.int32)

    #im = np.zeros(coords_norm.max(0) + 1)
    im = np.zeros((dims_um[1], dims_um[0], 4), dtype=np.float32)

    for att, pos in zip(attention / attention.max(), coords_norm, strict=True):
        im[*pos] = att
    foreground = im > 0
    im = plt.get_cmap("viridis")(im)
    im[..., -1] = foreground

    heatmap_im = Image.fromarray(np.uint8(im * 255)).resize(
        np.array(im.shape[:2][::-1]) * 8, Image.Resampling.NEAREST
    )
    slide_im = Image.fromarray(get_slide_thumbnail(wsi_path, im.shape[:2], heat_map_scale_factor=scale_factor))#.resize(
    #    np.array(im.shape[:2][::-1]) * 8, Image.Resampling.NEAREST
    #)

    if not (np.array(slide_im.size) == np.array(heatmap_im.size)).all():
        print(f"Slide image size: {slide_im.size}, Heatmap image size: {heatmap_im.size}")
        raise ValueError("Slide image size is not equal to heatmap image size.")
    
    # Convert heatmap_im and slide_im to numpy arrays
    heatmap_array = np.array(heatmap_im)
    slide_array = np.array(slide_im)

    # Dynamically adjust the figure size based on the aspect ratios of the images
    slide_aspect_ratio = slide_array.shape[1] / slide_array.shape[0]
    heatmap_aspect_ratio = heatmap_array.shape[1] / heatmap_array.shape[0]

    # Calculate the width ratios for the subplots
    width_ratios = [slide_aspect_ratio, heatmap_aspect_ratio]

    # Set the figure size dynamically based on the combined aspect ratios
    fig_width = 10  # Base width
    fig_height = fig_width / (sum(width_ratios) / len(width_ratios)) / 2  # Adjust height to maintain aspect ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': width_ratios})

    # Display slide image
    ax1.imshow(slide_array)
    ax1.axis('off')

    # Add annotation for scale bar on slide image
    scale_bar_length = 2000 / microns_per_patch * scale_factor  # Length of the scale bar in pixels 2000 um = 2 mm
    scale_bar_text = '2 mm'  # Text for the scale bar
    ax1.annotate('', xy=(10, slide_array.shape[0] - 20), xytext=(10 + scale_bar_length, slide_array.shape[0] - 20),
                 arrowprops=dict(arrowstyle='-', color='black', lw=2))
    ax1.text(10 + scale_bar_length / 2, slide_array.shape[0] - 30, scale_bar_text, color='black', ha='center')

    # Display heatmap image with colorbar
    cax = ax2.imshow(heatmap_array, cmap='viridis')
    ax2.axis('off')
    cbar = fig.colorbar(cax, ax=ax2, orientation='vertical', fraction=0.05, pad=0.05)

    plt.tight_layout()
    # Save the combined figure as a PDF
    print(f"Saving heatmap and slide images for {slide_name}...")
    plt.savefig(os.path.join(output_dir, f"{slide_name}.pdf"), dpi=300)

    

def main(device="cuda"):
    """
    Main function to extract top tiles using the COBRA model.
    """
    parser = argparse.ArgumentParser(description="Extract slide/pat top tiles using COBRA model")
    # parser.add_argument("-p", "--patch_encoder", type=str, required=False, default="Virchow2", help="patch encoder name")
    # parser.add_argument("-a", "--patch_encoder_a", type=str, required=False, default="ConchV1-5", help="patch encoder name used for aggregation")
    parser.add_argument("-f", "--feat_dir", type=str, default="/p/scratch/mfmpm/tim/data/junhao-data/LEEDS-FOCUS-CRC/features-virchow2/virchow2-341bdca9/FOCUS3-CRC-IMGS/04-05Aug2010", 
                        required=False, help="Directory containing tile feature files")
    parser.add_argument("-s", "--wsi_dir", type=str, default="/p/scratch/mfmpm/tim/data/junhao-data/LEEDS-FOCUS-CRC/FOCUS3-CRC-IMGS/04-05Aug2010", required=False, help="Directory containing WSI files")
    parser.add_argument("-w", "--checkpoint_path", type=str, default="/p/scratch/mfmpm/tim/code/COBRA/weights/cobraII.pth.tar", help="Path to model checkpoint")
    parser.add_argument("-r", "--microns", type=int, required=False, default=112, help="microns per patch used for extraction")
    parser.add_argument("-p", "--patch_size", type=int, required=False, default=224, help="Patch size used for extraction")
    parser.add_argument("-o", "--output_dir", type=str, required=False, default="/p/scratch/mfmpm/tim/data/cobra-heatmap-test-comb2", help="Directory to save top tiles in")

    args = parser.parse_args()
    print(f"Arguments: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_cobraII(download_weights=(not exists(args.checkpoint_path)), checkpoint_path=args.checkpoint_path)
    model.eval()
    model.to(device)
    for wsi in tqdm(os.listdir(args.wsi_dir)):
        wsi_path = os.path.join(args.wsi_dir, wsi)
        slide_name = os.path.splitext(wsi)[0]
        feat_path = os.path.join(args.feat_dir, slide_name + ".h5")
        if not exists(feat_path):
            tqdm.write(f"Feature file {feat_path} does not exist. Skipping...")
            continue
        if not exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        create_heatmap(model, slide_name, wsi_path, feat_path, args.output_dir, microns_per_patch=args.microns, patch_size=args.patch_size, scale_factor=8, device=device)

if __name__ == "__main__":
    main()