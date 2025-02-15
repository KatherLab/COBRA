warnings.simplefilter(action="ignore", category=FutureWarning)
def get_pat_embs(
    model,
    output_dir,
    feat_dir,
    output_file="cobra-feats.h5",
    model_name="COBRAII",
    slide_table_path=None,
    device="cuda",
    dtype=torch.float32,
):
    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby("PATIENT")
    slide_dict = {}

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
            with torch.inference_mode():
                assert all_feats_cat.ndim == 3, (
                    f"Expected 3D tensor, got {all_feats_cat.ndim}"
                )
                slide_feats = model(all_feats_cat.to(dtype))
                slide_dict[patient_id] = {
                    "feats": slide_feats.to(torch.float32)
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy(),
                    "extractor": f"{model_name}-V2",
                }
        else:
            tqdm.write(f"No features found for patient {patient_id}, skipping")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for patient_id, data in slide_dict.items():
            f.create_dataset(f"{patient_id}", data=data["feats"])
            f.attrs["extractor"] = data["extractor"]
        tqdm.write(f"Finished extraction, saved to {output_file}")


def get_slide_embs(
    model,
    output_dir,
    feat_dir,
    output_file="cobra-feats.h5",
    model_name="COBRAII",
    device="cuda",
    dtype=torch.float32,
):
    slide_dict = {}

    tile_emb_paths = glob(f"{feat_dir}/*.h5")
    for tile_emb_path in tqdm(tile_emb_paths):
        with h5py.File(h5_path, "r") as f:
                feats = f["feats"][:]

        tile_embs = torch.tensor(feats).to(device).unsqueeze(0)
        slide_name = Path(tile_emb_path).stem
        with torch.inference_mode():
            assert tile_embs.ndim == 3, f"Expected 3D tensor, got {tile_embs.ndim}"
            slide_feats = model(tile_embs.to(dtype))
            slide_dict[slide_name] = {
                "feats": slide_feats.to(torch.float32).detach().squeeze().cpu().numpy(),
                "extractor": f"{model_name}",
            }
    output_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for slide_name, data in slide_dict.items():
            f.create_dataset(f"{slide_name}", data=data["feats"])
            f.attrs["extractor"] = data["extractor"]
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
    # parser.add_argument(
    #     "-c", "--config", type=str, default=None, help="Path to model config"
    # )
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
        "-u", "--use_cobraI", action="store_true", help="wheter to use cobra I or II"
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.download_model:
        if args.use_cobraI:
            model = get_cobra(
                download_weights=args.download_model,
                checkpoint_path=args.checkpoint_path,
            )
        else: