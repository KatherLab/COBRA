import os
import yaml
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pickle
from cobra.crossval.train import MLP, PatientDataset
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def main(config_path):
    cfg = load_config(config_path)["deploy"]
    data = pd.read_csv(cfg["csv_path"])
    targets = data[cfg["target_column"]].values
    # Load the label encoder from the training phase
    with open(cfg["label_encoder_path"], "rb") as file:
        label_encoder = pickle.load(file)
    targets = label_encoder.transform(targets)
    patient_ids = data[cfg["patient_id_column"]].values

    with h5py.File(cfg["h5_path"], "r") as f:
        h5_patient_ids = list(f.keys())

    csv_patient_ids_set = set(patient_ids)
    h5_patient_ids_set = set(h5_patient_ids)

    common_patient_ids = list(csv_patient_ids_set & h5_patient_ids_set)
    common_indices = [
        i for i, pid in enumerate(patient_ids) if pid in common_patient_ids
    ]
    print(f"Found {len(common_indices)} patients")
    patient_ids = patient_ids[common_indices]
    targets = targets[common_indices]

    torch.set_float32_matmul_precision("high")

    test_dataset = PatientDataset(cfg["h5_path"], patient_ids, targets)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)

    input_dim = test_dataset[0][0].shape[0]
    output_dim = len(np.unique(targets))

    all_test_results = []
    auroc_scores = []

    for fold in range(cfg["hps"]["n_folds"]):
        fold_output_folder = os.path.join(cfg["output_folder"], f"fold_{fold}")
        model_path = os.path.join(fold_output_folder, "best_model.ckpt")
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} does not exist. Skipping this fold.")
            continue

        model = MLP.load_from_checkpoint(
            model_path, input_dim=input_dim, output_dim=output_dim, hidden_dim=cfg["hps"]["hidden_dim"],
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        y_true = []
        y_pred = []

        with torch.inference_mode():
            fold_test_results = []
            for i, (x, y) in enumerate(tqdm(test_loader)):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = model.criterion(y_hat, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.append(F.softmax(y_hat,dim=-1)[0][-1].item())
                fold_test_results.append(
                    {
                        "patient_id": patient_ids[i],
                        "ground_truth": label_encoder.inverse_transform(
                            y.cpu().numpy()
                        )[0],
                        "pred_label": label_encoder.inverse_transform(
                            y_hat.argmax(dim=-1).cpu().numpy()
                        )[0],
                        "pred_prob": F.softmax(y_hat,dim=1)[0][-1].item(),
                        "loss": loss,
                        "fold": fold,
                    }
                )
            all_test_results.extend(fold_test_results)

        auroc = roc_auc_score(y_true, y_pred)
        auroc_scores.append({"fold": fold, "auroc": auroc})
        print(f"Fold {fold} AUROC: {auroc}")
    avg_auroc = np.mean([score["auroc"] for score in auroc_scores])
    print(f"Average AUROC over all folds: {avg_auroc}")
    auroc_scores.append({"fold": "average", "auroc": avg_auroc})
        
    deploy_output_folder = os.path.join(cfg["output_folder"], "deploy")
    os.makedirs(deploy_output_folder, exist_ok=True)
    all_test_results_df = pd.DataFrame(all_test_results)
    all_test_results_df.to_csv(
        os.path.join(deploy_output_folder, "all_folds_test_results.csv"), index=False
    )

    auroc_scores_df = pd.DataFrame(auroc_scores)
    auroc_scores_df.to_csv(
        os.path.join(deploy_output_folder, "auroc_scores.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy MLP models on the total test set"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
