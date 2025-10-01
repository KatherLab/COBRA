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
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import pickle
from cobra.crossval.train import MLP, PatientDataset
from tqdm import tqdm


def load_config(config_path):
    """
    Load configuration from a YAML file.
    This function opens the file at the given path, reads its contents, and loads the configuration 
    using yaml.safe_load. The configuration is returned as a Python dictionary.
    Parameters:
        config_path (str): The file path to the YAML configuration file.
    Returns:
        dict: A dictionary representing the configuration settings loaded from the YAML file.
    Raises:
        FileNotFoundError: If the file at config_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """

    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def main(config_path):
    """
    Main function for deploying evaluation of the pre-trained models on test data.
    This function performs the following steps:
    1. Loads the deployment configuration from the specified config file.
    2. Reads patient data from a CSV file and extracts target labels and patient identifiers.
    3. Loads the label encoder from the training phase and transforms target labels.
    4. Retrieves patient IDs from an H5 file and identifies common patients present in both CSV and H5 files.
    5. Initializes a dataset and dataloader for test data.
    6. Iterates over each cross-validation fold:
        - Checks if a model checkpoint exists for the fold.
        - Loads the trained model and sets it to evaluation mode.
        - Processes the test dataset to compute predictions, losses, and transforms predicted labels back to the original label space.
        - Aggregates predictions and computes the AUROC for the fold.
    7. Calculates the average AUROC over all folds.
    8. Saves the detailed per-patient predictions and AUROC scores for each fold (and average) as CSV files in the specified output folder.
    Parameters:
         config_path (str): The file path to the configuration file containing deployment settings, including paths to CSV, H5, label encoder,
                                  output folder, and hyperparameters for the model.
    Returns:
         None
    Notes:
         - The function assumes that the CSV file contains columns for patient IDs, target labels, and that the H5 file keys correspond to patient IDs.
         - The models are loaded from checkpoints for each cross-validation fold as specified in the configuration.
         - The function uses torch.inference_mode for inference and computes softmax probabilities on the model outputs.
    """

    cfg = load_config(config_path)["deploy"]
    data = pd.read_csv(cfg["csv_path"])
    
    # Handle both single target column and multiple target columns
    target_columns = cfg.get("target_columns", [cfg.get("target_column")])
    if target_columns == [None]:
        raise ValueError("Either 'target_column' or 'target_columns' must be specified in config")
    
    is_multilabel = len(target_columns) > 1
    
    if is_multilabel:
        # For multilabel: drop rows with NaN in any target column
        data = data.dropna(subset=target_columns, axis=0)
        targets = data[target_columns].values.astype(float)
        # Ensure binary (0/1) for multilabel
        targets = (targets > 0).astype(float)
    else:
        # For single target: existing logic
        data = data.dropna(subset=target_columns, axis=0)
        targets = data[target_columns[0]].values
    
    # Load the label encoders from the training phase
    label_encoders_path = cfg.get("label_encoders_path", cfg.get("label_encoder_path"))
    with open(label_encoders_path, "rb") as file:
        label_encoders = pickle.load(file)
    
    if not is_multilabel:
        targets = label_encoders.transform(targets)
    
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

    test_dataset = PatientDataset(cfg["h5_path"], patient_ids, targets, is_multilabel=is_multilabel)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False)

    input_dim = test_dataset[0][0].shape[0]
    
    if is_multilabel:
        output_dim = len(target_columns)  # Number of labels for multilabel
    else:
        output_dim = len(np.unique(targets))  # Number of classes for multiclass

    all_test_results = []
    auroc_scores = []

    for fold in range(cfg["hps"]["n_folds"]):
        fold_output_folder = os.path.join(cfg["output_folder"], f"fold_{fold}")
        model_path = os.path.join(fold_output_folder, "best_model.ckpt")
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} does not exist. Skipping this fold.")
            continue

        model = MLP.load_from_checkpoint(
            model_path, 
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_dim=cfg["hps"]["hidden_dim"],
            is_multilabel=is_multilabel
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
                
                result_dict = {
                    "patient_id": patient_ids[i],
                    "loss": loss,
                    "fold": fold,
                }
                
                if is_multilabel:
                    # For multilabel: convert predictions to binary and add individual label columns
                    y_pred_prob = torch.sigmoid(y_hat)
                    y_pred_binary = (y_pred_prob > 0.5).float()
                    
                    for j, col in enumerate(target_columns):
                        result_dict[f"ground_truth_{col}"] = y.cpu().numpy()[0, j]
                        result_dict[f"prediction_{col}"] = y_pred_binary.cpu().numpy()[0, j]
                        result_dict[f"probability_{col}"] = y_pred_prob.cpu().numpy()[0, j]
                        
                    # For AUROC calculation in multilabel, we'll use the probabilities
                    y_true.extend(y.cpu().numpy().flatten())
                    y_pred.extend(y_pred_prob.cpu().numpy().flatten())
                else:
                    # For multiclass: use label encoder inverse transform
                    result_dict["ground_truth"] = label_encoders.inverse_transform(
                        y.cpu().numpy()
                    )[0]
                    result_dict["pred_label"] = label_encoders.inverse_transform(
                        y_hat.argmax(dim=-1).cpu().numpy()
                    )[0]
                    result_dict["pred_prob"] = F.softmax(y_hat, dim=1)[0][-1].item()
                    
                    # For AUROC calculation in multiclass
                    y_true.extend(y.cpu().numpy())
                    y_pred.append(F.softmax(y_hat, dim=-1)[0][-1].item())
                    
                fold_test_results.append(result_dict)
            all_test_results.extend(fold_test_results)

        if is_multilabel:
            # For multilabel, reshape arrays and calculate metrics
            y_true_array = np.array(y_true).reshape(-1, len(target_columns))
            y_pred_array = np.array(y_pred).reshape(-1, len(target_columns))
            y_pred_binary = (y_pred_array > 0.5).astype(int)
            
            # Calculate macro-averaged AUROC and F1
            auroc = roc_auc_score(y_true_array, y_pred_array, average='macro')
            f1_macro = f1_score(y_true_array, y_pred_binary, average='macro')
            
            # Calculate individual metrics for each target
            individual_metrics = {"fold": fold, "macro_auroc": auroc, "macro_f1": f1_macro}
            
            print(f"Fold {fold} AUROC (macro): {auroc:.4f}")
            print(f"Fold {fold} F1 (macro): {f1_macro:.4f}")
            
            for i, col in enumerate(target_columns):
                # Individual AUROC and F1 for each target
                try:
                    target_auroc = roc_auc_score(y_true_array[:, i], y_pred_array[:, i])
                    target_f1 = f1_score(y_true_array[:, i], y_pred_binary[:, i])
                    individual_metrics[f"auroc_{col}"] = target_auroc
                    individual_metrics[f"f1_{col}"] = target_f1
                    print(f"Fold {fold} {col} - AUROC: {target_auroc:.4f}, F1: {target_f1:.4f}")
                except ValueError as e:
                    # Handle cases where only one class is present
                    print(f"Fold {fold} {col} - Warning: {e}")
                    individual_metrics[f"auroc_{col}"] = np.nan
                    individual_metrics[f"f1_{col}"] = np.nan
            
            # Save individual metrics for this fold
            individual_metrics_df = pd.DataFrame([individual_metrics])
            individual_metrics_df.to_csv(
                os.path.join(cfg["output_folder"], f"fold_{fold}_individual_metrics.csv"),
                index=False,
            )
            
            auroc_scores.append({"fold": fold, "auroc": auroc, "f1": f1_macro})
        else:
            # For multiclass, use existing logic
            auroc = roc_auc_score(y_true, y_pred)
            auroc_scores.append({"fold": fold, "auroc": auroc})
            print(f"Fold {fold} AUROC: {auroc}")
    avg_auroc = np.mean([score["auroc"] for score in auroc_scores])
    print(f"Average AUROC over all folds: {avg_auroc}")
    
    if is_multilabel and any("f1" in score for score in auroc_scores):
        avg_f1 = np.mean([score["f1"] for score in auroc_scores if "f1" in score])
        print(f"Average F1 over all folds: {avg_f1}")
        auroc_scores.append({"fold": "average", "auroc": avg_auroc, "f1": avg_f1})
        
        # Aggregate individual metrics across folds
        all_individual_metrics = []
        for fold in range(cfg["hps"]["n_folds"]):
            individual_file = os.path.join(cfg["output_folder"], f"fold_{fold}_individual_metrics.csv")
            if os.path.exists(individual_file):
                fold_df = pd.read_csv(individual_file)
                all_individual_metrics.append(fold_df)
        
        if all_individual_metrics:
            combined_metrics = pd.concat(all_individual_metrics, ignore_index=True)
            
            print("\nAverage metrics per target across all folds:")
            for col in target_columns:
                col_auroc = combined_metrics[f"auroc_{col}"].mean()
                col_f1 = combined_metrics[f"f1_{col}"].mean()
                print(f"  {col} - AUROC: {col_auroc:.4f}, F1: {col_f1:.4f}")
            
            macro_auroc = combined_metrics['macro_auroc'].mean()
            macro_f1 = combined_metrics['macro_f1'].mean()
            print(f"  Macro average - AUROC: {macro_auroc:.4f}, F1: {macro_f1:.4f}")
            
            # Save combined individual metrics
            combined_metrics.to_csv(
                os.path.join(cfg["output_folder"], "all_folds_individual_metrics.csv"), 
                index=False
            )
    else:
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
