import os
import yaml
import pandas as pd
import numpy as np
import h5py
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAUROC, MultilabelAUROC
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, Dataset, random_split
import argparse
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import warnings
import pickle


class PatientDataset(Dataset):
    def __init__(self, h5_file, patient_ids, target_ids, is_multilabel=False):
        self.h5_file = h5_file
        self.target_ids = target_ids
        self.patient_ids = patient_ids
        self.is_multilabel = is_multilabel

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        target = self.target_ids[idx]
        with h5py.File(self.h5_file, 'r') as f:
            features = f[patient_id][:]
        
        if self.is_multilabel:
            # For multilabel, target should be a binary vector
            return torch.tensor(features, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32
            )
        else:
            # For multiclass, target should be a single integer
            return torch.tensor(features, dtype=torch.float32), torch.tensor(
                target, dtype=torch.long
            )


class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=512, lr=1e-4, dropout=0.5, is_multilabel=False):
        super(MLP, self).__init__()
        self.is_multilabel = is_multilabel
        self.output_dim = output_dim
        
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        if self.is_multilabel:
            self.criterion = nn.BCEWithLogitsLoss()
            self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=output_dim)
            self.valid_auroc = MultilabelAUROC(num_labels=output_dim, average=None)  # None for per-class
            self.test_auroc = MultilabelAUROC(num_labels=output_dim, average=None)   # None for per-class
            self.valid_auroc_macro = MultilabelAUROC(num_labels=output_dim, average="macro")
            self.test_auroc_macro = MultilabelAUROC(num_labels=output_dim, average="macro")
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
            self.valid_auroc = MulticlassAUROC(output_dim)
            self.test_auroc = MulticlassAUROC(output_dim)
        
        self.lr = lr

    def forward(self, x):
        logits = self.model(x)
        if self.is_multilabel:
            # For multilabel, return logits directly (will be passed to BCEWithLogitsLoss)
            return logits
        else:
            # For multiclass, return logits directly (will be passed to CrossEntropyLoss)
            return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        if self.is_multilabel:
            # For multilabel, apply sigmoid before accuracy calculation
            y_pred_prob = torch.sigmoid(y_hat)
            acc = self.accuracy(y_pred_prob, y.int())
        else:
            # For multiclass, use logits directly
            acc = self.accuracy(y_hat, y)
            
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        if self.is_multilabel:
            # For multilabel, apply sigmoid before metrics calculation
            y_pred_prob = torch.sigmoid(y_hat)
            acc = self.accuracy(y_pred_prob, y.int())
            self.valid_auroc.update(y_pred_prob, y.int())
            self.valid_auroc_macro.update(y_pred_prob, y.int())
        else:
            # For multiclass, use logits/probabilities directly
            acc = self.accuracy(y_hat, y)
            self.valid_auroc.update(y_hat, y)
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.is_multilabel:
            self.log("val_auroc", self.valid_auroc_macro, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("val_auroc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        if self.is_multilabel:
            # For multilabel, apply sigmoid before metrics calculation
            y_pred_prob = torch.sigmoid(y_hat)
            acc = self.accuracy(y_pred_prob, y.int())
            self.test_auroc.update(y_pred_prob, y.int())
            self.test_auroc_macro.update(y_pred_prob, y.int())
        else:
            # For multiclass, use logits/probabilities directly
            acc = self.accuracy(y_hat, y)
            self.test_auroc.update(y_hat, y)
            
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
        if self.is_multilabel:
            self.log("test_auroc", self.test_auroc_macro)
        else:
            self.log("test_auroc", self.test_auroc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def load_config(config_path):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def main(config_path):
    """
    Main function to orchestrate the training and cross-validation process for the MLP model.

    This function performs the following operations:
        - Loads the training configuration from a provided YAML file.
        - Ensures that the output folder exists and saves a copy of the configuration.
        - Reads input data from a CSV or Excel file, while verifying that the file format is supported.
        - Cleans the data by removing rows with missing target values.
        - Encodes target labels using a LabelEncoder and saves the encoder for later use.
        - Loads patient IDs from an H5 file and compares them with those in the CSV, issuing warnings if there are any mismatches.
        - Filters the dataset to include only common patient IDs present in both files.
        - Sets up stratified K-fold cross-validation, ensuring balanced splits based on the target labels.
        - For each fold:
                - Splits the data into training/validation and test sets.
                - Constructs PyTorch DataLoaders for training, validation, and testing.
                - Instantiates an MLP model with parameters specified in the configuration.
                - Trains the model using PyTorch Lightning, with callbacks for model checkpointing and early stopping.
                - Evaluates the model on the test set and logs the test AUROC metric.
                - Stores detailed test results (including patient IDs, predictions, ground truth, and loss) as CSV files.
        - Aggregates the AUROC scores across all folds and saves the summary to disk.

    Parameters:
            config_path (str): The file path to the YAML configuration file containing training settings, data paths, and hyperparameters.

    Raises:
            ValueError: If the input data file format is unsupported (i.e., not a CSV or XLSX file).

    Returns:
            None

    Side Effects:
            - Creates directories and files (config.yaml, label_encoder.pkl, best_model.ckpt, test results CSVs, and fold AUROC CSV) in the specified output folder.
            - Logs progress and warnings via printed messages and the warnings module.
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    cfg = load_config(config_path)["train"]
    if not os.path.exists(cfg["output_folder"]):
        os.makedirs(cfg["output_folder"])
    config_output_path = os.path.join(cfg["output_folder"], "config.yaml")
    with open(config_output_path, "w") as file:
        yaml.dump(cfg, file)
    hps = cfg["hps"]
    if cfg["csv_path"].endswith(".csv"):
        data = pd.read_csv(cfg["csv_path"])
    elif cfg["csv_path"].endswith(".xlsx"):
        data = pd.read_excel(cfg["csv_path"])
    else:
        raise ValueError(f"Unsupported file format: only .csv and .xlsx are supported found {os.path.splitext(cfg['csv_path'])[1]}")
    
    # Handle both single target column and multiple target columns
    target_columns = cfg.get("target_columns", [cfg.get("target_column")])
    if target_columns == [None]:
        raise ValueError("Either 'target_column' or 'target_columns' must be specified in config")
    
    is_multilabel = len(target_columns) > 1
    
    if is_multilabel:
        # For multilabel: drop rows with NaN in any target column
        data = data.dropna(subset=target_columns, axis=0)
        targets = data[target_columns].values.astype(float)
        label_encoders = {}
        
        # For multilabel, we expect binary targets (0/1) or we need to convert categorical to binary
        for i, col in enumerate(target_columns):
            unique_vals = np.unique(targets[:, i])
            if len(unique_vals) > 2:
                # If more than 2 unique values, use label encoding then convert to binary
                le = LabelEncoder()
                targets[:, i] = le.fit_transform(targets[:, i])
                label_encoders[col] = le
            # Ensure binary (0/1)
            targets[:, i] = (targets[:, i] > 0).astype(float)
    else:
        # For single target: existing logic
        data = data.dropna(subset=target_columns, axis=0)
        targets = data[target_columns[0]].values
        single_label_encoder = LabelEncoder()
        targets = single_label_encoder.fit_transform(targets)
        label_encoders = single_label_encoder  # Store as single encoder for backward compatibility
    
    patient_ids = data[cfg["patient_id_column"]].values

    torch.set_float32_matmul_precision("high")

    label_encoder_path = os.path.join(cfg["output_folder"], "label_encoders.pkl")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoders, f)
    with h5py.File(cfg["h5_path"], "r") as f:
        h5_patient_ids = list(f.keys())

    csv_patient_ids_set = set(patient_ids)
    h5_patient_ids_set = set(h5_patient_ids)

    missing_in_h5 = csv_patient_ids_set - h5_patient_ids_set
    missing_in_csv = h5_patient_ids_set - csv_patient_ids_set

    if missing_in_h5:
        warnings.warn(f"Patient IDs missing in H5 file: {missing_in_h5}")
    if missing_in_csv:
        warnings.warn(f"Patient IDs missing in CSV file: {missing_in_csv}")

    common_patient_ids = list(csv_patient_ids_set & h5_patient_ids_set)
    common_indices = [
       i for i, pid in enumerate(patient_ids) if pid in common_patient_ids
    ]

    patient_ids = patient_ids[common_indices]
    targets = targets[common_indices]

    # Choose appropriate cross-validation strategy
    if is_multilabel:
        # For multilabel, use simple KFold since stratified multilabel CV is complex
        cv = KFold(n_splits=hps["n_folds"], shuffle=True, random_state=42)
        cv_splits = cv.split(patient_ids)
    else:
        # For single-label, use stratified CV
        skf = StratifiedKFold(n_splits=hps["n_folds"], shuffle=True, random_state=42)
        cv_splits = skf.split(patient_ids, targets)
    
    all_fold_aurocs = []    
    for fold, (train_val_idx, test_idx) in enumerate(cv_splits):
        fold_output_folder = os.path.join(cfg["output_folder"], f"fold_{fold}")
        if os.path.exists(os.path.join(fold_output_folder, "best_model.ckpt")):
            print(f"Model for fold {fold} already exists. Skipping this fold.")
            continue
        train_val_ids = patient_ids[train_val_idx]
        train_val_targets = targets[train_val_idx]
        test_ids = patient_ids[test_idx]
        test_targets = targets[test_idx]

        train_val_dataset = PatientDataset(
            cfg["h5_path"],
            train_val_ids,
            train_val_targets,
            is_multilabel=is_multilabel
        )
        test_dataset = PatientDataset(
            cfg["h5_path"],
            test_ids,
            test_targets,
            is_multilabel=is_multilabel
        )

        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=hps["batch_size"],
            shuffle=True,
            num_workers=hps["num_workers"],
        )
        val_loader = DataLoader(val_dataset, batch_size=1, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

        input_dim = train_dataset[0][0].shape[0]
        
        if is_multilabel:
            output_dim = len(target_columns)  # Number of labels for multilabel
        else:
            output_dim = len(np.unique(targets))  # Number of classes for multiclass
            
        print(f"Input dim: {input_dim}, Output dim: {output_dim}, Multilabel: {is_multilabel}")

        model = MLP(
            input_dim, 
            output_dim, 
            hidden_dim=hps["hidden_dim"], 
            lr=hps["lr"], 
            dropout=hps["dropout"],
            is_multilabel=is_multilabel
        )

        if not os.path.exists(cfg["output_folder"]):
            os.makedirs(cfg["output_folder"])

        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_output_folder,
            filename="best_model",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=hps["patience"], mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=hps["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping_callback],
            devices=1,  # Use only one GPU
            accelerator="gpu",
        )

        trainer.fit(model, train_loader, val_loader)

        model = MLP.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hps["hidden_dim"],
            is_multilabel=is_multilabel
        )
        trainer.test(model, test_loader)

        with torch.inference_mode():
            test_results = []
            for i, (x, y) in enumerate(test_loader):
                y_hat = model(x)
                loss = model.criterion(y_hat, y).item()
                
                result_dict = {
                    "patient_id": test_ids[i],
                    "loss": loss,
                }
                
                if is_multilabel:
                    # For multilabel: convert predictions to binary and add individual label columns
                    y_pred_prob = torch.sigmoid(y_hat)
                    y_pred_binary = (y_pred_prob > 0.5).float()
                    
                    for j, col in enumerate(target_columns):
                        result_dict[f"ground_truth_{col}"] = y.numpy()[0, j]
                        result_dict[f"prediction_{col}"] = y_pred_binary.numpy()[0, j]
                        result_dict[f"probability_{col}"] = y_pred_prob.numpy()[0, j]
                else:
                    # For multiclass: use label encoder inverse transform
                    result_dict["ground_truth"] = label_encoders.inverse_transform(y.numpy())[0]
                    result_dict["prediction"] = label_encoders.inverse_transform(
                        y_hat.argmax(dim=1).numpy()
                    )[0]
                    
                test_results.append(result_dict)

        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(
            os.path.join(cfg["output_folder"], f"fold_{fold}_test_results.csv"),
            index=False,
        )
        
        if is_multilabel:
            # Get individual and macro AUROC scores
            avg_auroc = trainer.callback_metrics["test_auroc"].item()
            individual_aurocs = model.test_auroc.compute().cpu().numpy()
            
            print(f"Fold {fold} Test AUROC (macro): {avg_auroc}")
            for i, col in enumerate(target_columns):
                print(f"Fold {fold} Test AUROC ({col}): {individual_aurocs[i]}")
            
            # Save individual AUROC scores
            individual_auroc_data = {"fold": fold, "macro_auroc": avg_auroc}
            for i, col in enumerate(target_columns):
                individual_auroc_data[f"auroc_{col}"] = individual_aurocs[i]
            
            individual_auroc_df = pd.DataFrame([individual_auroc_data])
            individual_auroc_df.to_csv(
                os.path.join(cfg["output_folder"], f"fold_{fold}_individual_aurocs.csv"),
                index=False,
            )
        else:
            avg_auroc = trainer.callback_metrics["test_auroc"].item()
            print(f"Fold {fold} Test AUROC: {avg_auroc}")

        all_fold_aurocs.append(avg_auroc)
    auroc_df = pd.DataFrame({"fold": list(range(len(all_fold_aurocs))), "auroc": all_fold_aurocs})
    auroc_df.to_csv(os.path.join(cfg["output_folder"], "fold_aurocs.csv"), index=False)
    avg_auroc_over_folds = np.mean(all_fold_aurocs)
    print(f"Average Test AUROC over all folds: {avg_auroc_over_folds}")
    
    if is_multilabel:
        # Aggregate individual AUROC scores across folds
        all_individual_aurocs = []
        for fold in range(len(all_fold_aurocs)):
            individual_file = os.path.join(cfg["output_folder"], f"fold_{fold}_individual_aurocs.csv")
            if os.path.exists(individual_file):
                fold_df = pd.read_csv(individual_file)
                all_individual_aurocs.append(fold_df)
        
        if all_individual_aurocs:
            combined_aurocs = pd.concat(all_individual_aurocs, ignore_index=True)
            combined_aurocs.to_csv(os.path.join(cfg["output_folder"], "all_folds_individual_aurocs.csv"), index=False)
            
            # Calculate average AUROC for each target
            print("\nAverage AUROC per target across all folds:")
            for col in target_columns:
                col_auroc = combined_aurocs[f"auroc_{col}"].mean()
                print(f"  {col}: {col_auroc:.4f}")
            print(f"  Macro average: {combined_aurocs['macro_auroc'].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP with cross-validation")
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
