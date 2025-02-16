import os
import yaml
import pandas as pd
import numpy as np
import h5py
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAUROC
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, random_split
import argparse
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import warnings
import pickle


class PatientDataset(Dataset):
    def __init__(self, h5_file, patient_ids, target_ids):
        self.h5_file = h5_file
        #with h5py.File(self.h5_file, "r") as f:
        #    self.patient_dict = {k: v for k, v in f.items() if k in patient_ids}
        #assert len(self.patient_dict.keys()) == len(patient_ids), "Lengths do not match"
        # self.patient_ids = patient_ids
        #self.target_dict = target_dict
        self.target_ids = target_ids
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        target = self.target_ids[idx]
        with h5py.File(self.h5_file, 'r') as f:
            features = f[patient_id][:]
        #features = self.patient_dict[patient_id]
        #target = self.target_dict[patient_id]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.long
        )


class MLP(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim=512, lr=1e-4,dropout=0.5):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
        self.valid_auroc = MulticlassAUROC(output_dim)
        self.test_auroc = MulticlassAUROC(output_dim)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        # auroc = self.auroc(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.valid_auroc.update(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_auroc", self.valid_auroc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.test_auroc.update(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
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
    data = data.dropna(subset=[cfg["target_column"]], axis=0)
    targets = data[cfg["target_column"]].values
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(targets)
    patient_ids = data[cfg["patient_id_column"]].values

    torch.set_float32_matmul_precision("high")

    label_encoder_path = os.path.join(cfg["output_folder"], "label_encoder.pkl")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
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

    skf = StratifiedKFold(n_splits=hps["n_folds"], shuffle=True, random_state=42)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, targets)):
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
        )
        test_dataset = PatientDataset(
            cfg["h5_path"],
            test_ids,
            test_targets,
        )

        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size]
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
        output_dim = len(np.unique(targets))
        print(f"Input dim: {input_dim}, Output dim: {output_dim}")

        model = MLP(input_dim, output_dim, hidden_dim=hps["hidden_dim"], lr=hps["lr"],dropout=hps["dropout"])

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
        )
        trainer.test(model, test_loader)

        with torch.inference_mode():
            test_results = []
            for i, (x, y) in enumerate(test_loader):
                y_hat = model(x)
                loss = model.criterion(y_hat, y).item()
                test_results.append(
                    {
                        "patient_id": test_ids[i],
                        "ground_truth": label_encoder.inverse_transform(y.numpy())[0],
                        "prediction": label_encoder.inverse_transform(
                            y_hat.argmax(dim=1).numpy()
                        )[0],
                        "loss": loss,
                    }
                )

        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(
            os.path.join(cfg["output_folder"], f"fold_{fold}_test_results.csv"),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP with cross-validation")
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
