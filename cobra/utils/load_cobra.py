# %%
from huggingface_hub import login, hf_hub_download
from cobra.model.cobra import Cobra
import torch
import warnings
import os
import requests
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_cobra(download_weights=False,checkpoint_path=None,local_dir="weights"):
    if download_weights:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        checkpoint_path = hf_hub_download("KatherLab/COBRA", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        print(f"Saving model to {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = Cobra(input_dims=[768,1024,1280,1536],)
    model.load_state_dict(state_dict)
    print("COBRA model loaded successfully")
    return model


def get_cobraII(download_weights=True,checkpoint_path=None,local_dir="weights"):
    if download_weights:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        checkpoint_path = hf_hub_download("KatherLab/COBRA", filename="cobraII.pth.tar", local_dir=local_dir, force_download=True)
        print(f"Saving model to {checkpoint_path}")
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = Cobra(layers=1,input_dims=[512,1024,1280,1536],
                  num_heads=4,dropout=0.2,att_dim=256,d_state=128)
    if "state_dict" in list(state_dict.keys()):
        chkpt = state_dict["state_dict"]
        cobra_weights = {k.split("momentum_enc.")[-1]:v for k,v in chkpt.items() if "momentum_enc" in k and "momentum_enc.proj" not in k}
    else:
        cobra_weights = state_dict
    model.load_state_dict(cobra_weights)
    print("COBRA model loaded successfully")
    return model