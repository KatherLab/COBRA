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
        checkpoint_path = hf_hub_download("KatherLab/COBRA", filename="pytorch_model.bin", local_dir="weights", force_download=True)
        print(f"Saving model to {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = Cobra(input_dims=[768,1024,1280,1536],)
    model.load_state_dict(state_dict)
    print("COBRA model loaded successfully")
    return model
