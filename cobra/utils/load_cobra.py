# %%
from huggingface_hub import login, hf_hub_download
from cobra.model.cobra import Cobra
import torch
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_cobra(download_weights=False,checkpoint_path=None,local_dir="weights"):
    if download_weights:
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        try:
            checkpoint_path = hf_hub_download("KatherLab/COBRA", filename="pytorch_model.bin", local_dir="weights", force_download=True)
        except Exception as e:
            if "403 Client Error" in str(e):
                raise PermissionError("You do not have permission to access this model. Please ensure you have accepted the model's terms and conditions on https://huggingface.co/KatherLab/COBRA.")
            else:
                raise e
        print(f"Saving model to {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = Cobra()
    model.load_state_dict(state_dict)
    print("COBRA model loaded successfully")
    return model