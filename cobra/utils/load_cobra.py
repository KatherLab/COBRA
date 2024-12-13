# %%
from huggingface_hub import login, hf_hub_download
from cobra.model.cobra import Cobra
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_cobra(download_weights=False,checkpoint_path=None):
    if download_weights:
        checkpoint_path = hf_hub_download("TLenz/COBRA", filename="pytorch_model.bin", local_dir="weights/", force_download=True)
        print(f"Saving model to {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = Cobra()
    model.load_state_dict(state_dict)
    print("COBRA model loaded successfully")
    return model