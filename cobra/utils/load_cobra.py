# %%
from huggingface_hub import login, hf_hub_download
from cobra.model.cobra import Cobra
import torch
import warnings
import os
import requests
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_cobra(download_weights=False, checkpoint_path="weights/pytorch_model.bin"):
    """
    Load the COBRA model.

    Parameters:
    - download_weights (bool): If True, download the model weights from Hugging Face Hub.
    - checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    - Cobra: The loaded COBRA model.
    
    Raises:
    - FileNotFoundError: If the checkpoint file is not found and download_weights is False.
    """
    if download_weights:
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        download_path = hf_hub_download("KatherLab/COBRA", filename="pytorch_model.bin", 
                                          local_dir=os.path.dirname(checkpoint_path), 
                                          force_download=True)
        os.rename(download_path, checkpoint_path)
        print(f"Saving model to {checkpoint_path}")
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = Cobra(input_dims=[768,1024,1280,1536],)
    model.load_state_dict(state_dict)
    print("COBRA model loaded successfully")
    return model


def get_cobraII(download_weights=False, checkpoint_path="weights/cobraII.pth.tar"):
    """
    Load the COBRAII model.

    Parameters:
    - download_weights (bool): If True, download the model weights from Hugging Face Hub.
    - checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    - Cobra: The loaded COBRAII model.
    
    Raises:
    - FileNotFoundError: If the checkpoint file is not found and download_weights is False.
    """
    if download_weights:
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        download_path = hf_hub_download("KatherLab/COBRA", filename="cobraII.pth.tar", 
                                          local_dir=os.path.dirname(checkpoint_path), 
                                          force_download=True)
        os.rename(download_path, checkpoint_path)
        print(f"Saving model to {checkpoint_path}")
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Cobra(layers=1, input_dims=[512,1024,1280,1536],
                  num_heads=4, dropout=0.2, att_dim=256, d_state=128)
    if "state_dict" in list(state_dict.keys()):
        chkpt = state_dict["state_dict"]
        cobra_weights = {k.split("momentum_enc.")[-1]:v for k,v in chkpt.items() if "momentum_enc" in k and "momentum_enc.proj" not in k}
        if len(list(cobra_weights.keys())) == 0:
            # from stamp finetuning
            cobra_weights = {k.split("cobra.")[-1]:v for k,v in chkpt.items() if "cobra" in k}
    else:
        cobra_weights = state_dict
    model.load_state_dict(cobra_weights)
    print("COBRAII model loaded successfully")
    return model
