# COntrastive Biomarker Representation Alignment (COBRA) 
<p align="center">
    <img src="assets/cobra.png" alt="failed loading the image" width="300"/>
</p>

[Preprint](https://arxiv.org/abs/2411.13623) | [Download Model](https://huggingface.co/KatherLab/COBRA) 

### Abstract

Representation learning of pathology whole-slide images (WSIs) has primarily relied on weak supervision with Multiple Instance Learning (MIL). 
This approach leads to slide representations highly tailored to a specific clinical task. 
Self-supervised learning (SSL) has been successfully applied to train histopathology foundation models (FMs) for patch embedding generation.
However, generating patient or slide level embeddings remains challenging. 
Existing approaches for slide representation learning extend the principles of SSL from patch level learning to entire slides by aligning different augmentations of the slide or by utilizing multimodal data.
By integrating tile embeddings from multiple FMs, we propose a new single modality SSL method in feature space that generates useful slide representations.
Our contrastive pretraining strategy, called CORBA, employs multiple FMs and an architecture based on Mamba-2. CORBA exceeds performance of state-of-the-art slide encoders on four different public CPTAC cohorts on average by at least $+4.5\%$ AUC, despite only being pretrained on 3048 WSIs from TCGA. Additionally, COBRA is readily compatible at inference time with previously unseen feature extractors.

### Installation

To install the necessary dependencies, run the following commands:

```bash
git clone https://github.com/KatherLab/COBRA.git && cd COBRA
pip install uv
uv pip install torch==2.4.1 setuptools
uv sync
```

### WSI Level Embeddings

To deploy the COBRA model, follow these steps:

1. **Prepare your data**: extract tile embeddings with one or more patch encoders of your preference using [STAMP](https://github.com/KatherLab/STAMP).
2. **Deploy COBRA**: extract slide level embeddings using COBRA
```bash 
python -m cobra.inference.extract_feats --feat_dir <tile_emb_dir> --output_dir <slide_emb_dir> 
```



