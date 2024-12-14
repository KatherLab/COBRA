<h1>
    <img src="assets/cobra.png" alt="failed loading the image" width="50" style="vertical-align: middle; margin-right: 10px;">
    COntrastive Biomarker Representation Alignment (COBRA)
</h1>

[Preprint](https://arxiv.org/abs/2411.13623) | [Download Model](https://huggingface.co/KatherLab/COBRA) 

### Abstract

>Representation learning of pathology whole-slide images(WSIs) has primarily relied on weak supervision with Multiple Instance Learning (MIL). This approach leads to slide representations highly tailored to a specific clinical task. Self-supervised learning (SSL) has been successfully applied to train histopathology foundation models (FMs) for patch embedding generation. However, generating patient or slide level embeddings remains challenging. Existing approaches for slide representation learning extend the principles of SSL from patch level learning to entire slides by aligning different augmentations of the slide or by utilizing multimodal data. By integrating tile embeddings from multiple FMs, we propose a new single modality SSL method in feature space that generates useful slide representations. Our contrastive pretraining strategy, called COBRA, employs multiple FMs and an architecture based on Mamba-2. COBRA exceeds performance of state-of-the-art slide encoders on four different public Clinical Protemic Tumor Analysis Consortium (CPTAC) cohorts on average by at least +4.5% AUC, despite only being pretrained on 3048 WSIs from The Cancer Genome Atlas (TCGA). Additionally, COBRA is readily compatible at inference time with previously unseen feature extractors.

<p align="center">
    <img src="assets/fig1.png" alt="failed loading the image" width="1100"/>
</p>

### Installation

To install the necessary dependencies, run the following commands:

```bash
git clone https://github.com/KatherLab/COBRA.git && cd COBRA
pip install uv
uv venv --python=3.11
source .venv/bin/activate
uv pip install torch==2.4.1 setuptools packaging wheel numpy>=2.0.0
uv sync --no-build-isolation
```

### WSI Level Embeddings

To deploy the COBRA model, follow these steps:

1. **Prepare your data**: extract tile embeddings with one or more patch encoders of your preference using [STAMP](https://github.com/KatherLab/STAMP).
2. **Request Access on [Huggingface](https://huggingface.co/KatherLab/COBRA)**.
3. **Deploy COBRA**: extract slide level embeddings using COBRA
```bash 
python -m cobra.inference.extract_feats --feat_dir <tile_emb_dir> --output_dir <slide_emb_dir> 
```

#### References
- [CTransPath](https://github.com/Xiyue-Wang/TransPath)
>Xiyue Wang, Sen Yang, Jun Zhang, Minghui Wang,
>Jing Zhang, Wei Yang, Junzhou Huang, and Xiao Han.
>Transformer-based unsupervised contrastive learning for
>histopathological image classification. Medical Image Anal-
>ysis, 2022
- [UNI](https://github.com/mahmoodlab/uni)
>Richard J Chen, Tong Ding, Ming Y Lu, Drew FK
>Williamson, Guillaume Jaume, Bowen Chen, Andrew
>Zhang, Daniel Shao, Andrew H Song, Muhammad Shaban,
>et al. Towards a general-purpose foundation model for com-
>putational pathology. Nature Medicine, 2024
- [Virchow2](https://huggingface.co/paige-ai/Virchow2)
>Eric Zimmermann, Eugene Vorontsov, Julian Viret, Adam
>Casson, Michal Zelechowski, George Shaikovski, Neil
>Tenenholtz, James Hall, David Klimstra, Razik Yousfi,
>Thomas Fuchs, Nicolo Fusi, Siqi Liu, and Kristen Sever-
>son. Virchow2: Scaling self-supervised mixed magnification
>models in pathology, 2024
- [H-Optimus-0](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0)
>Charlie Saillard, Rodolphe Jenatton, Felipe Llinares-López,
>Zelda Mariet, David Cahané, Eric Durand, and Jean-Philippe
>Vert. H-optimus-0, 2024
- [STAMP](https://github.com/KatherLab/STAMP)
>Omar S. M. El Nahhas, Marko van Treeck, Georg Wölflein,
>Michaela Unger, Marta Ligero, Tim Lenz, Sophia J. Wagner,
>Katherine J. Hewitt, Firas Khader, Sebastian Foersch, Daniel
>Truhn, and Jakob Nikolas Kather. From whole-slide im-
>age to biomarker prediction: end-to-end weakly supervised
>deep learning in computational pathology. Nature Protocols,
>2024
- [MoCo-v3](https://github.com/facebookresearch/moco-v3)
>Xinlei Chen*, Saining Xie*, and Kaiming He. An empirical
>study of training self-supervised vision transformers. arXiv
>preprint arXiv:2104.02057, 2021

## Citation

If you find our work useful in your research or if you use parts of this code please consider citing our [preprint](https://arxiv.org/abs/2411.13623):

```bibtex
@misc{cobra,
      title={Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning}, 
      author={Tim Lenz* and Peter Neidlinger* and Marta Ligero and Georg Wölflein and Marko van Treeck and Jakob Nikolas Kather},
      year={2024},
      eprint={2411.13623},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13623}, 
}
```
