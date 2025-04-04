<h1>
    <img src="assets/cobra.png" alt="failed loading the image" width="50" style="vertical-align: middle; margin-right: 10px;">
    COntrastive Biomarker Representation Alignment (COBRA)
</h1>

<!-- [Preprint](https://arxiv.org/abs/2411.13623) | [Download Models](https://huggingface.co/KatherLab/COBRA) | [Cite](#citation) -->
<a href='https://arxiv.org/abs/2411.13623'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 
<a href='#citation'><img src='https://img.shields.io/badge/Paper-Cite-blue'></a>
<a href='https://huggingface.co/KatherLab/COBRA'><img src='https://img.shields.io/badge/Model-Huggingface-yellow'></a>



### Abstract

>Representation learning of pathology whole-slide images (WSIs) has primarily relied on weak supervision with Multiple Instance Learning (MIL). This approach leads to slide representations highly tailored to a specific clinical task. Self-supervised learning (SSL) has been successfully applied to train histopathology foundation models (FMs) for patch embedding generation. However, generating patient or slide level embeddings remains challenging. Existing approaches for slide representation learning extend the principles of SSL from patch level learning to entire slides by aligning different augmentations of the slide or by utilizing multimodal data. By integrating tile embeddings from multiple FMs, we propose a new single modality SSL method in feature space that generates useful slide representations. Our contrastive pretraining strategy, called COBRA, employs multiple FMs and an architecture based on Mamba-2. COBRA exceeds performance of state-of-the-art slide encoders on four different public Clinical Protemic Tumor Analysis Consortium (CPTAC) cohorts on average by at least +4.4% AUC, despite only being pretrained on 3048 WSIs from The Cancer Genome Atlas (TCGA). Additionally, COBRA is readily compatible at inference time with previously unseen feature extractors.

<p align="center">
    <img src="assets/fig1.png" alt="failed loading the image" width="1100"/>
</p>

### News
- [Feb 27th 2025] Our [paper](https://arxiv.org/abs/2411.13623) has been accepted to [CVPR 2025](https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers)! 🎉
- [Feb 7th 2025]: [COBRA II](https://huggingface.co/KatherLab/COBRA) trained on all TCGA cohorts, is now live and ready to use!!
### Installation

To install the necessary dependencies, run the following commands:

```bash
git clone https://github.com/KatherLab/COBRA.git && cd COBRA
pip install uv
uv venv --python=3.11
source .venv/bin/activate
uv pip install torch==2.4.1 setuptools packaging wheel numpy==2.0.0
uv sync --no-build-isolation
```

If there are any issues, consider also installing hatchling and editables:
```bash
uv pip install hatchling editables
```

### WSI Level Embeddings

To deploy the COBRA model, follow these steps:

1. **Prepare your data**: extract tile embeddings with one or more patch encoders of your preference using [STAMP](https://github.com/KatherLab/STAMP).
    - COBRA I:
        - supported tissue types: LUAD, LUSC, STAD, CRC, BRCA
        - supported patch encoders to generate weighting:
            - CTransPath, UNI, Virchow2, H_optimus_0
        - supported patch encoders for patch feature aggregation:
            - all existing patch encoders compatible with patch size 224x224 px
    - COBRA II:
        - supported tissue types: all cohorts included in TCGA
        - supported patch encoders to generate COBRAII weighting:
            - CONCH, UNI, Virchow2, H_optimus_0
        - supported patch encoders for patch feature aggregation with COBRAII:
            - all existing patch encoders compatible with patch size 224x224 px
    
2. **Request Access on [Huggingface](https://huggingface.co/KatherLab/COBRA)**.
3. **Deploy COBRA**: 
- extract slide level embeddings using COBRA I/II ( | refers to or not pipe)
```bash 
python -m cobra.inference.extract_feats_slide --feat_dir <tile_emb_dir> --output_dir <slide_emb_dir> (--checkpoint_path <checkpoint_path> --config <path_to_config> | -d)  
```
- extract patient level embeddings using COBRA I/II ( | refers to or not pipe)
```bash 
python -m cobra.inference.extract_feats_patient --feat_dir <tile_emb_dir> --output_dir <slide_emb_dir>  --slide_table <slide_table_path>  (--checkpoint_path <checkpoint_path> --config <path_to_config> | -d) 
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
- [CONCH](https://github.com/mahmoodlab/CONCH)
>Lu, Ming Y. and Chen, Bowen and Zhang, Andrew and Williamson, Drew F. K. and Chen, Richard J. and Ding, Tong and Le, Long Phi and Chuang, Yung-Sung and Mahmood, Faisal. A visual-language foundation model for computational pathology. Nature Medicine, (2024)
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
