# IFS-Light

✨ **Accepted by ACM MM 2025** ✨

This repository contains supplementary code and assets for the paper **"IFS-Light: An Interactive Framework for Single-view Face Relighting with both Facial and Lighting Consistency"**.

IFS-Light performs single-view portrait relighting by combining facial-condition guidance and lighting-consistency constraints. The repository includes inference code, preprocessing scripts, example data, visual comparisons, and notes for downloading the required models.

🌟 **Project materials**

- 📄 Paper: [`ACMMM25-IFS-Light.pdf`](ACMMM25-IFS-Light.pdf)
- 🧩 Supplementary appendix: [`Appendix-3726.pdf`](Appendix-3726.pdf)
- 🖼️ Visual comparisons: [`Visualization Comparison (gif)/`](Visualization%20Comparison%20(gif)/)
- 🎬 Application demo note: [`Application/video.txt`](Application/video.txt)

## Repository Structure

```text
.
├── Code/IFS-Light/                 # Main source code
│   ├── config/ifslight/             # Configuration files
│   ├── guided_diffusion/            # Diffusion model implementation
│   ├── model_3d/                    # FLAME/3D face utilities
│   ├── preprocess_scripts/          # Preprocessing pipeline and sample preprocessed data
│   ├── requirements/IFS-Light.yml   # Conda environment
│   └── sample_scripts/              # Inference scripts and sampling utilities
├── Model/                           # Model download instructions
├── Visualization Comparison (gif)/  # Qualitative comparison examples
├── Application/                     # Demo video note
├── ACMMM25-IFS-Light.pdf            # Paper
├── Appendix-3726.pdf                # Supplementary appendix
└── README.md
```

## Environment

The provided environment targets Python 3.8, PyTorch 1.12.1, CUDA 11.3, and the dependencies listed in `Code/IFS-Light/requirements/IFS-Light.yml`.

```bash
cd Code/IFS-Light
conda env create -f requirements/IFS-Light.yml
conda activate difa
```

If your local CUDA/PyTorch setup differs, install the matching PyTorch build first and then install the remaining packages from the environment file.

## Required Models

The repository does not include large model checkpoints. Download them separately and place them in the paths below.

### IFS-Light Models

1. Skin-tone scale model:
   - Download: <https://drive.google.com/drive/folders/1IQwxlIA5fLUK8Ely_7sbrcyxfDQ2x6ub?usp=sharing>
   - Place at:
     `Code/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/skintone/Model/model.pth`

2. IFS-Light relighting model:
   - Download: <https://drive.google.com/drive/folders/1pNHo3C6xHpw5trsAX9ON_7gKHpGYyOE-?usp=share_link>
   - Place the checkpoint files in:
     `Code/IFS-Light/sample_scripts/model_logs/ours/`

### Third-party Models for Preprocessing

Download the following checkpoints from their original project pages:

| Component | Target path |
| --- | --- |
| ArcFace | `Code/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/Arcface/pretrained/BEST_checkpoint_r18.tar` |
| DECA | `Code/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/DECA/data/` |
| DiffAE | `Code/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/diffae/checkpoints/ffhq256_autoenc/last.ckpt` and `latent.pkl` |
| Face parsing | `Code/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/face-parsing.PyTorch/res/cp/79999_iter.pth` |
| FFHQ align | `Code/IFS-Light/preprocess_scripts/Relighting_preprocessing_tools/FFHQ_align/temp/shape_predictor_68_face_landmarks.dat` |
| Hou et al. (2022) / GeomConsistentFR | `Code/IFS-Light/sample_scripts/model_logs/ours/model_epoch106.pth` |

## Quick Start: Relighting Example

The repository includes two sample FFHQ portraits and preprocessed parameters under:

```text
Code/IFS-Light/preprocess_scripts/ffhq/
```

After installing the environment and placing the required checkpoints, run:

```bash
cd Code/IFS-Light/sample_scripts/inference

python relight.py \
  --dataset ffhqvalid \
  --set valid \
  --step 080000 \
  --out_dir ./ffhq/ \
  --cfg_name ifslight_256.yaml \
  --log_dir ours \
  --diffusion_steps 1000 \
  --timestep_respacing 250 \
  --sample_pair_json ./relight.json \
  --sample_pair_mode pair \
  --itp render_face \
  --itp_step 2 \
  --batch_size 1 \
  --gpu_id 0 \
  --lerp \
  --idx 0 10
```

Results are saved under the output directory, for example:

```text
Code/IFS-Light/sample_scripts/inference/ffhq/
```

## Preprocessing

Preprocessing code is located in:

```text
Code/IFS-Light/preprocess_scripts/
```

The included `ffhq` folder provides two sample portraits and their preprocessed parameters. For new images, use the tools in `Relighting_preprocessing_tools/` to generate aligned images, DECA parameters, face segmentation masks, rendered face conditions, and other inputs expected by the inference script.

## Visualizations and Demo

- `Visualization Comparison (gif)/` contains input/reference images and GIF comparisons with Hou et al. (2022), Difareli, and IFS-Light.
- `Application/video.txt` describes the recorded demo material for the interactive relighting interface and workflow.

## Citation

If you use this code or data, please cite the IFS-Light paper:

```bibtex
@inproceedings{wang2025ifs,
  title={IFS-Light: An Interactive Framework for Single-view Face Relighting with both Facial and Lighting Consistency},
  author={Wang, Shuyang and Li, Chunxiao and Ming, Anlong},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={8332--8340},
  year={2025}
}
```
