# Rectifying the Depths

This repository is the ECCV 2026 project page for **Rectifying the Depths: A Unified Flow-Based Generative Model for Multi-Condition Underwater Enhancement**.

## Repository Status

The full training and inference code will be released after paper acceptance.


## Environment

- Python 3.12
- PyTorch 1.8
- `numpy`
- `matplotlib`
- `pandas`
- `Pillow`
- NVIDIA RTX A6000 for the paper-reported training setup

## Planned Usage

The commands below are placeholders for the upcoming code release.

```bash
# Placeholder command for training
python train.py --config configs/rf_mcuie.yaml

# Placeholder command for inference
python infer.py --config configs/rf_mcuie.yaml --input_dir data/examples --output_dir outputs

# Placeholder command for evaluation
python evaluate.py --config configs/rf_mcuie.yaml --dataset UIEB
```

## Repository Layout

- `PIC/`: visualization assets and auxiliary plotting scripts
- `scripts/`: result-figure helper scripts
- `input image/`: example input/output images for visualization utilities

## Benchmark Highlights

The current paper reports the following results for **Ours**.

| Dataset | UCIQE | UIQM | PSNR | SSIM | LPIPS | FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSUI | 0.59 | 2.98 | 24.91 | 0.90 | 0.06 | 26.69 |
| UIEB | 0.58 | 2.91 | 23.18 | 0.87 | 0.06 | 28.78 |
| U45  | 0.61 | 3.32 | -- | -- | -- | -- |

Default inference in the paper uses `N=3`, which gives the best quality-efficiency balance in the reported `512 x 512` analysis.
