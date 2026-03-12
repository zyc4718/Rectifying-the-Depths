# Rectifying the Depths: A Unified Flow-Based Generative Model for Multi-Condition Underwater Enhancement

This repository accompanies the ECCV 2026 paper on rectified-flow-based multi-condition underwater image enhancement.

## Repository Status

This repository currently contains the paper sources, supplementary material, and figure-generation utilities used in the submission.

The full training and inference code will be released after paper acceptance.

## Environment

### Manuscript and figure reproduction

- Python 3 with `numpy`, `matplotlib`, `pandas`, and `Pillow`
- TeX Live with `pdflatex` and `bibtex`

### Paper-reported training configuration

- Framework: PyTorch 1.8
- GPU: single NVIDIA RTX A6000
- Crop size: `256 x 256`
- Batch size: `8`
- Optimizer: Adam with learning rate `1e-4`
- Training iterations: `1,000,000`
- Rectified-flow step during training: `N=1`
- Default inference step during testing: `N=3`

## Quick Start

### Generate paper figures

```bash
python plot_uie_paper_figs.py
python PIC/plot_rf_ode_trajectories.py
python scripts/generate_result_figs.py
```

### Compile the main paper

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Compile the expanded supplementary material

```bash
pdflatex supplementary_expanded.tex
pdflatex supplementary_expanded.tex
```

## Repository Layout

- `main.tex`: main ECCV paper entry
- `supplementary_expanded.tex`: expanded supplementary material entry
- `Sections/`: paper and supplementary section files
- `PIC/`: manuscript figures and figure-generation scripts
- `scripts/`: auxiliary plotting scripts
- `input image/`: real input/output pair used by the ODE trajectory visualization
- `bibliography/`: BibTeX database

## Benchmark Highlights

The following numbers summarize the current paper results for **Ours**.

| Dataset | UCIQE | UIQM | PSNR | SSIM | LPIPS | FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSUI | 0.59 | 2.98 | 24.91 | 0.90 | 0.06 | 26.69 |
| UIEB | 0.58 | 2.91 | 23.18 | 0.87 | 0.06 | 28.78 |
| U45  | 0.61 | 3.32 | -- | -- | -- | -- |

Default inference uses `N=3`, which provides the best quality-efficiency balance in the reported `512 x 512` speed-quality trade-off analysis.
