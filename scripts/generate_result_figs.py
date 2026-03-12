#!/usr/bin/env python3
"""Generate result figures from Sections/Experiment.tex table values.

Outputs:
  - fig/result_dashboard.png
  - fig/result_dashboard.pdf
  - fig/result_heatmaps.png
  - fig/result_heatmaps.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


METHODS = [
    "ICSP",
    "ZAP",
    "UIEC2Net",
    "LANet",
    "Shallow",
    "UIE-WD",
    "Five-a+",
    "U-shape",
    "DM",
    "Ours",
]


LSUI = {
    "UCIQE": [0.55, 0.50, 0.60, 0.58, 0.53, 0.54, 0.59, 0.56, 0.57, 0.59],
    "UIQM": [2.45, 2.51, 3.06, 2.97, 2.69, 2.63, 2.94, 3.08, 2.90, 2.98],
    "PSNR": [11.99, 18.56, 22.96, 21.64, 19.00, 14.19, 22.38, 24.83, 25.12, 24.91],
    "SSIM": [0.55, 0.79, 0.86, 0.86, 0.79, 0.69, 0.86, 0.87, 0.88, 0.90],
    "LPIPS": [0.39, 0.21, 0.11, 0.11, 0.18, 0.24, 0.11, 0.10, 0.09, 0.06],
    "FID": [92.12, 62.35, 40.42, 37.58, 48.37, 79.55, 37.78, 31.49, 32.54, 26.69],
}

UIEB = {
    "UCIQE": [0.54, 0.49, 0.61, 0.57, 0.53, 0.56, 0.60, 0.57, 0.57, 0.58],
    "UIQM": [2.35, 2.44, 3.05, 2.95, 2.61, 2.38, 2.90, 3.10, 2.86, 2.91],
    "PSNR": [11.71, 16.54, 23.82, 21.56, 16.83, 12.56, 22.30, 22.17, 21.64, 23.18],
    "SSIM": [0.53, 0.74, 0.87, 0.85, 0.72, 0.62, 0.86, 0.82, 0.82, 0.87],
    "LPIPS": [0.37, 0.24, 0.07, 0.08, 0.19, 0.26, 0.08, 0.12, 0.10, 0.06],
    "FID": [122.35, 62.65, 30.01, 29.95, 59.68, 104.32, 36.60, 38.91, 33.08, 28.78],
}

U45 = {
    "UCIQE": [0.53, 0.49, 0.61, 0.63, 0.53, 0.54, 0.60, 0.55, 0.60, 0.61],
    "UIQM": [1.98, 2.79, 3.40, 3.29, 2.80, 1.97, 3.34, 3.26, 3.26, 3.32],
}

ABLATION_STEPS = {
    "step": [1, 3, 5, 10],
    "UCIQE": [0.56, 0.59, 0.61, 0.57],
    "UIQM": [2.90, 2.98, 2.92, 3.01],
    "PSNR": [24.13, 24.91, 24.87, 24.65],
    "SSIM": [0.88, 0.90, 0.88, 0.86],
}

ABLATION_MODELS = {
    "model": ["Model I", "Model II", "Model III", "Model IV (Ours)"],
    "UCIQE": [0.55, 0.58, 0.57, 0.59],
    "UIQM": [2.84, 2.92, 2.90, 2.98],
    "PSNR": [24.12, 24.37, 25.05, 24.91],
    "SSIM": [0.84, 0.86, 0.91, 0.90],
}

HIGHER_BETTER = {
    "UCIQE": True,
    "UIQM": True,
    "PSNR": True,
    "SSIM": True,
    "LPIPS": False,
    "FID": False,
}


def _best_baseline(values: list[float], higher_better: bool) -> float:
    baseline = values[:-1]  # exclude "Ours"
    return max(baseline) if higher_better else min(baseline)


def _ours_is_best(values: list[float], higher_better: bool, eps: float = 1e-12) -> bool:
    ours = values[-1]
    if higher_better:
        return ours >= max(values) - eps
    return ours <= min(values) + eps


def _colwise_minmax_norm(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=float)
    for c in range(matrix.shape[1]):
        col = matrix[:, c]
        cmin = np.nanmin(col)
        cmax = np.nanmax(col)
        if np.isclose(cmax, cmin):
            out[:, c] = 0.5
        else:
            out[:, c] = (col - cmin) / (cmax - cmin)
    return out


def draw_heatmap(
    ax: plt.Axes,
    color_data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    annotate_data: np.ndarray | None = None,
    fmt: str = ".2f",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    text_color_threshold: float | None = None,
) -> None:
    masked = np.ma.masked_invalid(color_data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate_data is None:
        annotate_data = color_data

    if text_color_threshold is None:
        if vmin is not None and vmax is not None:
            text_color_threshold = (vmin + vmax) / 2.0
        else:
            text_color_threshold = float(np.nanmean(color_data))

    for r in range(color_data.shape[0]):
        for c in range(color_data.shape[1]):
            v_color = color_data[r, c]
            v_text = annotate_data[r, c]
            if np.isnan(v_color) or np.isnan(v_text):
                ax.text(c, r, "-", ha="center", va="center", color="gray", fontsize=10)
                continue
            txt_color = "white" if v_color > text_color_threshold else "black"
            ax.text(c, r, format(v_text, fmt), ha="center", va="center", color=txt_color, fontsize=10)

    if cbar_label is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, rotation=270, labelpad=16)


def plot_dashboard(out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # 1) Win count across datasets
    ds_to_data = {
        "LSUI": LSUI,
        "UIEB": UIEB,
        "U45": U45,
    }
    wins = []
    for ds_name, ds_data in ds_to_data.items():
        count = 0
        for metric, values in ds_data.items():
            if _ours_is_best(values, HIGHER_BETTER[metric]):
                count += 1
        wins.append(count)
    ax1.bar(list(ds_to_data.keys()), wins, color="#2a7ab0")
    ax1.set_ylim(0, max(wins) + 1)
    ax1.set_title("1. Ours Win Count by Dataset")
    ax1.set_ylabel("metric wins")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # 2) PSNR of selected methods on LSUI/UIEB
    selected = ["UIEC2Net", "LANet", "U-shape", "DM", "Ours"]
    idx = [METHODS.index(m) for m in selected]
    x = np.arange(len(selected))
    ax2.plot(x, [LSUI["PSNR"][i] for i in idx], marker="o", label="LSUI")
    ax2.plot(x, [UIEB["PSNR"][i] for i in idx], marker="s", label="UIEB")
    ax2.set_xticks(x)
    ax2.set_xticklabels(selected, rotation=20)
    ax2.set_title("2. PSNR on Strong Baselines")
    ax2.set_ylabel("PSNR")
    ax2.grid(linestyle="--", alpha=0.6)
    ax2.legend(frameon=True)

    # 3) PSNR distribution across methods + ours marker
    psnr_lsui_baseline = LSUI["PSNR"][:-1]
    psnr_uieb_baseline = UIEB["PSNR"][:-1]
    box = ax3.boxplot(
        [psnr_lsui_baseline, psnr_uieb_baseline],
        tick_labels=["LSUI", "UIEB"],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ["#9ecae1", "#fdae6b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.scatter([1, 2], [LSUI["PSNR"][-1], UIEB["PSNR"][-1]], marker="^", s=90, c="tab:green", label="Ours")
    ax3.set_title("3. PSNR Distribution (Baselines)")
    ax3.set_ylabel("PSNR")
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    ax3.legend(frameon=True, loc="lower right")

    # 4) LPIPS-FID scatter on LSUI
    lpips = np.array(LSUI["LPIPS"])
    fid = np.array(LSUI["FID"])
    ax4.scatter(lpips[:-1], fid[:-1], s=50, alpha=0.8, c="#4f81bd", label="Baselines")
    ax4.scatter(lpips[-1], fid[-1], s=160, marker="*", c="crimson", label="Ours")
    for i, m in enumerate(METHODS):
        if m in {"Ours", "DM", "U-shape", "UIEC2Net"}:
            ax4.annotate(m, (lpips[i], fid[i]), textcoords="offset points", xytext=(5, 4), fontsize=9)
    ax4.set_title("4. LSUI Perceptual Tradeoff")
    ax4.set_xlabel("LPIPS (lower better)")
    ax4.set_ylabel("FID (lower better)")
    ax4.grid(linestyle="--", alpha=0.6)
    ax4.legend(frameon=True)

    # 5) Inference step sensitivity (normalized)
    steps = np.array(ABLATION_STEPS["step"])
    for metric in ["UCIQE", "UIQM", "PSNR", "SSIM"]:
        vals = np.array(ABLATION_STEPS[metric], dtype=float)
        vals_n = (vals - vals.min()) / (vals.max() - vals.min())
        ax5.plot(steps, vals_n, marker="o", linewidth=2, label=metric)
    ax5.axvline(3, linestyle="--", color="gray", linewidth=1.2, alpha=0.8)
    ax5.set_xticks(steps)
    ax5.set_ylim(-0.05, 1.05)
    ax5.set_title("5. Step Sensitivity (Normalized)")
    ax5.set_xlabel("inference steps")
    ax5.set_ylabel("normalized score")
    ax5.grid(linestyle="--", alpha=0.6)
    ax5.legend(frameon=True, ncol=2, fontsize=9)

    # 6) Gain over best baseline (+ means better)
    cols = ["UCIQE", "UIQM", "PSNR", "SSIM", "LPIPS", "FID"]
    rows = ["LSUI", "UIEB", "U45"]
    gain = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for r, ds in enumerate(rows):
        ds_data = {"LSUI": LSUI, "UIEB": UIEB, "U45": U45}[ds]
        for c, metric in enumerate(cols):
            if metric not in ds_data:
                continue
            vals = ds_data[metric]
            best_base = _best_baseline(vals, HIGHER_BETTER[metric])
            ours = vals[-1]
            if HIGHER_BETTER[metric]:
                gain[r, c] = ours - best_base
            else:
                gain[r, c] = best_base - ours
    draw_heatmap(
        ax=ax6,
        color_data=gain,
        row_labels=rows,
        col_labels=cols,
        annotate_data=gain,
        fmt="+.2f",
        cmap="YlGnBu",
        cbar_label="gain vs best baseline (+ better)",
        text_color_threshold=np.nanmean(gain),
    )
    ax6.set_title("6. Ours Margin Heatmap")

    fig.suptitle("RF-MCUIE Quantitative Overview", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "result_dashboard.png", dpi=300)
    fig.savefig(out_dir / "result_dashboard.pdf")
    plt.close(fig)


def plot_heatmap_suite(out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (a) Dataset x Method (UIQM)
    methods_sel = ["UIEC2Net", "U-shape", "DM", "Ours"]
    rows = ["LSUI", "UIEB", "U45"]
    a = np.array(
        [
            [LSUI["UIQM"][METHODS.index(m)] for m in methods_sel],
            [UIEB["UIQM"][METHODS.index(m)] for m in methods_sel],
            [U45["UIQM"][METHODS.index(m)] for m in methods_sel],
        ],
        dtype=float,
    )
    draw_heatmap(
        ax=ax1,
        color_data=a,
        row_labels=rows,
        col_labels=methods_sel,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_label="UIQM (higher is better)",
    )
    ax1.set_title("(a) Dataset groups: UIQM")

    # (b) Inference steps x metrics (column-wise normalized colors, actual annotations)
    b_actual = np.array(
        [
            [ABLATION_STEPS["UCIQE"][i], ABLATION_STEPS["UIQM"][i], ABLATION_STEPS["PSNR"][i], ABLATION_STEPS["SSIM"][i]]
            for i in range(len(ABLATION_STEPS["step"]))
        ],
        dtype=float,
    )
    b_norm = _colwise_minmax_norm(b_actual)
    draw_heatmap(
        ax=ax2,
        color_data=b_norm,
        row_labels=[f"N={s}" for s in ABLATION_STEPS["step"]],
        col_labels=["UCIQE", "UIQM", "PSNR", "SSIM"],
        annotate_data=b_actual,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar_label="column-wise normalized intensity",
        text_color_threshold=0.55,
    )
    ax2.set_title("(b) Step sensitivity (ablation)")

    # (c) Module ablation x metrics
    c_actual = np.array(
        [
            [ABLATION_MODELS["UCIQE"][i], ABLATION_MODELS["UIQM"][i], ABLATION_MODELS["PSNR"][i], ABLATION_MODELS["SSIM"][i]]
            for i in range(len(ABLATION_MODELS["model"]))
        ],
        dtype=float,
    )
    c_norm = _colwise_minmax_norm(c_actual)
    draw_heatmap(
        ax=ax3,
        color_data=c_norm,
        row_labels=ABLATION_MODELS["model"],
        col_labels=["UCIQE", "UIQM", "PSNR", "SSIM"],
        annotate_data=c_actual,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_label="column-wise normalized intensity",
        text_color_threshold=0.55,
    )
    ax3.set_title("(c) Module contribution matrix")

    # (d) Relative gain (%) over best baseline
    cols = ["UCIQE", "UIQM", "PSNR", "SSIM", "LPIPS", "FID"]
    rows = ["LSUI", "UIEB", "U45"]
    d = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for r, ds_name in enumerate(rows):
        ds_data = {"LSUI": LSUI, "UIEB": UIEB, "U45": U45}[ds_name]
        for c, metric in enumerate(cols):
            if metric not in ds_data:
                continue
            vals = ds_data[metric]
            ours = vals[-1]
            best_base = _best_baseline(vals, HIGHER_BETTER[metric])
            if HIGHER_BETTER[metric]:
                d[r, c] = (ours - best_base) / (abs(best_base) + 1e-12) * 100.0
            else:
                d[r, c] = (best_base - ours) / (abs(best_base) + 1e-12) * 100.0

    vmax = np.nanmax(np.abs(d))
    draw_heatmap(
        ax=ax4,
        color_data=d,
        row_labels=rows,
        col_labels=cols,
        annotate_data=d,
        fmt="+.1f",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
        cbar_label="relative gain (%) vs best baseline",
        text_color_threshold=0.0,
    )
    ax4.set_title("(d) Relative margin heatmap")

    fig.suptitle("RF-MCUIE Heatmap Analysis", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "result_heatmaps.png", dpi=300)
    fig.savefig(out_dir / "result_heatmaps.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts for experiment results.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fig"),
        help="Output directory for generated figures (default: fig).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    plot_dashboard(out_dir)
    plot_heatmap_suite(out_dir)

    for name in [
        "result_dashboard.png",
        "result_dashboard.pdf",
        "result_heatmaps.png",
        "result_heatmaps.pdf",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
