
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# Method-name typography tuning
METHOD_LABEL_SCATTER_FS = 8.5
METHOD_LABEL_TICK_FS = 11
METHOD_LABEL_BAR_FS = 10.5

# -----------------------------
# Data (copied from your tables)
# -----------------------------
lsui_data = {
    "Method": ["ICSP","ZAP","UIEC2Net","LANet","Shallow","UIE-WD","Five-a+","U-shape","DM-water","Ours"],
    "UCIQE": [0.55,0.50,0.60,0.58,0.53,0.54,0.59,0.56,0.57,0.59],
    "UIQM":  [2.45,2.51,3.06,2.97,2.69,2.63,2.94,3.08,2.90,2.98],
    "PSNR":  [11.99,18.56,22.96,21.64,19.00,14.19,22.38,24.83,25.12,24.91],
    "SSIM":  [0.55,0.79,0.86,0.86,0.79,0.69,0.86,0.87,0.88,0.90],
    "LPIPS": [0.39,0.21,0.11,0.11,0.18,0.24,0.11,0.10,0.09,0.06],
    "FID":   [92.12,62.35,40.42,37.58,48.37,79.55,37.78,31.49,32.54,26.69],
}
uieb_data = {
    "Method": ["ICSP","ZAP","UIEC2Net","LANet","Shallow","UIE-WD","Five-a+","U-shape","DM-water","Ours"],
    "UCIQE": [0.54,0.49,0.61,0.57,0.53,0.56,0.60,0.57,0.57,0.58],
    "UIQM":  [2.35,2.44,3.05,2.95,2.61,2.38,2.90,3.10,2.86,2.91],
    "PSNR":  [11.71,16.54,23.82,21.56,16.83,12.56,22.30,22.17,21.64,23.18],
    "SSIM":  [0.53,0.74,0.87,0.85,0.72,0.62,0.86,0.82,0.82,0.87],
    "LPIPS": [0.37,0.24,0.07,0.08,0.19,0.26,0.08,0.12,0.10,0.06],
    "FID":   [122.35,62.65,30.01,29.95,59.68,104.32,36.60,38.91,33.08,28.78],
}
u45_data = {
    "Method": ["ICSP","ZAP","UIEC2Net","LANet","Shallow","UIE-WD","Five-a+","U-shape","DM-water","Ours"],
    "UCIQE": [0.53,0.49,0.61,0.63,0.53,0.54,0.60,0.55,0.60,0.61],
    "UIQM":  [1.98,2.79,3.40,3.29,2.80,1.97,3.34,3.26,3.26,3.32],
}

ablation_steps = pd.DataFrame({
    "UCIQE":[0.56,0.59,0.61,0.57],
    "UIQM":[2.90,2.98,2.92,3.01],
    "PSNR":[24.13,24.91,24.87,24.65],
    "SSIM":[0.88,0.90,0.88,0.86],
}, index=["N=1","N=3","N=5","N=10"])

ablation_modules = pd.DataFrame({
    "UCIQE":[0.55,0.58,0.57,0.59],
    "UIQM":[2.84,2.92,2.90,2.98],
    "PSNR":[24.12,24.37,25.05,24.91],
    "SSIM":[0.84,0.86,0.91,0.90],
}, index=["Model I","Model II","Model III","Model IV"])

df_lsui = pd.DataFrame(lsui_data).set_index("Method")
df_uieb = pd.DataFrame(uieb_data).set_index("Method")
df_u45  = pd.DataFrame(u45_data).set_index("Method")

# -----------------------------
# Helpers
# -----------------------------
def normalize_df(df, higher_better_cols, lower_better_cols):
    """Column-wise min-max normalization; map 'best' to 1 and 'worst' to 0."""
    norm = df.copy().astype(float)
    for col in df.columns:
        col_vals = df[col].astype(float)
        vmin, vmax = col_vals.min(), col_vals.max()
        if math.isclose(vmax, vmin):
            norm[col] = 0.5
            continue
        if col in higher_better_cols:
            norm[col] = (col_vals - vmin) / (vmax - vmin)
        elif col in lower_better_cols:
            norm[col] = (vmax - col_vals) / (vmax - vmin)
        else:
            norm[col] = (col_vals - vmin) / (vmax - vmin)
    return norm

def plot_normalized_heatmap(ax, norm_df, raw_df, title, cmap="YlGnBu", fmt="{:.2f}", highlight_row=None):
    """Draw a heatmap colored by normalized score and annotated with raw values."""
    data = norm_df.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(norm_df.shape[1]))
    ax.set_yticks(np.arange(norm_df.shape[0]))
    ax.set_xticklabels(norm_df.columns, rotation=30, ha='right')
    ax.set_yticklabels(norm_df.index, fontsize=METHOD_LABEL_TICK_FS)
    ax.set_title(title, pad=8)

    # white grid lines
    ax.set_xticks(np.arange(-.5, norm_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, norm_df.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # cell text
    for i in range(norm_df.shape[0]):
        for j in range(norm_df.shape[1]):
            v_raw = raw_df.iloc[i, j]
            v_norm = data[i, j]
            color = 'white' if v_norm > 0.65 else 'black'
            ax.text(j, i, fmt.format(v_raw), ha='center', va='center', color=color, fontsize=8)

    # highlight a row (optional)
    if highlight_row is not None and highlight_row in norm_df.index:
        r = list(norm_df.index).index(highlight_row)
        rect = Rectangle((-0.5, r-0.5), norm_df.shape[1], 1, fill=False, edgecolor='black', linewidth=2.5)
        ax.add_patch(rect)

    return im

def scatter_tradeoff(
    ax,
    df,
    x_col,
    y_col,
    title,
    invert_y=False,
    highlight="Ours",
    custom_offsets=None,
    custom_text_positions=None,
):
    """Scatter + labels; optionally invert y-axis (useful for LPIPS/FID)."""
    if custom_offsets is None:
        custom_offsets = {}
    if custom_text_positions is None:
        custom_text_positions = {}

    ax.scatter(df[x_col], df[y_col], s=50)
    if highlight in df.index:
        ax.scatter([df.loc[highlight, x_col]], [df.loc[highlight, y_col]],
                   s=160, marker='*', edgecolor='black', linewidths=1.0, zorder=5)

    # Expand plot window to reduce label clipping on borders.
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_pad = max((x_max - x_min) * 0.08, 0.01)
    y_pad = max((y_max - y_min) * 0.12, 0.01)
    xl = x_min - x_pad
    xr = x_max + x_pad
    yb = y_min - y_pad
    yt = y_max + y_pad
    ax.set_xlim(xl, xr)
    if invert_y:
        ax.set_ylim(yt, yb)
    else:
        ax.set_ylim(yb, yt)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_margin = max((xr - xl) * 0.02, 0.005)
    y_margin = max((yt - yb) * 0.03, 0.005)
    for m in df.index.tolist():
        x = df.loc[m, x_col]
        y = df.loc[m, y_col]

        if m in custom_text_positions:
            tx, ty = custom_text_positions[m]
            tx = min(max(tx, xl + x_margin), xr - x_margin)
            ty = min(max(ty, yb + y_margin), yt - y_margin)
        else:
            # Fallback: push labels inward from borders.
            tx = x + (0.04 if x < x_mid else -0.04) * (xr - xl)
            if invert_y:
                ty = y + (0.04 if y > y_mid else -0.04) * (yt - yb)
            else:
                ty = y + (0.04 if y < y_mid else -0.04) * (yt - yb)
            tx = min(max(tx, xl + x_margin), xr - x_margin)
            ty = min(max(ty, yb + y_margin), yt - y_margin)

        if m in custom_offsets:
            dx, dy = custom_offsets[m]
            tx += dx * (xr - xl) / 1000.0
            ty += dy * (yt - yb) / 1000.0
            tx = min(max(tx, xl + x_margin), xr - x_margin)
            ty = min(max(ty, yb + y_margin), yt - y_margin)

        ha = "left" if tx >= x else "right"
        va = "bottom" if (ty >= y) ^ invert_y else "top"
        ax.annotate(m, (x, y),
                    textcoords="data", xytext=(tx, ty), fontsize=METHOD_LABEL_SCATTER_FS,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
                    ha=ha, va=va,
                    annotation_clip=True)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, pad=6)
    ax.grid(True, linestyle='--', alpha=0.4)
    if invert_y:
        ax.invert_yaxis()

# -----------------------------
# Figure 1: Quantitative heatmaps (LSUI / UIEB / U45)
# -----------------------------
higher_better = ["UCIQE","UIQM","PSNR","SSIM"]
lower_better = ["LPIPS","FID"]

norm_lsui = normalize_df(df_lsui, higher_better, lower_better)
norm_uieb = normalize_df(df_uieb, higher_better, lower_better)
norm_u45  = normalize_df(df_u45, ["UCIQE","UIQM"], [])

fig = plt.figure(figsize=(16,6))
gs = gridspec.GridSpec(1, 3, width_ratios=[6,6,2], wspace=0.25, figure=fig)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

im1 = plot_normalized_heatmap(ax1, norm_lsui, df_lsui, "(a) LSUI", highlight_row="Ours")
im2 = plot_normalized_heatmap(ax2, norm_uieb, df_uieb, "(b) UIEB", highlight_row="Ours")
im3 = plot_normalized_heatmap(ax3, norm_u45,  df_u45,  "(c) U45",  highlight_row="Ours")

cbar = fig.colorbar(im2, ax=[ax1,ax2,ax3], fraction=0.025, pad=0.02)
cbar.set_label("Normalized score (higher is better)")
fig.savefig("fig_quant_heatmap.png", dpi=300, bbox_inches='tight')
fig.savefig("fig_quant_heatmap.pdf", bbox_inches='tight')
plt.close(fig)

# -----------------------------
# Figure 2: Ablation heatmaps (steps / modules)
# -----------------------------
norm_ab_steps   = normalize_df(ablation_steps,   ["UCIQE","UIQM","PSNR","SSIM"], [])
norm_ab_modules = normalize_df(ablation_modules, ["UCIQE","UIQM","PSNR","SSIM"], [])

fig = plt.figure(figsize=(12,4.5))
gs = gridspec.GridSpec(1, 2, wspace=0.35, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

im1 = plot_normalized_heatmap(ax1, norm_ab_steps, ablation_steps, "(a) Inference steps", highlight_row="N=3")
im2 = plot_normalized_heatmap(ax2, norm_ab_modules, ablation_modules, "(b) Module ablation", highlight_row="Model IV")

cbar = fig.colorbar(im2, ax=[ax1,ax2], fraction=0.03, pad=0.04)
cbar.set_label("Normalized score (higher is better)")
fig.savefig("fig_ablation_heatmaps.png", dpi=300, bbox_inches='tight')
fig.savefig("fig_ablation_heatmaps.pdf", bbox_inches='tight')
plt.close(fig)

# -----------------------------
# Figure 3: Metric trade-off scatter plots
# -----------------------------
fig, axes = plt.subplots(2,2, figsize=(14.0,9.0))
scatter_tradeoff(
    axes[0,0], df_lsui, "PSNR", "LPIPS", "(a) LSUI: PSNR vs LPIPS (↓)", invert_y=True,
    custom_text_positions={
        "ICSP": (12.6, 0.385),
        "ZAP": (18.3, 0.205),
        "Shallow": (18.9, 0.18),
        "UIE-WD": (14.4, 0.24),
        "LANet": (20.80, 0.114),
        "Five-a+": (21.55, 0.095),
        "UIEC2Net": (23.25, 0.114),
        "U-shape": (25.30, 0.097),
        "DM-water": (25.30, 0.078),
        "Ours": (25.10, 0.050),
    },
)
axes[0,0].set_ylabel("LPIPS (lower is better)")
scatter_tradeoff(
    axes[0,1], df_lsui, "UCIQE", "UIQM", "(b) LSUI: UCIQE vs UIQM",
    custom_text_positions={
        "ICSP": (0.555, 2.47),
        "ZAP": (0.505, 2.52),
        "Shallow": (0.533, 2.70),
        "UIE-WD": (0.545, 2.64),
        "LANet": (0.573, 2.95),
        "UIEC2Net": (0.587, 3.06),
        "U-shape": (0.563, 3.05),
        "DM-water": (0.571, 2.915),
        "Five-a+": (0.588, 2.965),
        "Ours": (0.588, 2.99),
    },
)
scatter_tradeoff(
    axes[1,0], df_uieb, "PSNR", "LPIPS", "(c) UIEB: PSNR vs LPIPS (↓)", invert_y=True,
    custom_text_positions={
        "ICSP": (12.0, 0.368),
        "ZAP": (16.8, 0.23),
        "Shallow": (17.1, 0.195),
        "UIE-WD": (12.7, 0.258),
        "LANet": (20.40, 0.074),
        "Five-a+": (21.40, 0.071),
        "DM-water": (21.20, 0.097),
        "U-shape": (22.55, 0.121),
        "UIEC2Net": (23.25, 0.078),
        "Ours": (22.95, 0.048),
    },
)
axes[1,0].set_ylabel("LPIPS (lower is better)")
scatter_tradeoff(
    axes[1,1], df_uieb, "UCIQE", "UIQM", "(d) UIEB: UCIQE vs UIQM",
    custom_text_positions={
        "ICSP": (0.544, 2.37),
        "ZAP": (0.489, 2.46),
        "Shallow": (0.532, 2.63),
        "UIE-WD": (0.557, 2.39),
        "LANet": (0.566, 2.945),
        "DM-water": (0.569, 2.885),
        "U-shape": (0.571, 3.07),
        "UIEC2Net": (0.607, 3.055),
        "Five-a+": (0.598, 2.93),
        "Ours": (0.579, 2.92),
    },
)

plt.tight_layout()
fig.savefig("fig_tradeoff_scatter.png", dpi=300, bbox_inches='tight')
fig.savefig("fig_tradeoff_scatter.pdf", bbox_inches='tight')
plt.close(fig)

print("Done. Generated: fig_quant_heatmap.*, fig_ablation_heatmaps.*, fig_tradeoff_scatter.*")


# -----------------------------
# Figure 4: Perceptual metrics bars (LPIPS / FID)
# -----------------------------
def perceptual_bar_subplot(ax, df, metric, title, highlight="Ours"):
    sdf = df[[metric]].sort_values(metric, ascending=True)  # lower is better
    y = np.arange(len(sdf))
    ax.barh(y, sdf[metric].values)
    ax.set_yticks(y)
    ax.set_yticklabels(sdf.index, fontsize=METHOD_LABEL_BAR_FS)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title, pad=6)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    if highlight in sdf.index:
        idx = list(sdf.index).index(highlight)
        bar = ax.patches[idx]
        bar.set_edgecolor('black')
        bar.set_linewidth(2)
        ax.scatter([sdf.loc[highlight, metric]], [idx], marker='*', s=120, edgecolor='black', zorder=5)

fig, axes = plt.subplots(2,2, figsize=(12,8))
perceptual_bar_subplot(axes[0,0], df_lsui, "LPIPS", "(a) LSUI: LPIPS (lower is better)")
perceptual_bar_subplot(axes[0,1], df_lsui, "FID",   "(b) LSUI: FID (lower is better)")
perceptual_bar_subplot(axes[1,0], df_uieb, "LPIPS", "(c) UIEB: LPIPS (lower is better)")
perceptual_bar_subplot(axes[1,1], df_uieb, "FID",   "(d) UIEB: FID (lower is better)")

plt.tight_layout()
fig.savefig("fig_perceptual_bars.png", dpi=300, bbox_inches='tight')
fig.savefig("fig_perceptual_bars.pdf", bbox_inches='tight')
plt.close(fig)

print("Done. Generated: fig_quant_heatmap.*, fig_ablation_heatmaps.*, fig_tradeoff_scatter.*, fig_perceptual_bars.*")
