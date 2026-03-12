#!/usr/bin/env python3

"""Generate schematic and real-sample rectified-flow trajectory figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_IMAGE = ROOT_DIR / "input image" / "input.jpg"
OUTPUT_IMAGE = ROOT_DIR / "input image" / "output.png"
SCHEMATIC_OUT = SCRIPT_DIR / "rf_ode_trajectories.pdf"
REAL_CASE_OUT = SCRIPT_DIR / "rf_ode_real_case.pdf"


def make_base_scene(h: int = 128, w: int = 128) -> np.ndarray:
    """Create a synthetic RGB scene in [0, 1]."""
    y, x = np.mgrid[0:h, 0:w]
    y = y / (h - 1)
    x = x / (w - 1)

    bg = 0.4 + 0.6 * (0.5 * x + 0.5 * (1 - y))
    img = np.stack([bg, bg, bg], axis=-1).astype(np.float32)

    centers = [(0.35, 0.35), (0.65, 0.45), (0.50, 0.75)]
    colors = np.array(
        [[0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.2, 0.9]],
        dtype=np.float32,
    )
    sigmas = [0.08, 0.06, 0.07]

    for (cy, cx), col, sigma in zip(centers, colors, sigmas):
        gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        img += gauss[..., None] * col[None, None, :]

    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img**0.9


def apply_condition_tint(img: np.ndarray, tint: np.ndarray) -> np.ndarray:
    """Apply an RGB tint multiplier and renormalize."""
    out = img * tint[None, None, :]
    return out / (out.max() + 1e-8)


def interpolate_noise_to_target(
    target: np.ndarray, t: float, rng: np.random.Generator
) -> np.ndarray:
    """Straight-line transport between Gaussian noise and a target image."""
    noise = rng.standard_normal(target.shape).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    xt = (1.0 - t) * noise + t * target
    return np.clip(xt, 0.0, 1.0)


def simulate_trajectories(
    n_particles: int = 30, n_steps: int = 40, cond: int = 0, seed: int = 0
) -> np.ndarray:
    """Toy linear transport trajectories in R^2."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n_particles, 2))

    if cond == 0:
        mean = np.array([2.0, 1.2])
        transform = np.array([[1.15, 0.25], [0.05, 0.95]])
    elif cond == 1:
        mean = np.array([-2.2, 1.6])
        transform = np.array([[0.85, -0.25], [0.20, 1.10]])
    else:
        mean = np.array([0.3, -2.2])
        transform = np.array([[1.00, 0.35], [-0.25, 0.75]])

    t_grid = np.linspace(0.0, 1.0, n_steps)
    traj = np.zeros((n_steps, n_particles, 2), dtype=np.float32)
    for idx, t in enumerate(t_grid):
        traj[idx] = (1.0 - t) * x0 + t * (x0 @ transform.T + mean)
    return traj


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1]."""
    img = Image.open(path)
    if "A" in img.getbands():
        rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        rgb = rgba[..., :3]
    else:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def resize_to_match(image: np.ndarray, ref_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize an RGB image to the reference HWC shape."""
    height, width = ref_shape[:2]
    pil = Image.fromarray(np.round(image * 255.0).astype(np.uint8))
    pil = pil.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0


def blend_states(
    degraded: np.ndarray, restored: np.ndarray, times: list[float]
) -> list[np.ndarray]:
    """Create deterministic intermediate states by linear interpolation."""
    states = []
    for t in times:
        xt = (1.0 - t) * degraded + t * restored
        states.append(np.clip(xt, 0.0, 1.0))
    return states


def compute_statistics(image: np.ndarray) -> tuple[float, float]:
    """Compute the 2D trajectory coordinates for a state image."""
    red_mean = float(image[..., 0].mean())
    blue_mean = float(image[..., 2].mean())
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    cast_index = blue_mean - red_mean
    contrast_index = float(gray.std())
    return cast_index, contrast_index


def generate_schematic_figure(out_path: Path) -> None:
    """Generate the original schematic ODE transport figure."""
    base = make_base_scene()
    tints = [
        np.array([0.95, 1.00, 1.10], dtype=np.float32),
        np.array([0.95, 1.08, 0.95], dtype=np.float32),
        np.array([1.08, 1.00, 0.92], dtype=np.float32),
    ]
    cond_names = ["Cond-A (Blue/clear)", "Cond-B (Green)", "Cond-C (Turbid/warm)"]
    times = [0.25, 0.50, 0.75]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))

    for row in range(3):
        ax = axes[row, 0]
        traj = simulate_trajectories(cond=row, seed=10 + row)
        for particle in range(traj.shape[1]):
            ax.plot(traj[:, particle, 0], traj[:, particle, 1], linewidth=0.8, alpha=0.6)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=8, marker="o", alpha=0.8, label="t=0")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=10, marker="x", alpha=0.9, label="t=1")
        ax.set_title(cond_names[row], fontsize=10)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        if row == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)

        target = apply_condition_tint(base, tints[row])
        noise_rng = np.random.default_rng(1000 + row)
        for col, t in enumerate(times, start=1):
            ax_img = axes[row, col]
            xt = interpolate_noise_to_target(target, t, noise_rng)
            ax_img.imshow(xt)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if row == 0:
                ax_img.set_title(f"$t={t:.2f}$", fontsize=10)

    axes[0, 0].set_title("Latent ODE trajectories\n(transport $\\pi_0\\!\\to\\!\\pi_1$)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_real_case_figure(out_path: Path) -> None:
    """Generate a real-sample trajectory and state progression figure."""
    degraded = load_rgb_image(INPUT_IMAGE)
    restored = load_rgb_image(OUTPUT_IMAGE)
    if restored.shape != degraded.shape:
        restored = resize_to_match(restored, degraded.shape)

    times = [0.00, 0.25, 0.50, 0.75, 1.00]
    states = blend_states(degraded, restored, times)
    stats = np.array([compute_statistics(state) for state in states], dtype=np.float32)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=6,
        figsize=(14, 3.4),
        gridspec_kw={"width_ratios": [1.25, 1, 1, 1, 1, 1]},
    )

    ax_traj = axes[0]
    ax_traj.plot(stats[:, 0], stats[:, 1], color="#1f77b4", linewidth=1.8, zorder=1)
    ax_traj.scatter(stats[:, 0], stats[:, 1], color="#d62728", s=28, zorder=2)
    for idx in range(len(times) - 1):
        ax_traj.annotate(
            "",
            xy=(stats[idx + 1, 0], stats[idx + 1, 1]),
            xytext=(stats[idx, 0], stats[idx, 1]),
            arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.1},
        )
    for t, (x_val, y_val) in zip(times, stats):
        ax_traj.text(x_val + 0.0025, y_val + 0.0015, f"t={t:.2f}", fontsize=8)

    x_pad = max(0.01, float(np.ptp(stats[:, 0])) * 0.25)
    y_pad = max(0.01, float(np.ptp(stats[:, 1])) * 0.25)
    ax_traj.set_xlim(float(stats[:, 0].min()) - x_pad, float(stats[:, 0].max()) + x_pad)
    ax_traj.set_ylim(float(stats[:, 1].min()) - y_pad, float(stats[:, 1].max()) + y_pad)
    ax_traj.set_title("Real sample trajectory", fontsize=10)
    ax_traj.set_xlabel("color-cast index (mean(B)-mean(R))", fontsize=9)
    ax_traj.set_ylabel("contrast index (gray std)", fontsize=9)
    ax_traj.grid(True, linewidth=0.3, alpha=0.4)

    titles = ["Input", "t=0.25", "t=0.50", "t=0.75", "Output"]
    for ax_img, title, state in zip(axes[1:], titles, states):
        ax_img.imshow(state)
        ax_img.set_title(title, fontsize=10)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    generate_schematic_figure(SCHEMATIC_OUT)
    generate_real_case_figure(REAL_CASE_OUT)
    print(f"[OK] Saved: {SCHEMATIC_OUT.resolve()}")
    print(f"[OK] Saved: {REAL_CASE_OUT.resolve()}")


if __name__ == "__main__":
    main()
