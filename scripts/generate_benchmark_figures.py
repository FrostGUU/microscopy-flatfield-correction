"""
Reproducible Benchmark Figure & QC Table Generator (`generate_benchmark_figures.py`)
------------------------------------------------------------------------------------
Processes the bundled anonymized brightfield microscopy sample images (`assets/examples/*_raw.png`)
using the refactored Mode B (`pseudo-flatfield`, `bg_method='masked_gaussian'`, shared luminance gain)
pipeline, saves the corrected images and their QC JSON reports, and generates:
  - `assets/flatfield_comparison_overview.png`
  - `reports/real_samples_qc_summary.json`

All numbers cited in `README.md` are produced directly by this script.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from correct_illumination import correct_image_pipeline
from io_utils import write_image_and_metadata

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def get_diagonal_profile(arr: np.ndarray, window: int = 16) -> tuple:
    """Compute local median luminance along the main diagonal (Top-Left to Bottom-Right)."""
    lum = arr.mean(axis=2).astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
    h, w = lum.shape
    n_pts = 120
    ys = np.linspace(0, h - 1, n_pts).astype(int)
    xs = np.linspace(0, w - 1, n_pts).astype(int)
    prof = []
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - window), min(h, y + window + 1)
        x0, x1 = max(0, x - window), min(w, x + window + 1)
        prof.append(float(np.median(lum[y0:y1, x0:x1])))
    return np.linspace(0, 100, n_pts), np.array(prof)


def generate_all():
    assets_dir = os.path.join(REPO_ROOT, "assets")
    examples_dir = os.path.join(assets_dir, "examples")
    reports_dir = os.path.join(REPO_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    samples = [
        {
            "id": "sample1_4x",
            "label": "Sample 1 (4x Brightfield)",
            "sigma_px": 75.0,
            "raw_path": os.path.join(examples_dir, "sample1_4x_raw.png"),
            "corr_path": os.path.join(examples_dir, "sample1_4x_corrected.png"),
        },
        {
            "id": "sample2_4x",
            "label": "Sample 2 (4x Brightfield)",
            "sigma_px": 75.0,
            "raw_path": os.path.join(examples_dir, "sample2_4x_raw.png"),
            "corr_path": os.path.join(examples_dir, "sample2_4x_corrected.png"),
        },
        {
            "id": "sample3_20x",
            "label": "Sample 3 (20x Brightfield)",
            "sigma_px": 96.0,
            "raw_path": os.path.join(examples_dir, "sample3_20x_raw.png"),
            "corr_path": os.path.join(examples_dir, "sample3_20x_corrected.png"),
        },
    ]

    summary_records = {}
    plot_items = []

    for s in samples:
        res = correct_image_pipeline(
            sample=s["raw_path"],
            mode="pseudo-flatfield",
            sigma_px=s["sigma_px"],
            bg_method="masked_gaussian",
            per_channel=False,
            target_val=None,  # Native median background preservation
        )
        out_arr = res["output_array"]
        report = res["report"]
        report["output_path"] = s["corr_path"]
        write_image_and_metadata(s["corr_path"], out_arr, metadata=report)

        unif_b = report["background_uniformity_independent_grid"]["before"]
        unif_a = report["background_uniformity_independent_grid"]["after"]
        ch_b = report["channel_ratio_analysis"]["background_region"]
        ch_f = report["channel_ratio_analysis"]["foreground_region"]

        summary_records[s["id"]] = {
            "label": s["label"],
            "shape": report["input_summary"]["shape"],
            "dtype": report["input_summary"]["dtype"],
            "sigma_px": s["sigma_px"],
            "target_median_used": round(report["parameters"]["target_values_used"][0], 2),
            "before_center_median": unif_b["center_block_median"],
            "before_corner_min_median": unif_b["corner_min_median"],
            "before_corner_drop_pct": unif_b["corner_max_drop_pct"],
            "before_grid_cv": unif_b["cv"],
            "before_max_min_ratio": unif_b["max_min_ratio"],
            "after_center_median": unif_a["center_block_median"],
            "after_corner_min_median": unif_a["corner_min_median"],
            "after_corner_drop_pct": unif_a["corner_max_drop_pct"],
            "after_grid_cv": unif_a["cv"],
            "after_max_min_ratio": unif_a["max_min_ratio"],
            "cv_reduction_pct": report["background_uniformity_independent_grid"]["cv_reduction_pct"],
            "clipped_total_fraction_pct": round(report["output_summary"]["clipped_total_fraction"] * 100.0, 4),
            "gain_min_median_max": [
                report["gain_distribution"]["min"],
                report["gain_distribution"]["median"],
                report["gain_distribution"]["max"],
            ],
            "bg_channel_ratio_max_shift_pct": ch_b["max_ratio_rel_shift_pct"] if ch_b else 0.0,
            "fg_channel_ratio_max_shift_pct": ch_f["max_ratio_rel_shift_pct"] if ch_f else 0.0,
            "warning_codes": [w["code"] for w in report["warnings"]],
        }

        diag_x, raw_prof = get_diagonal_profile(res["loaded"].data)
        _, corr_prof = get_diagonal_profile(out_arr)
        plot_items.append({
            "meta": s,
            "rec": summary_records[s["id"]],
            "raw_rgb": res["loaded"].data,
            "corr_rgb": out_arr,
            "bg_est": res["bg_estimate"],
            "diag_x": diag_x,
            "raw_prof": raw_prof,
            "corr_prof": corr_prof,
        })

    summary_json_path = os.path.join(reports_dir, "real_samples_qc_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2, ensure_ascii=False)

    # Render 3x4 overview figure
    fig = plt.figure(figsize=(18.5, 13.2), dpi=160, facecolor="#F8FAFC")
    gs = GridSpec(
        3, 4,
        figure=fig,
        width_ratios=[1.08, 1.08, 1.08, 1.12],
        height_ratios=[1, 1, 1],
        wspace=0.16,
        hspace=0.33,
        left=0.025, right=0.98, top=0.905, bottom=0.05
    )

    fig.suptitle(
        "Mode B Pseudo-Flat-Field Illumination Correction — Visual Benchmark & Independent QC",
        fontsize=16.5, fontweight="bold", color="#0F172A", y=0.965
    )
    fig.text(
        0.5, 0.933,
        "Foreground-masked background estimation (masked_gaussian) + shared luminance gain (preserves RGB channel ratios; for visual uniformity, not instrument calibration)",
        ha="center", fontsize=10.5, color="#475569"
    )

    for row_idx, item in enumerate(plot_items):
        s = item["meta"]
        rec = item["rec"]
        raw_rgb = item["raw_rgb"]
        corr_rgb = item["corr_rgb"]
        bg_est = item["bg_est"]
        h, w, _ = raw_rgb.shape

        # Col 0: Raw Image
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(raw_rgb)
        ax0.plot([0, w - 1], [0, h - 1], color="#F59E0B", linestyle="--", linewidth=1.4, alpha=0.9)
        ax0.set_title(
            f"BEFORE — {s['label']}\nGrid CV: {rec['before_grid_cv']:.4f} | Corner Drop: -{rec['before_corner_drop_pct']:.1f}%",
            fontsize=9.8, fontweight="bold", color="#B91C1C", pad=7
        )
        ax0.text(
            0.03, 0.95, f"Center Med: {rec['before_center_median']:.0f} | Corner Min: {rec['before_corner_min_median']:.0f}",
            transform=ax0.transAxes, fontsize=8.5, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E293B", alpha=0.85, edgecolor="none")
        )
        ax0.set_xticks([])
        ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_edgecolor("#EF4444")
            spine.set_linewidth(1.6)

        # Col 1: Estimated Background Surface
        ax1 = fig.add_subplot(gs[row_idx, 1])
        vmin = max(90.0, float(np.percentile(bg_est, 1)) - 5.0)
        vmax = min(245.0, float(np.percentile(bg_est, 99)) + 5.0)
        im1 = ax1.imshow(bg_est, cmap="inferno", vmin=vmin, vmax=vmax)
        X, Y = np.meshgrid(np.arange(0, w, 8), np.arange(0, h, 8))
        Z = bg_est[::8, ::8]
        cs = ax1.contour(X, Y, Z, levels=5, colors="white", alpha=0.65, linewidths=0.85)
        ax1.clabel(cs, inline=True, fontsize=7.5, fmt="%.0f")
        ax1.set_title(
            f"Foreground-Masked Background Surface\nmasked_gaussian (σ_px = {s['sigma_px']:.0f} px)",
            fontsize=9.8, fontweight="bold", color="#334155", pad=7
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        cax = inset_axes(ax1, width="42%", height="5.5%", loc="lower right", borderpad=1.2)
        cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7, colors="white", length=2, pad=1)
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(0.8)

        # Col 2: Corrected Image
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.imshow(corr_rgb)
        ax2.plot([0, w - 1], [0, h - 1], color="#10B981", linestyle="--", linewidth=1.4, alpha=0.9)
        ax2.set_title(
            f"AFTER — Mode B Pseudo-Flat-Field\nGrid CV: {rec['after_grid_cv']:.4f} | Corner Drop: {rec['after_corner_drop_pct']:.1f}%",
            fontsize=9.8, fontweight="bold", color="#047857", pad=7
        )
        ax2.text(
            0.03, 0.95, f"Center Med: {rec['after_center_median']:.0f} | Corner Min: {rec['after_corner_min_median']:.0f}",
            transform=ax2.transAxes, fontsize=8.5, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#047857", alpha=0.88, edgecolor="none")
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor("#10B981")
            spine.set_linewidth(1.6)

        # Col 3: Diagonal Profile
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.set_facecolor("#FFFFFF")
        t_used = rec["target_median_used"]
        ax3.axhline(t_used, color="#64748B", linestyle=":", linewidth=1.3, label=f"Native Median ({t_used:.0f})")
        ax3.plot(item["diag_x"], item["raw_prof"], color="#EF4444", linewidth=2.0, alpha=0.88, label="Before (Raw)")
        ax3.plot(item["diag_x"], item["corr_prof"], color="#10B981", linewidth=2.1, alpha=0.94, label="After (Mode B)")
        ax3.set_title(
            "Diagonal Local Median Luminance\n(Top-Left → Center → Bottom-Right)",
            fontsize=9.8, fontweight="bold", color="#1E293B", pad=7
        )
        ax3.set_xlabel("Diagonal Position (%)", fontsize=8.5, color="#475569")
        ax3.set_ylabel("Luminance (0–255)", fontsize=8.5, color="#475569")
        ax3.set_xlim(0, 100)
        y_min = min(105.0, float(item["raw_prof"].min()) - 12.0)
        ax3.set_ylim(y_min, 245.0)
        ax3.grid(True, linestyle="--", alpha=0.38, color="#CBD5E1")
        ax3.tick_params(labelsize=8)
        ax3.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#E2E8F0")

    overview_path = os.path.join(assets_dir, "flatfield_comparison_overview.png")
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)
    return summary_records


if __name__ == "__main__":
    records = generate_all()
    print(json.dumps(records, indent=2, ensure_ascii=False))
