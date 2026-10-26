"""
Synthetic Benchmark Comparison: Legacy vs. New Mode B & Mode A
--------------------------------------------------------------
Evaluates Legacy (`legacy_pseudo_flatfield_correction`), New Mode B (`gaussian` and
`masked_gaussian`), and New Mode A (`flatfield`) on identical synthetic datasets with
known ground-truth object transmittance, known non-Gaussian illumination fields
(cos^4 vignetting and tilted plane), multi-scale absorbing structures (small, medium,
large discs + stripe), 16-bit dynamic range, and Poisson photon noise.

Saves machine-readable results to `reports/synthetic_benchmark_old_vs_new.json`
and human-readable markdown tables to `reports/SYNTHETIC_BENCHMARK_OLD_VS_NEW.md`.
"""

import os
import sys
import json
import tempfile
import numpy as np
import tifffile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from correct_illumination import (
    correct_image_pipeline,
    legacy_imread_unicode,
    legacy_pseudo_flatfield_correction,
)


def build_synthetic_dataset(seed: int = 12345):
    """
    Create synthetic 16-bit grayscale and 8-bit RGB brightfield datasets with known ground truth:
      - H x W = 384 x 384
      - Multi-scale absorbing structures (transmittance = 0.40, i.e., 60% absorption):
        * Small disc: radius = 8 px (< sigma)
        * Medium disc: radius = 22 px (~ 0.5 * sigma)
        * Large disc: radius = 38 px (~ 0.85 * sigma)
        * Elongated bar/stripe: 16 x 90 px
      - Illumination fields:
        1. Non-Gaussian cos^4 optical vignetting: L(r) = 1 / (1 + 0.38 * r^2)^2
        2. Tilted planar illumination gradient: L(x, y) = 0.62 + 0.24*(x/W) + 0.18*(y/H)
      - Dark-field offset D = 500 ADU (for 16-bit)
      - Poisson shot noise
    """
    rng = np.random.default_rng(seed)
    h, w = 384, 384
    yy, xx = np.mgrid[0:h, 0:w]
    yn = (yy - h / 2.0) / (h / 2.0)
    xn = (xx - w / 2.0) / (w / 2.0)
    r2 = xn ** 2 + yn ** 2

    # Multi-scale foreground masks
    small_mask = ((yy - 90) ** 2 + (xx - 95) ** 2) <= (8 ** 2)
    med_mask = ((yy - 105) ** 2 + (xx - 280) ** 2) <= (22 ** 2)
    large_mask = ((yy - 245) ** 2 + (xx - 192) ** 2) <= (38 ** 2)
    stripe_mask = (yy >= 250) & (yy <= 266) & (xx >= 60) & (xx <= 130)
    fg_mask = small_mask | med_mask | large_mask | stripe_mask
    bg_mask = ~fg_mask

    # Halo ring right around the large disc (39 < dist <= 56 px)
    dist_large = np.hypot(yy - 245, xx - 192)
    halo_ring_mask = (dist_large > 40.0) & (dist_large <= 58.0) & bg_mask

    transmittance = np.ones((h, w), dtype=np.float64)
    transmittance[fg_mask] = 0.40  # True Weber contrast = 0.60

    illum_cos4 = 1.0 / ((1.0 + 0.38 * r2) ** 2)
    illum_tilt = 0.62 + 0.24 * (xx / float(w)) + 0.18 * (yy / float(h))

    return {
        "rng": rng,
        "shape": (h, w),
        "small_mask": small_mask,
        "med_mask": med_mask,
        "large_mask": large_mask,
        "stripe_mask": stripe_mask,
        "fg_mask": fg_mask,
        "bg_mask": bg_mask,
        "halo_ring_mask": halo_ring_mask,
        "transmittance": transmittance,
        "illum_cos4": illum_cos4,
        "illum_tilt": illum_tilt,
    }


def evaluate_prediction(
    corr_arr: np.ndarray,
    transmittance: np.ndarray,
    bg_mask: np.ndarray,
    fg_mask: np.ndarray,
    halo_ring_mask: np.ndarray,
) -> dict:
    """Compute quantitative accuracy metrics against ground-truth transmittance."""
    c_f = corr_arr.mean(axis=2).astype(np.float64) if corr_arr.ndim == 3 else corr_arr.astype(np.float64)
    bg_scale = float(np.median(c_f[bg_mask]))
    ideal_signal = transmittance * bg_scale

    bg_rel_err = np.abs(c_f[bg_mask] - ideal_signal[bg_mask]) / np.maximum(ideal_signal[bg_mask], 1e-6)
    bg_med_rel_err_pct = float(np.median(bg_rel_err) * 100.0)
    bg_p95_rel_err_pct = float(np.percentile(bg_rel_err, 95.0) * 100.0)

    fg_mean_meas = float(c_f[fg_mask].mean())
    fg_mean_ideal = float(ideal_signal[fg_mask].mean())
    fg_mean_rel_err_pct = abs(fg_mean_meas - fg_mean_ideal) / max(fg_mean_ideal, 1e-6) * 100.0

    true_contrast = 1.0 - 0.40  # 0.60
    bg_mean_meas = float(c_f[bg_mask].mean())
    meas_contrast = (bg_mean_meas - fg_mean_meas) / max(bg_mean_meas, 1e-6)
    contrast_retention_pct = (meas_contrast / true_contrast) * 100.0

    halo_mean = float(c_f[halo_ring_mask].mean())
    halo_overshoot_pct = ((halo_mean - bg_scale) / max(bg_scale, 1e-6)) * 100.0

    return {
        "output_dtype": str(corr_arr.dtype),
        "bg_median_level": round(bg_scale, 2),
        "bg_median_rel_error_pct": round(bg_med_rel_err_pct, 3),
        "bg_p95_rel_error_pct": round(bg_p95_rel_err_pct, 3),
        "target_mean_rel_error_pct": round(fg_mean_rel_err_pct, 3),
        "true_weber_contrast": round(true_contrast, 4),
        "measured_weber_contrast": round(meas_contrast, 4),
        "contrast_retention_pct": round(contrast_retention_pct, 2),
        "halo_overshoot_pct": round(halo_overshoot_pct, 3),
    }


def run_all_benchmarks() -> dict:
    data = build_synthetic_dataset(seed=12345)
    rng = data["rng"]
    h, w = data["shape"]
    trans = data["transmittance"]
    bg_mask = data["bg_mask"]
    fg_mask = data["fg_mask"]
    halo_mask = data["halo_ring_mask"]

    benchmark_results = {"seed": 12345, "image_shape": [h, w], "scenarios": {}}

    for field_name, illum in [("cos4_vignetting", data["illum_cos4"]), ("tilted_plane", data["illum_tilt"])]:
        # 16-bit physical image: flatfield peak = 48000, darkfield = 500
        dark_true = np.full((h, w), 500.0, dtype=np.float64)
        flat_clean = dark_true + 48000.0 * illum
        sample_clean = dark_true + (48000.0 * illum * trans)

        # Add Poisson shot noise
        flat_u16 = np.clip(rng.poisson(flat_clean), 0, 65535).astype(np.uint16)
        dark_u16 = np.clip(rng.poisson(dark_true), 0, 65535).astype(np.uint16)
        sample_u16 = np.clip(rng.poisson(sample_clean), 0, 65535).astype(np.uint16)

        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, f"{field_name}_16bit.tif")
            tifffile.imwrite(tif_path, sample_u16)
            # 1. Legacy algorithm (reads via cv2.IMREAD_COLOR -> 8-bit, sigma=48, target_val=220)
            leg_loaded = legacy_imread_unicode(tif_path)
            leg_out = legacy_pseudo_flatfield_correction(leg_loaded, sigma=48.0, target_val=220.0)

        # 2. New Mode B — Naive unmasked Gaussian (`--bg-method gaussian`) on dark-subtracted or raw 16-bit
        # Note: for fair 16-bit comparison without dark offset, evaluate on net photon image (sample_u16 - 500)
        sample_net_u16 = np.clip(sample_u16.astype(np.int32) - 500, 0, 65535).astype(np.uint16)
        res_b_naive = correct_image_pipeline(
            sample_net_u16,
            mode="pseudo-flatfield",
            sigma_px=48.0,
            bg_method="gaussian",
            ground_truth_bg_mask=bg_mask,
        )

        # 3. New Mode B — Default Foreground-Masked Gaussian (`--bg-method masked_gaussian`)
        res_b_masked = correct_image_pipeline(
            sample_net_u16,
            mode="pseudo-flatfield",
            sigma_px=48.0,
            bg_method="masked_gaussian",
            ground_truth_bg_mask=bg_mask,
        )

        # 4. New Mode B — Negative Control: Too-small sigma (`sigma_px = 10.0`)
        res_b_small_sigma = correct_image_pipeline(
            sample_net_u16,
            mode="pseudo-flatfield",
            sigma_px=10.0,
            bg_method="masked_gaussian",
            ground_truth_bg_mask=bg_mask,
        )

        # 5. New Mode A — True Flat-Field Calibration (`flatfield_u16`, `darkfield_u16`)
        res_a_true = correct_image_pipeline(
            sample_u16,
            mode="flatfield",
            flatfield=flat_u16,
            darkfield=dark_u16,
            ground_truth_bg_mask=bg_mask,
        )

        benchmark_results["scenarios"][field_name] = {
            "legacy_8bit_per_channel_gaussian": evaluate_prediction(leg_out, trans, bg_mask, fg_mask, halo_mask),
            "new_mode_b_naive_gaussian_16bit": evaluate_prediction(res_b_naive["output_array"], trans, bg_mask, fg_mask, halo_mask),
            "new_mode_b_masked_gaussian_16bit": evaluate_prediction(res_b_masked["output_array"], trans, bg_mask, fg_mask, halo_mask),
            "negative_control_small_sigma_10px": evaluate_prediction(res_b_small_sigma["output_array"], trans, bg_mask, fg_mask, halo_mask),
            "new_mode_a_true_flatfield_16bit": evaluate_prediction(res_a_true["output_array"], trans, bg_mask, fg_mask, halo_mask),
        }

    # RGB Color Ratio Benchmark (Stained droplets under cos^4 vignetting)
    illum = data["illum_cos4"]
    rgb_clean = np.zeros((h, w, 3), dtype=np.float64)
    # Warm brightfield background: R=225, G=205, B=180
    rgb_clean[:, :, 0] = 225.0
    rgb_clean[:, :, 1] = 205.0
    rgb_clean[:, :, 2] = 180.0
    # Red stained droplets (R=195, G=75, B=85)
    rgb_clean[fg_mask, 0] = 195.0
    rgb_clean[fg_mask, 1] = 75.0
    rgb_clean[fg_mask, 2] = 85.0
    rgb_vignetted = np.clip(np.rint(rgb_clean * illum[:, :, None]), 0, 255).astype(np.uint8)

    res_rgb_shared = correct_image_pipeline(
        rgb_vignetted, mode="pseudo-flatfield", sigma_px=48.0, per_channel=False, ground_truth_bg_mask=bg_mask
    )
    res_rgb_per_ch = correct_image_pipeline(
        rgb_vignetted, mode="pseudo-flatfield", sigma_px=48.0, per_channel=True, ground_truth_bg_mask=bg_mask
    )
    # Legacy expects BGR
    leg_rgb_bgr = legacy_pseudo_flatfield_correction(rgb_vignetted[:, :, ::-1], sigma=48.0, target_val=220.0)
    leg_rgb_out = leg_rgb_bgr[:, :, ::-1]

    def _rgb_shift(out_rgb):
        out_f = out_rgb.astype(np.float64)
        true_bg_rg = 225.0 / 205.0
        true_bg_rb = 225.0 / 180.0
        true_fg_rg = 195.0 / 75.0
        meas_bg = out_f[bg_mask].mean(axis=0)
        meas_fg = out_f[fg_mask].mean(axis=0)
        bg_rg_err = abs((meas_bg[0] / meas_bg[1]) - true_bg_rg) / true_bg_rg * 100.0
        bg_rb_err = abs((meas_bg[0] / meas_bg[2]) - true_bg_rb) / true_bg_rb * 100.0
        fg_rg_err = abs((meas_fg[0] / meas_fg[1]) - true_fg_rg) / true_fg_rg * 100.0
        return {
            "bg_R_over_G_error_pct": round(float(bg_rg_err), 3),
            "bg_R_over_B_error_pct": round(float(bg_rb_err), 3),
            "fg_droplet_R_over_G_error_pct": round(float(fg_rg_err), 3),
        }

    benchmark_results["rgb_color_ratio_benchmark"] = {
        "legacy_per_channel_target220": _rgb_shift(leg_rgb_out),
        "new_mode_b_per_channel_flag": _rgb_shift(res_rgb_per_ch["output_array"]),
        "new_mode_b_shared_luminance_default": _rgb_shift(res_rgb_shared["output_array"]),
    }

    return benchmark_results


def write_benchmark_reports():
    res = run_all_benchmarks()
    out_dir = os.path.join(REPO_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "synthetic_benchmark_old_vs_new.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Synthetic Benchmark Comparison: Legacy vs. Refactored Mode B & Mode A",
        "",
        "Generated by `scripts/run_synthetic_benchmark.py` (fixed seed `12345`, `384x384` 16-bit synthetic images with Poisson noise, multi-scale absorbing discs/stripes of true transmittance `0.40` / Weber contrast `0.6000`).",
        "",
        "## 1. 16-Bit Intensity & Contrast Recovery Under Non-Gaussian Illumination",
        "",
    ]

    for field_name, title in [
        ("cos4_vignetting", "Scenario A: Non-Gaussian $\\cos^4(\\theta)$ Optical Vignetting + Poisson Noise"),
        ("tilted_plane", "Scenario B: Tilted Planar Illumination Gradient + Poisson Noise"),
    ]:
        sc = res["scenarios"][field_name]
        md_lines.extend([
            f"### {title}",
            "",
            "| Method | Output Dtype | BG Median Rel. Error (%) | BG p95 Rel. Error (%) | Target Mean Rel. Error (%) | Contrast Retention (%) | Halo Overshoot (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        labels = [
            ("legacy_8bit_per_channel_gaussian", "Legacy (`uint8` downcast + unmasked Gaussian)"),
            ("new_mode_b_naive_gaussian_16bit", "New Mode B (`--bg-method gaussian`, unmasked)"),
            ("new_mode_b_masked_gaussian_16bit", "**New Mode B (`masked_gaussian`, default)**"),
            ("negative_control_small_sigma_10px", "Negative Control (`sigma_px=10` too small)"),
            ("new_mode_a_true_flatfield_16bit", "**New Mode A (`flatfield` with F & D)**"),
        ]
        for key, label in labels:
            m = sc[key]
            md_lines.append(
                f"| {label} | `{m['output_dtype']}` | `{m['bg_median_rel_error_pct']:.3f}%` | "
                f"`{m['bg_p95_rel_error_pct']:.3f}%` | `{m['target_mean_rel_error_pct']:.3f}%` | "
                f"`{m['contrast_retention_pct']:.2f}%` | `{m['halo_overshoot_pct']:+.3f}%` |"
            )
        md_lines.append("")

    rgb_b = res["rgb_color_ratio_benchmark"]
    md_lines.extend([
        "## 2. RGB Chromatic Ratio Preservation (Stained Lipid Droplets on Warm Lamp Background)",
        "",
        "| Method | BG `R/G` Error (%) | BG `R/B` Error (%) | Foreground Droplet `R/G` Error (%) |",
        "| :--- | :---: | :---: | :---: |",
        f"| Legacy (`per-channel`, `target=220`) | `{rgb_b['legacy_per_channel_target220']['bg_R_over_G_error_pct']:.3f}%` | `{rgb_b['legacy_per_channel_target220']['bg_R_over_B_error_pct']:.3f}%` | `{rgb_b['legacy_per_channel_target220']['fg_droplet_R_over_G_error_pct']:.3f}%` |",
        f"| New Mode B (`--per-channel` flag) | `{rgb_b['new_mode_b_per_channel_flag']['bg_R_over_G_error_pct']:.3f}%` | `{rgb_b['new_mode_b_per_channel_flag']['bg_R_over_B_error_pct']:.3f}%` | `{rgb_b['new_mode_b_per_channel_flag']['fg_droplet_R_over_G_error_pct']:.3f}%` |",
        f"| **New Mode B (Shared Luminance Gain, default)** | **`{rgb_b['new_mode_b_shared_luminance_default']['bg_R_over_G_error_pct']:.3f}%`** | **`{rgb_b['new_mode_b_shared_luminance_default']['bg_R_over_B_error_pct']:.3f}%`** | **`{rgb_b['new_mode_b_shared_luminance_default']['fg_droplet_R_over_G_error_pct']:.3f}%`** |",
        "",
    ])

    md_path = os.path.join(out_dir, "SYNTHETIC_BENCHMARK_OLD_VS_NEW.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return res


if __name__ == "__main__":
    results = write_benchmark_reports()
    print(json.dumps(results, indent=2))
