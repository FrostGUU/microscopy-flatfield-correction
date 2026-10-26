"""
Stage 0 Baseline Characterization Tests
---------------------------------------
Demonstrates and quantifies the exact behaviors and limitations of the legacy implementation
(`scripts/correct_illumination.py` at baseline commit db715dd) on synthetic 16-bit grayscale
and RGB images:
  1. Question 1: 16-bit TIFF silent downconversion to 8-bit BGR (`cv2.IMREAD_COLOR`).
  2. Question 2: Intensity information loss via `[0, 255]` clipping, floor truncation (`astype(np.uint8)`
     instead of rounding), and dynamic range collapse.
  3. Question 3: Halo artifact and foreground intensity distortion caused by unmasked Gaussian blur.
  4. Question 4: Per-channel independent normalization altering RGB channel ratios and chromaticity.
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2
import tifffile
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import correct_illumination as ci


def _get_legacy_imread():
    return getattr(ci, "legacy_imread_unicode", ci.imread_unicode)


def _get_legacy_pseudo_ff():
    return getattr(ci, "legacy_pseudo_flatfield_correction", ci.pseudo_flatfield_correction)


def _get_legacy_imwrite():
    return getattr(ci, "legacy_imwrite_unicode", ci.imwrite_unicode)


def run_stage0_characterization():
    """Run quantitative characterization of legacy behavior and return metrics dict."""
    imread_fn = _get_legacy_imread()
    pseudo_ff_fn = _get_legacy_pseudo_ff()
    imwrite_fn = _get_legacy_imwrite()

    rng = np.random.default_rng(42)
    results = {}

    # -------------------------------------------------------------------------
    # Q1 & Q2: 16-bit Grayscale TIFF bit-depth loss, dynamic range collapse & clipping
    # -------------------------------------------------------------------------
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    # Smooth gradient from 4000 to 52000 in 16-bit space + fine 16-bit steps (0..15 LSB)
    fine_steps = ((xx + yy) % 16).astype(np.uint16)
    synth_u16 = np.clip(
        4000 + ((xx / (w - 1.0)) * 48000.0) + fine_steps, 0, 65535
    ).astype(np.uint16)
    # Add a bright highlight region (62000) above local background (~30000)
    synth_u16[100:140, 100:140] = 62000

    with tempfile.TemporaryDirectory() as tmpdir:
        tif_in = os.path.join(tmpdir, "synth_16bit_测试.tif")
        tif_out = os.path.join(tmpdir, "synth_16bit_out_测试.tif")
        tifffile.imwrite(tif_in, synth_u16)

        loaded = imread_fn(tif_in)
        corr = pseudo_ff_fn(loaded, sigma=50, target_val=220.0)
        imwrite_fn(tif_out, corr)
        saved_back = tifffile.imread(tif_out)

        unique_in = int(len(np.unique(synth_u16)))
        unique_loaded = int(len(np.unique(loaded)))
        unique_out = int(len(np.unique(corr)))

        clipped_255_fraction = float(np.mean(corr == 255))
        highlight_clipped_fraction = float(np.mean(corr[100:140, 100:140] == 255))

        # Truncation bias check: uniform image with intensity=10, target_val=220.0
        # 10.0 / (10.0 + 1e-5) * 220.0 = 219.99978 -> astype(np.uint8) floors to 219 instead of rounding to 220
        uniform_u8 = np.full((64, 64, 3), 10, dtype=np.uint8)
        corr_uniform = pseudo_ff_fn(uniform_u8, sigma=50, target_val=220.0)
        uniform_out_val = int(corr_uniform[32, 32, 0])

        results["q1_16bit_to_8bit"] = {
            "input_dtype": str(synth_u16.dtype),
            "input_shape": list(synth_u16.shape),
            "input_min_max": [int(synth_u16.min()), int(synth_u16.max())],
            "loaded_dtype": str(loaded.dtype),
            "loaded_shape": list(loaded.shape),
            "loaded_min_max": [int(loaded.min()), int(loaded.max())],
            "saved_output_dtype": str(saved_back.dtype),
            "unique_levels_input_16bit": unique_in,
            "unique_levels_after_imread": unique_loaded,
            "unique_levels_after_correction": unique_out,
            "information_levels_lost_pct": round((1.0 - unique_loaded / unique_in) * 100.0, 2),
        }
        results["q2_clipping_and_quantization"] = {
            "total_pixels_clipped_at_255_pct": round(clipped_255_fraction * 100.0, 3),
            "highlight_patch_clipped_at_255_pct": round(highlight_clipped_fraction * 100.0, 2),
            "uniform_input_50_target_220_output": uniform_out_val,
            "truncation_bias_lsb": 220 - uniform_out_val,
        }

    # -------------------------------------------------------------------------
    # Q3: Halo artifact & foreground attenuation around dense dark structure
    # -------------------------------------------------------------------------
    bg_true = 200.0
    img_halo = np.full((256, 256, 3), int(bg_true), dtype=np.uint8)
    cy, cx = 128, 128
    dist = np.hypot(yy - cy, xx - cx)
    fg_mask = dist <= 45.0
    halo_ring_mask = (dist > 48.0) & (dist <= 68.0)
    far_bg_mask = dist > 110.0
    img_halo[fg_mask] = 60  # True Weber contrast: (200 - 60) / 200 = 0.70; ideal scaled fg at target=220 is 66.0

    corr_halo = pseudo_ff_fn(img_halo, sigma=50, target_val=220.0)
    ideal_fg_after = 60.0 * (220.0 / 200.0)
    fg_mean_after = float(corr_halo[:, :, 0][fg_mask].mean())
    fg_rel_error_pct = abs(fg_mean_after - ideal_fg_after) / ideal_fg_after * 100.0
    halo_ring_max = int(corr_halo[:, :, 0][halo_ring_mask].max())
    halo_ring_mean = float(corr_halo[:, :, 0][halo_ring_mask].mean())
    far_bg_mean = float(corr_halo[:, :, 0][far_bg_mask].mean())
    true_contrast = (200.0 - 60.0) / 200.0
    measured_contrast_after = (far_bg_mean - fg_mean_after) / far_bg_mean

    results["q3_halo_and_contrast_loss"] = {
        "true_bg": 200.0,
        "true_fg": 60.0,
        "ideal_scaled_fg": round(ideal_fg_after, 2),
        "corrected_fg_mean": round(fg_mean_after, 2),
        "fg_intensity_distortion_pct": round(fg_rel_error_pct, 2),
        "true_weber_contrast": round(true_contrast, 4),
        "corrected_far_bg_mean": round(far_bg_mean, 2),
        "corrected_weber_contrast": round(measured_contrast_after, 4),
        "contrast_retention_pct": round((measured_contrast_after / true_contrast) * 100.0, 2),
        "halo_ring_mean_intensity": round(halo_ring_mean, 2),
        "halo_ring_max_intensity": halo_ring_max,
        "halo_overshoot_above_target_pct": round(((halo_ring_mean - 220.0) / 220.0) * 100.0, 2),
    }

    # -------------------------------------------------------------------------
    # Q4: Per-channel independent normalization altering RGB ratios & stain color
    # -------------------------------------------------------------------------
    img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    # Warm background: B=176, G=198, R=220 (true B:G:R = 0.80 : 0.90 : 1.00)
    img_rgb[:, :, 0] = 176
    img_rgb[:, :, 1] = 198
    img_rgb[:, :, 2] = 220

    # Place red-stained lipid droplets across a 120x120 region (B=80, G=70, R=196)
    droplet_mask = np.zeros((256, 256), dtype=bool)
    droplet_mask[68:188, 68:188] = rng.random((120, 120)) < 0.45
    img_rgb[droplet_mask, 0] = 80
    img_rgb[droplet_mask, 1] = 70
    img_rgb[droplet_mask, 2] = 196

    bg_mask = ~droplet_mask
    raw_bg_bgr = img_rgb[bg_mask].mean(axis=0)
    raw_fg_bgr = img_rgb[droplet_mask].mean(axis=0)
    raw_bg_r_over_b = float(raw_bg_bgr[2] / raw_bg_bgr[0])
    raw_fg_r_over_g = float(raw_fg_bgr[2] / raw_fg_bgr[1])
    raw_fg_r_minus_g = float(raw_fg_bgr[2] - raw_fg_bgr[1])

    corr_rgb = pseudo_ff_fn(img_rgb, sigma=45, target_val=220.0)
    corr_bg_bgr = corr_rgb[bg_mask].mean(axis=0)
    corr_fg_bgr = corr_rgb[droplet_mask].mean(axis=0)
    corr_bg_r_over_b = float(corr_bg_bgr[2] / corr_bg_bgr[0])
    corr_fg_r_over_g = float(corr_fg_bgr[2] / corr_fg_bgr[1])
    corr_fg_r_minus_g = float(corr_fg_bgr[2] - corr_fg_bgr[1])

    results["q4_per_channel_color_shift"] = {
        "raw_bg_mean_BGR": [round(float(x), 2) for x in raw_bg_bgr],
        "corr_bg_mean_BGR": [round(float(x), 2) for x in corr_bg_bgr],
        "raw_bg_R_over_B_ratio": round(raw_bg_r_over_b, 4),
        "corr_bg_R_over_B_ratio": round(corr_bg_r_over_b, 4),
        "bg_R_over_B_shift_pct": round(abs(corr_bg_r_over_b - raw_bg_r_over_b) / raw_bg_r_over_b * 100.0, 2),
        "raw_droplet_mean_BGR": [round(float(x), 2) for x in raw_fg_bgr],
        "corr_droplet_mean_BGR": [round(float(x), 2) for x in corr_fg_bgr],
        "raw_droplet_R_over_G_ratio": round(raw_fg_r_over_g, 4),
        "corr_droplet_R_over_G_ratio": round(corr_fg_r_over_g, 4),
        "droplet_R_over_G_shift_pct": round(abs(corr_fg_r_over_g - raw_fg_r_over_g) / raw_fg_r_over_g * 100.0, 2),
        "raw_droplet_R_minus_G": round(raw_fg_r_minus_g, 2),
        "corr_droplet_R_minus_G": round(corr_fg_r_minus_g, 2),
    }

    return results


def test_q1_legacy_converts_16bit_tiff_to_8bit():
    """Characterization test for Question 1: legacy imread_unicode silently converts 16-bit grayscale TIFF to 8-bit 3-channel BGR."""
    res = run_stage0_characterization()["q1_16bit_to_8bit"]
    assert res["input_dtype"] == "uint16"
    assert res["loaded_dtype"] == "uint8"
    assert res["loaded_shape"] == [256, 256, 3]
    assert res["saved_output_dtype"] == "uint8"
    assert res["information_levels_lost_pct"] > 90.0


def test_q2_legacy_clips_and_truncates_intensity():
    """Characterization test for Question 2: legacy pseudo_flatfield_correction clips highlights to 255 and truncates float via astype(uint8)."""
    res = run_stage0_characterization()["q2_clipping_and_quantization"]
    assert res["highlight_patch_clipped_at_255_pct"] == 100.0
    assert res["truncation_bias_lsb"] == 1  # 50 -> 219 instead of 220 due to +1e-5 denominator and astype(uint8) floor truncation


def test_q3_legacy_gaussian_halo_and_contrast_loss():
    """Characterization test for Question 3: unmasked Gaussian blur inflates dark foreground intensity by >20%, attenuates contrast, and creates a bright halo ring clipped at 255."""
    res = run_stage0_characterization()["q3_halo_and_contrast_loss"]
    assert res["fg_intensity_distortion_pct"] > 20.0
    assert res["contrast_retention_pct"] < 92.0
    assert res["halo_ring_max_intensity"] == 255
    assert res["halo_overshoot_above_target_pct"] > 12.0


def test_q4_legacy_per_channel_shifts_color_ratios():
    """Characterization test for Question 4: per-channel division shifts both background R/B ratio and stained droplet R/G ratio by >20%."""
    res = run_stage0_characterization()["q4_per_channel_color_shift"]
    assert res["bg_R_over_B_shift_pct"] > 20.0
    assert res["droplet_R_over_G_shift_pct"] > 20.0


if __name__ == "__main__":
    metrics = run_stage0_characterization()
    out_dir = os.path.join(REPO_ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage0_characterization_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
