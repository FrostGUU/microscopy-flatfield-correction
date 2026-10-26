"""
Stage 4: Independent Quality Control (QC) Metrics & Machine-Readable JSON Report
--------------------------------------------------------------------------------
Computes independent background uniformity (coarse block medians on background pixels,
never reusing the correction algorithm's own background estimate), saturation & clipping
statistics, gain distribution percentiles, RGB foreground/background channel ratio shifts,
and structured warning codes.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from io_utils import LoadedImage, QuantizationResult, SOFTWARE_VERSION, make_warning

STAIN_CAUTION_TEXT = (
    "CAUTION (Oil Red O / DAB-IHC / H&E Stained Images): Illumination correction alters pixel "
    "intensities and can shift color segmentation thresholds, positive stained area percentages, "
    "and integrated optical density / staining intensity. Mode B (pseudo-flat-field) outputs are "
    "approximations for visual uniformity and should NOT be used for quantitative intensity claims "
    "unless independently validated against physical flat-field calibration standards."
)


def compute_independent_bg_fg_masks(
    raw_arr: np.ndarray,
    ground_truth_bg_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify background vs. foreground pixels independently of the flat-field estimator.
    If `ground_truth_bg_mask` is provided (e.g. in synthetic benchmarks), uses it directly.
    Otherwise, applies an independent Otsu + local percentile separation on image luminance.
    """
    lum = raw_arr.mean(axis=2).astype(np.float32) if raw_arr.ndim == 3 else raw_arr.astype(np.float32)
    h, w = lum.shape

    if ground_truth_bg_mask is not None:
        bg_mask = ground_truth_bg_mask.astype(bool)
        fg_mask = ~bg_mask
        return bg_mask, fg_mask

    # Normalize luminance to 8-bit [0, 255] strictly for independent Otsu threshold estimation
    l_min, l_max = float(lum.min()), float(lum.max())
    if l_max - l_min < 1e-3:
        return np.ones((h, w), dtype=bool), np.zeros((h, w), dtype=bool)

    lum_u8 = np.clip((lum - l_min) / (l_max - l_min) * 255.0, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(lum_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_raw_thresh = l_min + (float(otsu_val) / 255.0) * (l_max - l_min)

    # Also compute coarse 8x8 block 50th percentiles so vignette corners aren't misclassified as foreground by global Otsu
    bh = max(1, h // 8)
    bw = max(1, w // 8)
    bg_mask = np.zeros((h, w), dtype=bool)
    for gy in range(8):
        y0 = gy * bh
        y1 = h if gy == 7 else (gy + 1) * bh
        for gx in range(8):
            x0 = gx * bw
            x1 = w if gx == 7 else (gx + 1) * bw
            patch = lum[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            p45 = float(np.percentile(patch, 45.0))
            p85 = float(np.percentile(patch, 85.0))
            # If block has high contrast (contains both dark stain and bright background), split mid-way;
            # otherwise almost the whole block is background
            if (p85 - p45) > 0.08 * max(p85, 1.0):
                local_thresh = 0.5 * (p45 + p85)
            else:
                local_thresh = p45 - 0.05 * max(p85, 1.0)
            bg_mask[y0:y1, x0:x1] = patch >= min(local_thresh, otsu_raw_thresh * 1.05)

    if np.count_nonzero(bg_mask) == 0:
        bg_mask = lum >= float(np.median(lum))

    fg_mask = ~bg_mask
    return bg_mask, fg_mask


def compute_independent_grid_uniformity(
    arr: np.ndarray,
    bg_mask: np.ndarray,
    grid_size: int = 8,
) -> Dict[str, Any]:
    """
    Measure background spatial uniformity on an independent coarse grid (`grid_size x grid_size`)
    by taking the median of background pixels (`bg_mask == True`) inside each grid block.
    Never uses the correction algorithm's own Gaussian/polynomial background surface.
    """
    lum = arr.mean(axis=2).astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
    h, w = lum.shape
    gh = min(grid_size, max(2, h // 4))
    gw = min(grid_size, max(2, w // 4))
    bh = h // gh
    bw = w // gw

    block_medians: List[float] = []
    grid_map = np.full((gh, gw), np.nan, dtype=np.float32)

    for iy in range(gh):
        y0 = iy * bh
        y1 = h if iy == gh - 1 else (iy + 1) * bh
        for ix in range(gw):
            x0 = ix * bw
            x1 = w if ix == gw - 1 else (ix + 1) * bw
            patch_lum = lum[y0:y1, x0:x1]
            patch_mask = bg_mask[y0:y1, x0:x1]
            bg_vals = patch_lum[patch_mask]
            if bg_vals.size >= 4:
                val = float(np.median(bg_vals))
            elif patch_lum.size > 0:
                val = float(np.percentile(patch_lum, 80.0))
            else:
                continue
            block_medians.append(val)
            grid_map[iy, ix] = val

    arr_meds = np.array(block_medians, dtype=np.float64)
    if arr_meds.size == 0 or float(arr_meds.mean()) <= 0.0:
        return {
            "grid_shape": [gh, gw],
            "num_valid_blocks": int(arr_meds.size),
            "block_median_min": 0.0,
            "block_median_max": 0.0,
            "block_median_mean": 0.0,
            "cv": 0.0,
            "max_min_ratio": 1.0,
            "center_block_median": 0.0,
            "corner_min_median": 0.0,
            "corner_mean_median": 0.0,
            "corner_max_drop_pct": 0.0,
        }

    mean_val = float(arr_meds.mean())
    std_val = float(arr_meds.std(ddof=0))
    min_val = float(arr_meds.min())
    max_val = float(arr_meds.max())
    cv_val = std_val / max(mean_val, 1e-6)
    max_min_ratio = max_val / max(min_val, 1e-6)

    # Center vs 4 corners on the grid
    cy0, cy1 = max(0, gh // 3), max(1, (2 * gh) // 3)
    cx0, cx1 = max(0, gw // 3), max(1, (2 * gw) // 3)
    center_blocks = grid_map[cy0:cy1, cx0:cx1]
    center_med = float(np.nanmedian(center_blocks)) if np.any(np.isfinite(center_blocks)) else mean_val

    corners = [
        float(grid_map[0, 0]),
        float(grid_map[0, gw - 1]),
        float(grid_map[gh - 1, 0]),
        float(grid_map[gh - 1, gw - 1]),
    ]
    valid_corners = [c for c in corners if np.isfinite(c)]
    corner_min = min(valid_corners) if valid_corners else min_val
    corner_mean = float(np.mean(valid_corners)) if valid_corners else mean_val
    corner_drop_pct = abs(center_med - corner_min) / max(center_med, 1e-6) * 100.0

    return {
        "grid_shape": [gh, gw],
        "num_valid_blocks": int(arr_meds.size),
        "block_median_min": round(min_val, 3),
        "block_median_max": round(max_val, 3),
        "block_median_mean": round(mean_val, 3),
        "cv": round(cv_val, 5),
        "max_min_ratio": round(max_min_ratio, 4),
        "center_block_median": round(center_med, 3),
        "corner_min_median": round(corner_min, 3),
        "corner_mean_median": round(corner_mean, 3),
        "corner_max_drop_pct": round(corner_drop_pct, 2),
    }


def compute_channel_ratio_metrics(
    raw_arr: np.ndarray,
    corr_arr: np.ndarray,
    bg_mask: np.ndarray,
    fg_mask: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    For 3-channel RGB images (in RGB order), compute mean channel ratios (R/G, B/G, R/B)
    in both background (`bg_mask`) and foreground (`fg_mask`) regions before vs. after correction.
    """
    if raw_arr.ndim != 3 or raw_arr.shape[2] < 3:
        return None

    raw_rgb = raw_arr[:, :, :3].astype(np.float64)
    corr_rgb = corr_arr[:, :, :3].astype(np.float64)

    def _region_ratios(img: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        if np.count_nonzero(mask) < 4:
            return None
        means = img[mask].mean(axis=0)  # [R, G, B]
        r_m, g_m, b_m = float(means[0]), float(means[1]), float(means[2])
        return {
            "mean_RGB": [round(r_m, 3), round(g_m, 3), round(b_m, 3)],
            "R_over_G": round(r_m / max(g_m, 1e-6), 5),
            "B_over_G": round(b_m / max(g_m, 1e-6), 5),
            "R_over_B": round(r_m / max(b_m, 1e-6), 5),
        }

    def _compare_region(mask: np.ndarray) -> Optional[Dict[str, Any]]:
        before = _region_ratios(raw_rgb, mask)
        after = _region_ratios(corr_rgb, mask)
        if before is None or after is None:
            return None
        rg_shift_pct = abs(after["R_over_G"] - before["R_over_G"]) / max(before["R_over_G"], 1e-6) * 100.0
        bg_shift_pct = abs(after["B_over_G"] - before["B_over_G"]) / max(before["B_over_G"], 1e-6) * 100.0
        rb_shift_pct = abs(after["R_over_B"] - before["R_over_B"]) / max(before["R_over_B"], 1e-6) * 100.0
        return {
            "pixel_count": int(np.count_nonzero(mask)),
            "before": before,
            "after": after,
            "R_over_G_rel_shift_pct": round(rg_shift_pct, 4),
            "B_over_G_rel_shift_pct": round(bg_shift_pct, 4),
            "R_over_B_rel_shift_pct": round(rb_shift_pct, 4),
            "max_ratio_rel_shift_pct": round(max(rg_shift_pct, bg_shift_pct, rb_shift_pct), 4),
        }

    return {
        "background_region": _compare_region(bg_mask),
        "foreground_region": _compare_region(fg_mask),
    }


def build_qc_report(
    loaded: LoadedImage,
    quant: QuantizationResult,
    gain_map: np.ndarray,
    mode_params: Dict[str, Any],
    mode_warnings: List[Dict[str, str]],
    protected_fraction: float = 0.0,
    capped_gain_fraction: float = 0.0,
    extra_diagnostics: Optional[Dict[str, Any]] = None,
    ground_truth_bg_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Assemble complete machine-readable QC report dictionary.
    """
    raw_core = loaded.data
    out_core = quant.core_array

    in_ceiling = 4095.0 if loaded.detected_12bit_container else (65535.0 if loaded.bit_depth == 16 else 255.0)
    out_ceiling = 65535.0 if loaded.bit_depth == 16 else 255.0

    in_sat_frac = float(np.count_nonzero(raw_core >= in_ceiling) / float(raw_core.size))
    out_sat_frac = float(np.count_nonzero(out_core >= out_ceiling) / float(out_core.size))

    bg_mask, fg_mask = compute_independent_bg_fg_masks(raw_core, ground_truth_bg_mask=ground_truth_bg_mask)
    unif_before = compute_independent_grid_uniformity(raw_core, bg_mask=bg_mask)
    unif_after = compute_independent_grid_uniformity(out_core, bg_mask=bg_mask)

    g_flat = gain_map.astype(np.float64).ravel()
    gain_stats = {
        "min": round(float(g_flat.min()), 5),
        "p1": round(float(np.percentile(g_flat, 1.0)), 5),
        "median": round(float(np.median(g_flat)), 5),
        "p99": round(float(np.percentile(g_flat, 99.0)), 5),
        "max": round(float(g_flat.max()), 5),
        "exceeded_max_gain_fraction": round(float(capped_gain_fraction), 6),
        "protected_low_bg_or_invalid_denom_fraction": round(float(protected_fraction), 6),
    }

    channel_ratios = compute_channel_ratio_metrics(raw_core, out_core, bg_mask=bg_mask, fg_mask=fg_mask)

    all_warnings: List[Dict[str, str]] = []
    seen_codes = set()
    for w in (loaded.io_warnings + mode_warnings + quant.warnings):
        key = (w["code"], w["message"])
        if key not in seen_codes:
            seen_codes.add(key)
            all_warnings.append(w)

    # Always include the fixed caution for brightfield stained images
    all_warnings.append(make_warning("STAIN_QUANTIFICATION_CAUTION", STAIN_CAUTION_TEXT))

    report: Dict[str, Any] = {
        "software_version": SOFTWARE_VERSION,
        "source_path": loaded.source_path,
        "mode": mode_params.get("mode", "unknown"),
        "parameters": mode_params,
        "input_summary": {
            "dtype": str(loaded.orig_dtype),
            "bit_depth": loaded.bit_depth,
            "detected_12bit_in_16bit_container": loaded.detected_12bit_container,
            "shape": list(loaded.orig_shape),
            "channels": int(loaded.orig_shape[2]) if len(loaded.orig_shape) == 3 else 1,
            "has_alpha": loaded.has_alpha,
            "min": float(raw_core.min()),
            "max": float(raw_core.max()),
            "median": float(np.median(raw_core)),
            "saturated_pixel_fraction": round(in_sat_frac, 6),
        },
        "output_summary": {
            "dtype": str(quant.array.dtype),
            "shape": list(quant.array.shape),
            "channels": int(quant.array.shape[2]) if quant.array.ndim == 3 else 1,
            "min": float(out_core.min()),
            "max": float(out_core.max()),
            "median": float(np.median(out_core)),
            "saturated_pixel_fraction": round(out_sat_frac, 6),
            "clipped_low_count": quant.clipped_low_count,
            "clipped_high_count": quant.clipped_high_count,
            "clipped_low_fraction": round(quant.clipped_low_fraction, 6),
            "clipped_high_fraction": round(quant.clipped_high_fraction, 6),
            "clipped_total_fraction": round(quant.clipped_total_fraction, 6),
        },
        "background_uniformity_independent_grid": {
            "measurement_method": (
                "Coarse 8x8 block medians computed strictly on background pixels identified by an "
                "independent Otsu/percentile mask (or ground-truth mask), never using the correction background surface."
            ),
            "before": unif_before,
            "after": unif_after,
            "cv_reduction_pct": round(
                (1.0 - unif_after["cv"] / max(unif_before["cv"], 1e-8)) * 100.0
                if unif_before["cv"] > 1e-7
                else 0.0,
                2,
            ),
        },
        "gain_distribution": gain_stats,
        "channel_ratio_analysis": channel_ratios,
        "stain_quantification_caution": STAIN_CAUTION_TEXT,
        "warnings": all_warnings,
    }
    if extra_diagnostics:
        report["diagnostics"] = extra_diagnostics

    return report
