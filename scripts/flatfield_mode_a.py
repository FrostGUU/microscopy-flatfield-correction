"""
Stage 2: Mode A — True Instrument Flat-Field Correction
-------------------------------------------------------
Formula:
  I_corr = ((I_raw - D) / (F - D)) * norm(F - D) [+ median(D) if keep_offset]

Features:
  - Single-frame or multi-frame (pixel-wise median) flat-field F and dark-field D
  - Strict channel and spatial resolution matching (unless --allow-resize is explicitly set)
  - Flat-field reference quality diagnostics (saturation ratio, high-frequency dust/bubble artifacts, bit-depth match)
  - Denominator protection where (F - D) <= denom_eps_rel * max(F - D) with spatial summary reporting
  - Negative dark-subtracted pixel counting and clipping
"""

import os
import glob
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from io_utils import LoadedImage, read_image, inspect_loaded_array, make_warning


@dataclass
class ModeAResult:
    """Result container for Mode A true flat-field correction."""
    corrected_float: np.ndarray            # (H, W) or (H, W, 3) float32
    gain_map: np.ndarray                   # Effective multiplicative gain map (H, W) or (H, W, 3) float32
    invalid_mask: np.ndarray               # Boolean mask of protected denominator pixels
    invalid_fraction: float
    invalid_spatial_summary: Dict[str, Any]
    negative_after_dark_count: int
    negative_after_dark_fraction: float
    norm_factors: List[float]              # Per-channel normalization factor median(F-D) or mean(F-D)
    dark_offsets_added: List[float]        # Per-channel offset added back (0.0 unless keep_offset=True)
    ref_diagnostics: Dict[str, Any]
    warnings: List[Dict[str, str]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


def load_reference_frames(
    source: Union[str, Sequence[str], np.ndarray, LoadedImage],
    role: str = "flatfield",
) -> Tuple[LoadedImage, int]:
    """
    Load a single reference image or multiple frames (from a directory or list of file paths).
    When multiple frames are provided, computes the pixel-wise median in float32 while preserving
    the original container dtype metadata.
    Returns (LoadedImage, num_frames).
    """
    if isinstance(source, LoadedImage):
        return source, 1
    if isinstance(source, np.ndarray):
        return inspect_loaded_array(source, source_path=f"<{role}_array>"), 1

    paths: List[str] = []
    if isinstance(source, str):
        if os.path.isdir(source):
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                paths.extend(glob.glob(os.path.join(source, f"*{ext}")))
                paths.extend(glob.glob(os.path.join(source, f"*{ext.upper()}")))
            paths = sorted(list(set(paths)))
            if not paths:
                raise FileNotFoundError(f"No reference images found in {role} directory: {source}")
        else:
            paths = [source]
    elif isinstance(source, (list, tuple)):
        paths = list(source)
    else:
        raise ValueError(f"Unsupported {role} source type: {type(source).__name__}")

    if len(paths) == 1:
        return read_image(paths[0]), 1

    loaded_frames = [read_image(p) for p in paths]
    first = loaded_frames[0]
    for idx, lf in enumerate(loaded_frames[1:], start=2):
        if lf.data.shape != first.data.shape or lf.orig_dtype != first.orig_dtype:
            raise ValueError(
                f"Inconsistent shape/dtype across {role} frames: frame 1 ({paths[0]}) has "
                f"{first.data.shape}/{first.orig_dtype}, frame {idx} ({paths[idx-1]}) has "
                f"{lf.data.shape}/{lf.orig_dtype}."
            )

    stack = np.stack([lf.data.astype(np.float32) for lf in loaded_frames], axis=0)
    median_arr = np.rint(np.median(stack, axis=0)).astype(first.orig_dtype)
    combined = LoadedImage(
        data=median_arr,
        alpha=first.alpha,
        orig_dtype=first.orig_dtype,
        orig_shape=first.orig_shape,
        is_rgb=first.is_rgb,
        has_alpha=first.has_alpha,
        bit_depth=first.bit_depth,
        detected_12bit_container=first.detected_12bit_container,
        source_path=f"{paths[0]} (+{len(paths)-1} frames)",
        io_warnings=list(first.io_warnings),
    )
    return combined, len(paths)


def _summarize_invalid_mask(mask_2d: np.ndarray) -> Dict[str, Any]:
    """Summarize spatial distribution of invalid denominator pixels."""
    total_px = int(mask_2d.size)
    inv_count = int(np.count_nonzero(mask_2d))
    if inv_count == 0:
        return {
            "invalid_pixel_count": 0,
            "invalid_pixel_fraction": 0.0,
            "bounding_box_yx": None,
            "quadrant_counts": {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0},
        }
    ys, xs = np.where(mask_2d)
    h, w = mask_2d.shape
    mid_y, mid_x = h // 2, w // 2
    return {
        "invalid_pixel_count": inv_count,
        "invalid_pixel_fraction": float(inv_count / float(total_px)),
        "bounding_box_yx": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
        "quadrant_counts": {
            "top_left": int(np.count_nonzero(mask_2d[:mid_y, :mid_x])),
            "top_right": int(np.count_nonzero(mask_2d[:mid_y, mid_x:])),
            "bottom_left": int(np.count_nonzero(mask_2d[mid_y:, :mid_x])),
            "bottom_right": int(np.count_nonzero(mask_2d[mid_y:, mid_x:])),
        },
    }


def inspect_flatfield_reference(
    flat_img: LoadedImage,
    sample_img: LoadedImage,
    sat_warn_threshold: float = 0.001,
    hf_artifact_threshold: float = 0.05,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Audit flat-field reference image F for:
      1. Bit-depth consistency with sample image
      2. Sensor saturation fraction (pixels >= 99% of container ceiling)
      3. High-frequency local artifacts (dust, bubbles, debris) via band-pass residual
    """
    warnings_list: List[Dict[str, str]] = []
    f_arr = flat_img.data.astype(np.float32)

    if flat_img.orig_dtype != sample_img.orig_dtype:
        warnings_list.append(
            make_warning(
                "REFERENCE_BIT_DEPTH_MISMATCH",
                f"Flat-field reference dtype ({flat_img.orig_dtype}) differs from sample dtype ({sample_img.orig_dtype}). "
                "True flat-field calibration requires matching sensor bit-depth and gain settings.",
            )
        )

    ceiling = 4095.0 if flat_img.detected_12bit_container else (65535.0 if flat_img.bit_depth == 16 else 255.0)
    sat_mask = f_arr >= (0.99 * ceiling)
    sat_fraction = float(np.count_nonzero(sat_mask) / float(f_arr.size))
    if sat_fraction > sat_warn_threshold:
        warnings_list.append(
            make_warning(
                "REFERENCE_SATURATED",
                f"Flat-field reference has {sat_fraction * 100.0:.3f}% saturated/near-saturated pixels "
                f"(>= 99% of {int(ceiling)} ceiling). Nonlinear saturation in F distorts flat-field gain.",
            )
        )

    # Check high-frequency local anomalies (dust motes, air bubbles)
    # Compare small-scale smoothed F (sigma=1.5) against medium-scale smoothed F (sigma=18)
    f_lum = f_arr.mean(axis=2) if f_arr.ndim == 3 else f_arr
    h, w = f_lum.shape
    sigma_hi = min(18.0, max(4.0, min(h, w) / 10.0))
    sigma_lo = min(1.5, sigma_hi / 4.0)
    smooth_lo = gaussian_filter(f_lum, sigma=sigma_lo, mode="reflect")
    smooth_hi = gaussian_filter(f_lum, sigma=sigma_hi, mode="reflect")
    med_f = float(np.median(f_lum))
    rel_residual = np.abs(smooth_lo - smooth_hi) / max(med_f, 1.0)
    hf_p999 = float(np.percentile(rel_residual, 99.9))
    hf_max = float(rel_residual.max())

    if hf_p999 > hf_artifact_threshold:
        warnings_list.append(
            make_warning(
                "REFERENCE_HIGH_FREQUENCY_ARTIFACTS",
                f"Flat-field reference contains localized high-frequency anomalies (p99.9 relative residual = "
                f"{hf_p999 * 100.0:.2f}% > {hf_artifact_threshold * 100.0:.1f}% threshold, max = {hf_max * 100.0:.2f}%). "
                "Inspect reference for dust, bubbles, or residual cells that will imprint inverse ghosts onto corrected images.",
            )
        )

    diagnostics = {
        "reference_dtype": str(flat_img.orig_dtype),
        "reference_shape": list(flat_img.data.shape),
        "saturation_ceiling_used": ceiling,
        "saturated_pixel_fraction": sat_fraction,
        "high_freq_residual_p999": hf_p999,
        "high_freq_residual_max": hf_max,
    }
    return diagnostics, warnings_list


def flatfield_correct_mode_a(
    sample: Union[LoadedImage, np.ndarray],
    flatfield: Union[str, Sequence[str], np.ndarray, LoadedImage],
    darkfield: Optional[Union[str, Sequence[str], np.ndarray, LoadedImage]] = None,
    norm_stat: str = "median",
    keep_offset: bool = False,
    denom_eps_rel: float = 0.01,
    allow_resize: bool = False,
    clip_negative: bool = True,
    modality: str = "brightfield",
) -> ModeAResult:
    """
    Execute Mode A true instrument flat-field correction:
      I_corr = ((I_raw - D) / (F - D)) * norm(F - D) [+ median(D) if keep_offset]
    """
    sample_img = sample if isinstance(sample, LoadedImage) else inspect_loaded_array(sample, "<sample>")
    flat_img, num_flat_frames = load_reference_frames(flatfield, role="flatfield")
    if darkfield is not None:
        dark_img, num_dark_frames = load_reference_frames(darkfield, role="darkfield")
    else:
        dark_img, num_dark_frames = None, 0

    if norm_stat not in ("median", "mean"):
        raise ValueError(f"Unsupported norm_stat '{norm_stat}'. Choose 'median' or 'mean'.")
    if denom_eps_rel <= 0.0 or denom_eps_rel >= 1.0 or not np.isfinite(denom_eps_rel):
        raise ValueError(f"denom_eps_rel must be in (0, 1), got {denom_eps_rel}.")

    warnings_list: List[Dict[str, str]] = []

    # 1. Check channel compatibility
    if flat_img.is_rgb != sample_img.is_rgb:
        raise ValueError(
            f"Channel mismatch between sample ({'RGB 3-channel' if sample_img.is_rgb else 'Grayscale 1-channel'}) "
            f"and flat-field reference ({'RGB 3-channel' if flat_img.is_rgb else 'Grayscale 1-channel'}). "
            "Mode A requires per-channel paired flat-field references."
        )
    if dark_img is not None and dark_img.is_rgb != sample_img.is_rgb:
        raise ValueError(
            f"Channel mismatch between sample ({'RGB 3-channel' if sample_img.is_rgb else 'Grayscale 1-channel'}) "
            f"and dark-field reference ({'RGB 3-channel' if dark_img.is_rgb else 'Grayscale 1-channel'})."
        )

    # 2. Check spatial dimensions & handle --allow-resize
    sh, sw = sample_img.data.shape[:2]
    fh, fw = flat_img.data.shape[:2]
    f_arr = flat_img.data.astype(np.float32)

    if (fh, fw) != (sh, sw):
        if not allow_resize:
            raise ValueError(
                f"Spatial resolution mismatch: sample is ({sh}, {sw}) but flat-field reference is ({fh}, {fw}). "
                "Pass --allow-resize to explicitly allow bilinear resizing of the reference."
            )
        f_arr = cv2.resize(f_arr, (sw, sh), interpolation=cv2.INTER_LINEAR)
        warnings_list.append(
            make_warning(
                "RESIZED_REFERENCE",
                f"Flat-field reference was resized from ({fh}, {fw}) to match sample ({sh}, {sw}) via --allow-resize. "
                "If sample was cropped rather than binned/scaled, resizing the full-field reference is optically invalid.",
            )
        )

    if dark_img is not None:
        dh, dw = dark_img.data.shape[:2]
        d_arr = dark_img.data.astype(np.float32)
        if (dh, dw) != (sh, sw):
            if not allow_resize:
                raise ValueError(
                    f"Spatial resolution mismatch: sample is ({sh}, {sw}) but dark-field reference is ({dh}, {dw}). "
                    "Pass --allow-resize to explicitly allow resizing."
                )
            d_arr = cv2.resize(d_arr, (sw, sh), interpolation=cv2.INTER_LINEAR)
            warnings_list.append(
                make_warning(
                    "RESIZED_REFERENCE",
                    f"Dark-field reference was resized from ({dh}, {dw}) to match sample ({sh}, {sw}) via --allow-resize.",
                )
            )
    else:
        d_arr = np.zeros_like(f_arr, dtype=np.float32)

    # 3. Inspect flat-field reference quality
    ref_diag, ref_warnings = inspect_flatfield_reference(flat_img, sample_img)
    ref_diag["num_flatfield_frames"] = num_flat_frames
    ref_diag["num_darkfield_frames"] = num_dark_frames
    warnings_list.extend(ref_warnings)

    raw_f = sample_img.data.astype(np.float32)
    if not (np.all(np.isfinite(raw_f)) and np.all(np.isfinite(f_arr)) and np.all(np.isfinite(d_arr))):
        raise ValueError("Input sample, flat-field, or dark-field contains non-finite (NaN or Inf) values.")

    # 4. Dark-field subtraction & negative pixel handling
    net_sample = raw_f - d_arr
    neg_mask = net_sample < 0.0
    neg_count = int(np.count_nonzero(neg_mask))
    neg_frac = float(neg_count / float(net_sample.size))
    if neg_count > 0:
        if clip_negative:
            net_sample = np.maximum(net_sample, 0.0)
        warnings_list.append(
            make_warning(
                "NEGATIVE_AFTER_DARK_SUBTRACTION",
                f"{neg_count} values ({neg_frac * 100.0:.3f}%) were negative after subtracting dark-field D "
                f"and were {'clipped to 0' if clip_negative else 'left unclipped'}.",
            )
        )

    # 5. Per-channel denominator protection and normalization
    denom = f_arr - d_arr
    if raw_f.ndim == 2:
        channels_net = [net_sample]
        channels_denom = [denom]
        channels_d = [d_arr]
    else:
        channels_net = [net_sample[:, :, c] for c in range(3)]
        channels_denom = [denom[:, :, c] for c in range(3)]
        channels_d = [d_arr[:, :, c] for c in range(3)]

    corr_channels: List[np.ndarray] = []
    gain_channels: List[np.ndarray] = []
    invalid_masks: List[np.ndarray] = []
    norm_factors: List[float] = []
    dark_offsets: List[float] = []

    for c_idx, (c_net, c_den, c_d) in enumerate(zip(channels_net, channels_denom, channels_d)):
        max_den = float(c_den.max())
        if max_den <= 0.0:
            # Entire channel flat-field <= dark-field! Protect all pixels.
            c_inv = np.ones(c_den.shape, dtype=bool)
            norm_val = 0.0
            c_gain = np.ones_like(c_den, dtype=np.float32)
            c_corr = c_net.copy()
        else:
            thresh = denom_eps_rel * max_den
            c_inv = c_den <= thresh
            valid_den = c_den[~c_inv]
            if valid_den.size == 0:
                norm_val = 0.0
                c_gain = np.ones_like(c_den, dtype=np.float32)
                c_corr = c_net.copy()
            else:
                norm_val = float(np.median(valid_den)) if norm_stat == "median" else float(np.mean(valid_den))
                safe_den = np.where(c_inv, 1.0, c_den)
                c_gain = np.where(c_inv, 1.0, norm_val / safe_den).astype(np.float32)
                c_corr = c_net * c_gain

        offset_val = float(np.median(c_d)) if (keep_offset and dark_img is not None) else 0.0
        if offset_val != 0.0:
            c_corr = c_corr + offset_val

        corr_channels.append(c_corr.astype(np.float32))
        gain_channels.append(c_gain.astype(np.float32))
        invalid_masks.append(c_inv)
        norm_factors.append(norm_val)
        dark_offsets.append(offset_val)

    combined_invalid_mask = np.logical_or.reduce(invalid_masks)
    invalid_summary = _summarize_invalid_mask(combined_invalid_mask)
    invalid_fraction = invalid_summary["invalid_pixel_fraction"]

    if invalid_summary["invalid_pixel_count"] > 0:
        warnings_list.append(
            make_warning(
                "INVALID_FLATFIELD_DENOMINATOR",
                f"{invalid_summary['invalid_pixel_count']} pixels ({invalid_fraction * 100.0:.3f}%) had "
                f"(F - D) <= {denom_eps_rel * 100.0:.2f}% of max(F - D) and were left uncorrected (gain = 1.0).",
            )
        )

    if raw_f.ndim == 2:
        corrected_out = corr_channels[0]
        gain_out = gain_channels[0]
    else:
        corrected_out = np.stack(corr_channels, axis=2)
        gain_out = np.stack(gain_channels, axis=2)

    params = {
        "mode": "flatfield",
        "modality": modality,
        "norm_stat": norm_stat,
        "norm_factors_per_channel": norm_factors,
        "dark_subtracted": dark_img is not None,
        "keep_offset": keep_offset,
        "zero_offset_changed": (dark_img is not None) and (not keep_offset),
        "dark_offsets_added": dark_offsets,
        "denom_eps_rel": denom_eps_rel,
        "allow_resize": allow_resize,
        "clip_negative": clip_negative,
        "CrossChannelQuantitativeNote": (
            "Each channel is normalized by its own reference median/mean(F - D); "
            "relative intensity across different channels is preserved only if calibrated against a radiometric standard."
        ),
    }

    return ModeAResult(
        corrected_float=corrected_out,
        gain_map=gain_out,
        invalid_mask=combined_invalid_mask,
        invalid_fraction=invalid_fraction,
        invalid_spatial_summary=invalid_summary,
        negative_after_dark_count=neg_count,
        negative_after_dark_fraction=neg_frac,
        norm_factors=norm_factors,
        dark_offsets_added=dark_offsets,
        ref_diagnostics=ref_diag,
        warnings=warnings_list,
        parameters=params,
    )
