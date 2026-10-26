"""
Stage 3: Mode B — Pseudo-Flat-Field Illumination Correction
-----------------------------------------------------------
Approximate low-frequency background estimation for brightfield microscopy
when no physical instrument flat-field reference F is available.
NOT an instrument flat-field calibration.

Key Features:
  - Pixel-scale smoothing parameter `--sigma-px` (default: min(H, W) / 8.0)
  - Foreground-resistant background estimation (`masked_gaussian` default; `gaussian` optional with warning)
  - Shared luminance gain field for RGB images by default (preserving RGB channel ratios);
    `--per-channel` optional with warning
  - Native dynamic range preservation (`T = median(bg)` by default, or explicit `--target-val`)
  - Low-background noise amplification protection (`--min-bg-rel`) and gain ceiling (`--max-gain`)
  - Explicit `reflect` boundary handling
  - Automatic rejection of fluorescence / dark-background sparse-bright-object images unless `--force` is set
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from io_utils import LoadedImage, inspect_loaded_array, resolve_sigma_px, make_warning


@dataclass
class ModeBResult:
    """Result container for Mode B pseudo-flat-field correction."""
    corrected_float: np.ndarray            # (H, W) or (H, W, 3) float32
    bg_estimate: np.ndarray                # (H, W) or (H, W, 3) float32 estimated background
    gain_map: np.ndarray                   # (H, W) or (H, W, 3) float32 applied multiplicative gain
    foreground_mask: np.ndarray            # (H, W) bool mask of pixels excluded from background estimation
    low_bg_mask: np.ndarray                # (H, W) bool mask of pixels where bg <= min_bg_rel * max(bg)
    capped_gain_mask: np.ndarray           # (H, W) bool mask of pixels where raw gain > max_gain
    low_bg_fraction: float
    capped_gain_fraction: float
    foreground_excluded_fraction: float
    target_values: List[float]             # Normalization target T used (scalar list of length 1 or 3)
    sigma_px: float
    sigma_source: str
    warnings: List[Dict[str, str]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


def detect_dark_background_or_fluorescence(arr: np.ndarray, bit_depth: int, detected_12bit: bool) -> bool:
    """
    Detect whether an image has a dark background with sparse bright foreground objects
    (characteristic of fluorescence or dark-field microscopy).
    """
    lum = arr.mean(axis=2) if arr.ndim == 3 else arr.astype(np.float32)
    med = float(np.median(lum))
    p99 = float(np.percentile(lum, 99.0))
    p10 = float(np.percentile(lum, 10.0))
    nominal_full = 4095.0 if detected_12bit else (65535.0 if bit_depth == 16 else 255.0)

    if med < 0.12 * nominal_full and p99 > max(2.2 * med, p10 + 0.05 * nominal_full):
        return True
    return False


def _fit_robust_polynomial_trend(
    field_2d: np.ndarray,
    deg: int = 4,
    n_iter: int = 6,
    sigma_noise: float = 1.0,
) -> np.ndarray:
    """
    Fit a low-degree 2D polynomial illumination trend via asymmetric Iteratively Reweighted Least Squares (IRLS).
    Downweights dark absorbing foreground structures and bright hot-pixel spikes while tracking smooth
    large-scale illumination surfaces (e.g. cos^4 vignetting or tilted planes) right up to image corners
    without reflection boundary slope bias.
    """
    h, w = field_2d.shape
    sy = max(1, h // 64)
    sx = max(1, w // 64)
    ys, xs = np.mgrid[0:h:sy, 0:w:sx]
    yn = (ys - h / 2.0) / max(h / 2.0, 1.0)
    xn = (xs - w / 2.0) / max(w / 2.0, 1.0)

    exponents = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    sub_terms = [(xn ** i) * (yn ** j) for (i, j) in exponents]
    A = np.stack([t.ravel() for t in sub_terms], axis=1)
    z = field_2d[0:h:sy, 0:w:sx].ravel().astype(np.float64)

    # Guard against degenerate small arrays where samples < basis terms
    if z.size <= len(exponents):
        return np.full_like(field_2d, float(np.median(field_2d)), dtype=np.float32)

    weights = np.ones_like(z, dtype=np.float64)
    coeffs = np.zeros(len(exponents), dtype=np.float64)
    for _ in range(n_iter):
        Aw = A * weights[:, None]
        coeffs, _, _, _ = np.linalg.lstsq(Aw, z * weights, rcond=None)
        pred = A @ coeffs
        res = z - pred
        med_res = float(np.median(res))
        mad_scale = float(1.4826 * np.median(np.abs(res - med_res)))
        scale = max(sigma_noise, mad_scale, 0.01 * max(float(np.median(pred)), 1.0))

        w_new = np.ones_like(res)
        dark = res < (-2.0 * scale)
        w_new[dark] = np.exp(-((res[dark] / (2.0 * scale)) ** 2)) * 0.04
        bright = res > (3.5 * scale)
        w_new[bright] = 0.0
        if np.sum(w_new) < len(exponents) + 2:
            break
        weights = w_new

    yy, xx = np.mgrid[0:h, 0:w]
    y_full = (yy - h / 2.0) / max(h / 2.0, 1.0)
    x_full = (xx - w / 2.0) / max(w / 2.0, 1.0)
    poly_surface = np.zeros((h, w), dtype=np.float64)
    for c, (i, j) in zip(coeffs, exponents):
        if c != 0.0:
            poly_surface += c * (x_full ** i) * (y_full ** j)
    return poly_surface.astype(np.float32)


def estimate_background_2d(
    field_2d: np.ndarray,
    sigma_px: float,
    method: str = "masked_gaussian",
    border_mode: str = "reflect",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate smooth low-frequency illumination background from a 2D intensity field.
    Returns `(bg_2d, foreground_mask_2d)`.

    Methods:
      - `masked_gaussian` (default): Detrended foreground-masked normalized Gaussian convolution.
        1. Fits a smooth 2D polynomial trend via asymmetric IRLS to capture global vignetting/tilt
           without boundary reflection slope bias.
        2. Identifies localized foreground structures (`res < -threshold`) and hot-pixel spikes
           (`res > +threshold`) relative to the trend, enforcing scale consistency at `sigma_px`
           (structures much wider than `sigma_px` where local mask density > 0.70 are not bridged).
        3. Applies normalized Gaussian convolution (`G_sigma(W * res) / G_sigma(W)`) with `border_mode`
           on pure background residuals and adds back the trend.
      - `gaussian`: Naive unmasked Gaussian filter `G_sigma(I)`.
    """
    f = field_2d.astype(np.float32)
    h, w = f.shape

    if method == "gaussian":
        bg = gaussian_filter(f, sigma=sigma_px, mode=border_mode)
        fg_mask = np.zeros((h, w), dtype=bool)
        return bg.astype(np.float32), fg_mask

    if method != "masked_gaussian":
        raise ValueError(f"Unsupported background estimation method '{method}'. Choose 'masked_gaussian' or 'gaussian'.")

    # Check if image is completely uniform
    if float(f.max() - f.min()) == 0.0:
        return f.copy(), np.zeros((h, w), dtype=bool)

    fine_smooth = gaussian_filter(f, sigma=1.0, mode=border_mode)
    hf_diff = f - fine_smooth
    sigma_noise = float(1.4826 * np.median(np.abs(hf_diff - np.median(hf_diff))))

    poly = _fit_robust_polynomial_trend(f, deg=4, n_iter=6, sigma_noise=sigma_noise)
    res_field = f - poly
    thresh_dark = np.maximum(3.2 * sigma_noise, 0.03 * np.abs(poly))
    thresh_hot = np.maximum(4.5 * sigma_noise, 0.06 * np.abs(poly))

    fg_raw = (res_field < -thresh_dark) | (res_field > thresh_hot)
    if not np.any(fg_raw):
        num = gaussian_filter(res_field, sigma=sigma_px, mode=border_mode)
        return (poly + num).astype(np.float32), fg_raw

    fg_dilated = binary_dilation(fg_raw, structure=np.ones((5, 5), dtype=bool), iterations=1)
    # Scale consistency: if a dark structure's radius is much larger than sigma_px,
    # its interior density at scale sigma_px exceeds 0.70 and cannot be bridged at scale sigma_px
    mask_density = gaussian_filter(fg_dilated.astype(np.float32), sigma=sigma_px, mode=border_mode)
    fg_mask = fg_dilated & (mask_density <= 0.70)

    w_bg = (~fg_mask).astype(np.float32)
    num = gaussian_filter(w_bg * res_field, sigma=sigma_px, mode=border_mode)
    den = gaussian_filter(w_bg, sigma=sigma_px, mode=border_mode)
    delta_bg = np.where(den > 0.05, num / np.maximum(den, 0.05), res_field)
    bg_est = (poly + delta_bg).astype(np.float32)
    return bg_est, fg_mask


def pseudo_flatfield_correct_mode_b(
    sample: Union[LoadedImage, np.ndarray],
    sigma_px: Optional[float] = None,
    deprecated_mag: Optional[str] = None,
    bg_method: str = "masked_gaussian",
    per_channel: bool = False,
    target_val: Optional[float] = None,
    min_bg_rel: float = 0.01,
    max_gain: float = 4.0,
    border_mode: str = "reflect",
    modality: str = "brightfield",
    force: bool = False,
) -> ModeBResult:
    """
    Execute Mode B pseudo-flat-field correction.
    """
    sample_img = sample if isinstance(sample, LoadedImage) else inspect_loaded_array(sample, "<sample>")
    raw_f = sample_img.data.astype(np.float32)

    if not np.all(np.isfinite(raw_f)):
        raise ValueError("Input image contains non-finite (NaN or Inf) values.")
    if max_gain < 1.0 or not np.isfinite(max_gain):
        raise ValueError(f"--max-gain must be >= 1.0 and finite, got {max_gain}.")
    if min_bg_rel <= 0.0 or min_bg_rel >= 1.0 or not np.isfinite(min_bg_rel):
        raise ValueError(f"--min-bg-rel must be in (0, 1), got {min_bg_rel}.")
    if border_mode not in ("reflect", "nearest", "mirror", "wrap"):
        raise ValueError(f"Unsupported border_mode '{border_mode}'.")

    warnings_list: List[Dict[str, str]] = []

    is_dark_bg = detect_dark_background_or_fluorescence(
        raw_f, sample_img.bit_depth, sample_img.detected_12bit_container
    )
    if modality.lower() == "fluorescence" or is_dark_bg:
        if not force:
            reason = (
                "modality='fluorescence' was specified"
                if modality.lower() == "fluorescence"
                else "dark background with sparse bright features (fluorescence/dark-field pattern) was detected"
            )
            raise ValueError(
                f"Mode B (pseudo-flat-field) rejected because {reason}. "
                "Dividing a dark-background image by its smoothed background amplifies background noise and distorts "
                "emitter intensities. Use Mode A with a measured flat-field reference, or pass --force to override."
            )
        warnings_list.append(
            make_warning(
                "FLUORESCENCE_OR_DARK_BG_FORCED",
                "Mode B pseudo-flat-field was forced (--force) on a fluorescence or dark-background image. "
                "Output intensities are NOT suitable for quantitative comparison.",
            )
        )

    eff_sigma, sigma_source, sigma_warnings = resolve_sigma_px(
        raw_f.shape, sigma_px=sigma_px, deprecated_mag=deprecated_mag
    )
    warnings_list.extend(sigma_warnings)

    if bg_method == "gaussian":
        warnings_list.append(
            make_warning(
                "NAIVE_GAUSSIAN_BG",
                "Using unmasked Gaussian background estimation (--bg-method gaussian): dense dark biological structures "
                "will depress the local background estimate, reducing foreground contrast and creating bright halos.",
            )
        )

    if sample_img.is_rgb and per_channel:
        warnings_list.append(
            make_warning(
                "PER_CHANNEL_COLOR_SHIFT",
                "Per-channel pseudo-flat-field (--per-channel) estimates separate R, G, B backgrounds and normalizes "
                "each channel independently, which alters RGB channel ratios and shifts staining colors/thresholds.",
            )
        )

    if target_val is not None and sample_img.bit_depth == 16 and target_val <= 255.0:
        warnings_list.append(
            make_warning(
                "LOW_TARGET_VAL_FOR_16BIT",
                f"Explicit --target-val={target_val} is <= 255 on a 16-bit (uint16) input image; "
                "this compresses 16-bit intensities down to an 8-bit numeric range.",
            )
        )

    if not sample_img.is_rgb:
        bg_2d, fg_mask = estimate_background_2d(
            raw_f, sigma_px=eff_sigma, method=bg_method, border_mode=border_mode
        )
        max_bg = float(bg_2d.max())
        if max_bg <= 0.0:
            low_bg_mask = np.ones(bg_2d.shape, dtype=bool)
            capped_mask = np.zeros(bg_2d.shape, dtype=bool)
            t_val = 0.0 if target_val is None else float(target_val)
            gain_2d = np.ones_like(bg_2d, dtype=np.float32)
        else:
            thresh = min_bg_rel * max_bg
            low_bg_mask = bg_2d <= thresh
            valid_bg = bg_2d[~low_bg_mask]
            t_val = float(target_val) if target_val is not None else (float(np.median(valid_bg)) if valid_bg.size > 0 else max_bg)
            safe_bg = np.where(low_bg_mask, 1.0, bg_2d)
            raw_gain = np.where(low_bg_mask, 1.0, t_val / safe_bg)
            capped_mask = (~low_bg_mask) & (raw_gain > max_gain)
            gain_2d = np.minimum(raw_gain, max_gain).astype(np.float32)

        corr_out = (raw_f * gain_2d).astype(np.float32)
        bg_out = bg_2d
        gain_out = gain_2d
        target_values = [t_val]

    elif not per_channel:
        lum = raw_f.mean(axis=2)
        bg_lum, fg_mask = estimate_background_2d(
            lum, sigma_px=eff_sigma, method=bg_method, border_mode=border_mode
        )
        max_bg = float(bg_lum.max())
        if max_bg <= 0.0:
            low_bg_mask = np.ones(bg_lum.shape, dtype=bool)
            capped_mask = np.zeros(bg_lum.shape, dtype=bool)
            t_val = 0.0 if target_val is None else float(target_val)
            gain_2d = np.ones_like(bg_lum, dtype=np.float32)
        else:
            thresh = min_bg_rel * max_bg
            low_bg_mask = bg_lum <= thresh
            valid_bg = bg_lum[~low_bg_mask]
            t_val = float(target_val) if target_val is not None else (float(np.median(valid_bg)) if valid_bg.size > 0 else max_bg)
            safe_bg = np.where(low_bg_mask, 1.0, bg_lum)
            raw_gain = np.where(low_bg_mask, 1.0, t_val / safe_bg)
            capped_mask = (~low_bg_mask) & (raw_gain > max_gain)
            gain_2d = np.minimum(raw_gain, max_gain).astype(np.float32)

        corr_out = (raw_f * gain_2d[:, :, None]).astype(np.float32)
        bg_out = bg_lum
        gain_out = gain_2d
        target_values = [t_val]

    else:
        bg_channels: List[np.ndarray] = []
        gain_channels: List[np.ndarray] = []
        fg_masks: List[np.ndarray] = []
        low_masks: List[np.ndarray] = []
        cap_masks: List[np.ndarray] = []
        corr_channels: List[np.ndarray] = []
        target_values = []

        for c in range(3):
            ch = raw_f[:, :, c]
            bg_c, fg_c = estimate_background_2d(
                ch, sigma_px=eff_sigma, method=bg_method, border_mode=border_mode
            )
            max_bg = float(bg_c.max())
            if max_bg <= 0.0:
                low_c = np.ones(bg_c.shape, dtype=bool)
                cap_c = np.zeros(bg_c.shape, dtype=bool)
                t_val = 0.0 if target_val is None else float(target_val)
                gain_c = np.ones_like(bg_c, dtype=np.float32)
            else:
                thresh = min_bg_rel * max_bg
                low_c = bg_c <= thresh
                valid_bg = bg_c[~low_c]
                t_val = float(target_val) if target_val is not None else (float(np.median(valid_bg)) if valid_bg.size > 0 else max_bg)
                safe_bg = np.where(low_c, 1.0, bg_c)
                raw_gain = np.where(low_c, 1.0, t_val / safe_bg)
                cap_c = (~low_c) & (raw_gain > max_gain)
                gain_c = np.minimum(raw_gain, max_gain).astype(np.float32)

            bg_channels.append(bg_c)
            gain_channels.append(gain_c)
            fg_masks.append(fg_c)
            low_masks.append(low_c)
            cap_masks.append(cap_c)
            corr_channels.append((ch * gain_c).astype(np.float32))
            target_values.append(t_val)

        corr_out = np.stack(corr_channels, axis=2)
        bg_out = np.stack(bg_channels, axis=2)
        gain_out = np.stack(gain_channels, axis=2)
        fg_mask = np.logical_or.reduce(fg_masks)
        low_bg_mask = np.logical_or.reduce(low_masks)
        capped_mask = np.logical_or.reduce(cap_masks)

    total_2d_px = float(raw_f.shape[0] * raw_f.shape[1])
    low_bg_frac = float(np.count_nonzero(low_bg_mask) / total_2d_px)
    capped_frac = float(np.count_nonzero(capped_mask) / total_2d_px)
    fg_frac = float(np.count_nonzero(fg_mask) / total_2d_px)

    if low_bg_frac > 0.0:
        warnings_list.append(
            make_warning(
                "LOW_BG_REGION",
                f"{low_bg_frac * 100.0:.3f}% of pixels had estimated background <= {min_bg_rel * 100.0:.2f}% "
                "of max background; gain was set to 1.0 in those regions to prevent noise amplification.",
            )
        )

    if capped_frac > 0.0:
        warnings_list.append(
            make_warning(
                "GAIN_CAPPED",
                f"{capped_frac * 100.0:.3f}% of pixels exceeded --max-gain={max_gain} and were capped at {max_gain}.",
            )
        )

    params = {
        "mode": "pseudo-flat-field",
        "method_description": "pseudo-flat-field (approximate low-frequency background estimation, not instrument flat-field calibration)",
        "bg_method": bg_method,
        "sigma_px": eff_sigma,
        "sigma_source": sigma_source,
        "per_channel": per_channel,
        "color_gain_mode": "per_channel_independent" if (sample_img.is_rgb and per_channel) else "shared_luminance_gain",
        "target_val_overridden": target_val is not None,
        "target_values_used": target_values,
        "min_bg_rel": min_bg_rel,
        "max_gain": max_gain,
        "border_mode": border_mode,
        "modality": modality,
        "force": force,
    }

    return ModeBResult(
        corrected_float=corr_out,
        bg_estimate=bg_out,
        gain_map=gain_out,
        foreground_mask=fg_mask,
        low_bg_mask=low_bg_mask,
        capped_gain_mask=capped_mask,
        low_bg_fraction=low_bg_frac,
        capped_gain_fraction=capped_frac,
        foreground_excluded_fraction=fg_frac,
        target_values=target_values,
        sigma_px=eff_sigma,
        sigma_source=sigma_source,
        warnings=warnings_list,
        parameters=params,
    )
