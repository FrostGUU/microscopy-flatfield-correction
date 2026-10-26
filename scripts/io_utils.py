"""
Stage 1: Bit-Depth Preserving Image I/O, Validation, Quantization, and Metadata
-------------------------------------------------------------------------------
Supports:
  - 2D grayscale (`uint8`, `uint16`)
  - 3D RGB `(H, W, 3)` (`uint8`, `uint16`)
  - 3D RGBA `(H, W, 4)` (`uint8`, `uint16`, Alpha preserved untouched)
Rejects:
  - Multi-page / multi-series 3D volumetric TIFFs
  - Channel-first `(C, H, W)` arrays
  - Unsupported bit depths or floating-point inputs
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import tifffile

SOFTWARE_VERSION = "2.0.0-rigorous"

MAG_TO_SIGMA_PX = {
    "4x": 75.0,
    "10x": 60.0,
    "20x": 45.0,
    "40x": 40.0,
    "100x": 40.0,
}


@dataclass
class LoadedImage:
    """Container for a validated input image and its I/O diagnostics."""
    data: np.ndarray                  # (H, W) or (H, W, 3) in RGB order, uint8 or uint16
    alpha: Optional[np.ndarray]       # (H, W) if input was RGBA, else None
    orig_dtype: np.dtype              # np.uint8 or np.uint16
    orig_shape: Tuple[int, ...]       # Original validated shape
    is_rgb: bool                      # True if 3-channel RGB (or 4-channel RGBA)
    has_alpha: bool                   # True if 4-channel RGBA
    bit_depth: int                    # 8 or 16
    detected_12bit_container: bool    # True if uint16 with max in (255, 4095]
    source_path: Optional[str] = None
    io_warnings: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class QuantizationResult:
    """Container for quantized output array and clipping statistics."""
    array: np.ndarray                 # Quantized uint8/uint16 (or float32) output array (with alpha reattached if present)
    core_array: np.ndarray            # Quantized output without alpha channel
    clipped_low_count: int
    clipped_high_count: int
    total_pixels: int
    clipped_low_fraction: float
    clipped_high_fraction: float
    clipped_total_fraction: float
    warnings: List[Dict[str, str]] = field(default_factory=list)


def make_warning(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def validate_array_shape_and_dtype(arr: np.ndarray, source_desc: str = "input") -> Tuple[bool, bool]:
    """
    Validate that array is (H, W), (H, W, 3), or (H, W, 4) and dtype is uint8 or uint16.
    Returns (is_rgb, has_alpha).
    Raises ValueError with clear explanation otherwise.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"[{source_desc}] Expected numpy.ndarray, got {type(arr).__name__}.")

    if np.iscomplexobj(arr) or arr.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError(
            f"[{source_desc}] Unsupported data type '{arr.dtype}'. "
            f"Only uint8 and uint16 integer microscopy images are supported; "
            f"floating-point, signed, or 32-bit types are rejected without silent conversion."
        )

    if arr.ndim == 2:
        h, w = arr.shape
        if h < 2 or w < 2:
            raise ValueError(f"[{source_desc}] Image dimensions {arr.shape} are too small (minimum 2x2).")
        return False, False

    if arr.ndim == 3:
        h, w, c = arr.shape
        if c not in (3, 4):
            if h in (3, 4) and w > 4 and c > 4:
                raise ValueError(
                    f"[{source_desc}] Unsupported channel-first shape {arr.shape} (looks like (C, H, W)). "
                    f"Only planar (H, W) grayscale, (H, W, 3) RGB, or (H, W, 4) RGBA are accepted."
                )
            raise ValueError(
                f"[{source_desc}] Unsupported 3D shape {arr.shape}. "
                f"Multi-page Z-stacks/time-series or {c}-channel arrays are not supported."
            )
        if h in (3, 4) and w > 16 and c in (3, 4):
            raise ValueError(
                f"[{source_desc}] Ambiguous or channel-first shape {arr.shape} rejected."
            )
        if h < 2 or w < 2:
            raise ValueError(f"[{source_desc}] Image spatial dimensions ({h}, {w}) are too small.")
        return True, (c == 4)

    raise ValueError(
        f"[{source_desc}] Unsupported array dimensionality {arr.ndim}D with shape {arr.shape}. "
        f"Only 2D (H, W) grayscale or 3D (H, W, 3)/(H, W, 4) RGB/RGBA images are accepted."
    )


def inspect_loaded_array(
    arr: np.ndarray,
    source_path: Optional[str] = None,
    extra_warnings: Optional[List[Dict[str, str]]] = None,
) -> LoadedImage:
    """Validate an already-loaded array (in RGB/RGBA or 2D grayscale order) and build LoadedImage."""
    desc = source_path if source_path else "array"
    is_rgb, has_alpha = validate_array_shape_and_dtype(arr, source_desc=desc)
    warnings_list: List[Dict[str, str]] = list(extra_warnings) if extra_warnings else []

    orig_shape = arr.shape
    orig_dtype = arr.dtype
    bit_depth = 16 if orig_dtype == np.dtype("uint16") else 8

    if has_alpha:
        alpha = arr[:, :, 3].copy()
        core_data = arr[:, :, :3].copy()
        warnings_list.append(
            make_warning(
                "ALPHA_CHANNEL_PRESERVED",
                "4-channel RGBA input detected: Alpha channel is excluded from illumination correction and preserved unmodified.",
            )
        )
    else:
        alpha = None
        core_data = arr.copy()

    max_val = int(core_data.max()) if core_data.size > 0 else 0
    detected_12bit = False
    if bit_depth == 16 and 255 < max_val <= 4095:
        detected_12bit = True
        warnings_list.append(
            make_warning(
                "CONTAINER_12BIT_IN_16BIT",
                f"16-bit image has maximum intensity {max_val} <= 4095 (likely 12-bit sensor data in a 16-bit container). "
                "Original intensity scale is preserved without stretching to 65535.",
            )
        )

    if bit_depth == 8:
        warnings_list.append(
            make_warning(
                "GAMMA_ENCODED_8BIT",
                "8-bit input detected: standard 8-bit images are commonly gamma-encoded (sRGB), whereas multiplicative illumination models assume linear sensor response.",
            )
        )

    return LoadedImage(
        data=core_data,
        alpha=alpha,
        orig_dtype=orig_dtype,
        orig_shape=orig_shape,
        is_rgb=is_rgb,
        has_alpha=has_alpha,
        bit_depth=bit_depth,
        detected_12bit_container=detected_12bit,
        source_path=source_path,
        io_warnings=warnings_list,
    )


def read_image(fpath: str) -> LoadedImage:
    """
    Read a microscopy image preserving bit depth (uint8/uint16), channels, and supporting non-ASCII paths.
    Uses `tifffile` for `.tif`/`.tiff` and `cv2.imdecode(..., IMREAD_UNCHANGED)` for `.png`/`.jpg`/`.jpeg`.
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Input image file does not exist: {fpath}")

    ext = os.path.splitext(fpath)[1].lower()
    extra_warnings: List[Dict[str, str]] = []

    if ext in (".tif", ".tiff"):
        try:
            with open(fpath, "rb") as fobj:
                with tifffile.TiffFile(fobj) as tif:
                    if len(tif.pages) > 1:
                        raise ValueError(
                            f"[{fpath}] Multi-page TIFF with {len(tif.pages)} pages is not supported. "
                            "Only single-plane 2D grayscale, RGB, or RGBA TIFFs are accepted."
                        )
                    page = tif.pages[0]
                    # Check planar configuration or channel-first before reading
                    arr = page.asarray()
        except ValueError as e:
            msg = str(e)
            if "imagecodecs" in msg.lower() or "compression" in msg.lower() or "decompress" in msg.lower():
                raise RuntimeError(
                    f"[{fpath}] Compressed TIFF requires the 'imagecodecs' package to decode ({msg}). "
                    "Please install it via: pip install imagecodecs"
                ) from e
            raise
    elif ext in (".png", ".jpg", ".jpeg"):
        if ext in (".jpg", ".jpeg"):
            extra_warnings.append(
                make_warning(
                    "JPEG_INPUT",
                    "Lossy 8-bit JPEG input detected: block compression artifacts and chroma subsampling make JPEG unsuitable for quantitative microscopy analysis.",
                )
            )
        raw_bytes = np.fromfile(fpath, dtype=np.uint8)
        if raw_bytes.size == 0:
            raise ValueError(f"[{fpath}] File is empty or unreadable.")
        decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"[{fpath}] Failed to decode image file.")
        if decoded.ndim == 3 and decoded.shape[2] == 3:
            arr = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        elif decoded.ndim == 3 and decoded.shape[2] == 4:
            arr = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGBA)
        else:
            arr = decoded
    else:
        raise ValueError(
            f"[{fpath}] Unsupported file extension '{ext}'. Supported extensions: .tif, .tiff, .png, .jpg, .jpeg"
        )

    return inspect_loaded_array(arr, source_path=fpath, extra_warnings=extra_warnings)


def check_overwrite_safety(input_path: str, output_path: str, overwrite: bool = False) -> None:
    """
    Prevent accidental overwriting of input files or existing output files unless overwrite=True.
    Even with overwrite=True, writing directly back to the exact same input file path is rejected
    unless explicitly forced, and by default any path collision raises ValueError / FileExistsError.
    """
    abs_in = os.path.normcase(os.path.abspath(input_path))
    abs_out = os.path.normcase(os.path.abspath(output_path))
    if abs_in == abs_out and not overwrite:
        raise ValueError(
            f"Refusing to overwrite input file in-place ({input_path}). "
            "Specify a distinct output path or pass --overwrite explicitly."
        )
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output file already exists ({output_path}). "
            "Pass --overwrite to allow overwriting existing files."
        )


def check_dir_overwrite_safety(input_dir: str, output_dir: str, overwrite: bool = False) -> None:
    """Prevent batch output_dir from colliding with input_dir unless overwrite=True."""
    abs_in = os.path.normcase(os.path.abspath(input_dir))
    abs_out = os.path.normcase(os.path.abspath(output_dir))
    if abs_in == abs_out and not overwrite:
        raise ValueError(
            f"Refusing to use the same directory for --input-dir and --output-dir ({input_dir}). "
            "Specify a separate output directory or pass --overwrite."
        )


def quantize_and_clip(
    arr_float: np.ndarray,
    orig_dtype: np.dtype,
    alpha: Optional[np.ndarray] = None,
    float_output: bool = False,
    clip_warn_threshold: float = 0.001,
) -> QuantizationResult:
    """
    Convert float32 corrected array back to orig_dtype (uint8 or uint16) using round-to-nearest (`np.rint`),
    or keep float32 if `float_output=True`.
    Counts and reports out-of-range pixels before clipping.
    """
    if not np.all(np.isfinite(arr_float)):
        raise ValueError("Internal error: non-finite (NaN or Inf) values encountered prior to quantization.")

    warnings_list: List[Dict[str, str]] = []
    max_dtype_val = 65535.0 if orig_dtype == np.dtype("uint16") else 255.0
    total_elements = int(arr_float.size)

    low_mask = arr_float < 0.0
    high_mask = arr_float > max_dtype_val
    clipped_low_count = int(np.count_nonzero(low_mask))
    clipped_high_count = int(np.count_nonzero(high_mask))
    clipped_low_frac = clipped_low_count / float(total_elements) if total_elements > 0 else 0.0
    clipped_high_frac = clipped_high_count / float(total_elements) if total_elements > 0 else 0.0
    clipped_total_frac = (clipped_low_count + clipped_high_count) / float(total_elements) if total_elements > 0 else 0.0

    if float_output:
        core_out = arr_float.astype(np.float32)
        warnings_list.append(
            make_warning(
                "FLOAT32_OUTPUT_COMPATIBILITY",
                f"Float32 TIFF output enabled (range [{ float(core_out.min()):.2f}, {float(core_out.max()):.2f}]). "
                "Readable by ImageJ/Fiji and scientific Python tools, but standard OS image viewers may not render float32 TIFFs.",
            )
        )
        if alpha is not None:
            alpha_f = alpha.astype(np.float32)
            full_out = np.dstack([core_out, alpha_f])
        else:
            full_out = core_out
    else:
        rounded = np.rint(arr_float)
        clipped = np.clip(rounded, 0.0, max_dtype_val)
        core_out = clipped.astype(orig_dtype)
        if alpha is not None:
            full_out = np.dstack([core_out, alpha.astype(orig_dtype)])
        else:
            full_out = core_out

    if clipped_total_frac > clip_warn_threshold:
        warnings_list.append(
            make_warning(
                "CLIPPED_PIXELS",
                f"{clipped_total_frac * 100.0:.3f}% of output values ({clipped_low_count} low < 0, "
                f"{clipped_high_count} high > {int(max_dtype_val)}) exceeded the {orig_dtype} range and were clipped "
                f"(threshold: {clip_warn_threshold * 100.0:.3f}%).",
            )
        )

    return QuantizationResult(
        array=full_out,
        core_array=core_out,
        clipped_low_count=clipped_low_count,
        clipped_high_count=clipped_high_count,
        total_pixels=total_elements,
        clipped_low_fraction=clipped_low_frac,
        clipped_high_fraction=clipped_high_frac,
        clipped_total_fraction=clipped_total_frac,
        warnings=warnings_list,
    )


def write_image_and_metadata(
    fpath: str,
    arr: np.ndarray,
    metadata: Dict[str, Any],
    quality: int = 95,
    write_sidecar_for_tiff: bool = False,
) -> Optional[str]:
    """
    Write output image preserving bit-depth and embed/write metadata:
      - `.tif` / `.tiff`: embeds JSON metadata in TIFF `ImageDescription` tag.
      - `.png` / `.jpg` / `.jpeg`: writes image via `cv2.imencode` (supporting uint16 PNG and Unicode paths)
        and writes a companion `<stem>.json` sidecar file.
    Returns path to sidecar JSON file if written, else None.
    """
    ext = os.path.splitext(fpath)[1].lower()
    out_dir = os.path.dirname(os.path.abspath(fpath))
    os.makedirs(out_dir, exist_ok=True)

    meta_json_str = json.dumps(metadata, ensure_ascii=False, indent=2)
    sidecar_path: Optional[str] = None

    if ext in (".tif", ".tiff"):
        photometric = "rgb" if (arr.ndim == 3 and arr.shape[2] in (3, 4)) else "minisblack"
        extrasamples = [tifffile.EXTRASAMPLE.UNASSALPHA] if (arr.ndim == 3 and arr.shape[2] == 4) else None
        with open(fpath, "wb") as fobj:
            if extrasamples:
                tifffile.imwrite(
                    fobj,
                    arr,
                    photometric=photometric,
                    extrasamples=extrasamples,
                    description=meta_json_str,
                )
            else:
                tifffile.imwrite(
                    fobj,
                    arr,
                    photometric=photometric,
                    description=meta_json_str,
                )
        if write_sidecar_for_tiff:
            sidecar_path = os.path.splitext(fpath)[0] + ".json"
            with open(sidecar_path, "w", encoding="utf-8") as sf:
                sf.write(meta_json_str)
    elif ext in (".png", ".jpg", ".jpeg"):
        if arr.dtype == np.dtype("float32"):
            raise ValueError(f"Float32 output is only supported for .tif/.tiff files, not '{ext}'.")
        if ext in (".jpg", ".jpeg") and arr.dtype == np.dtype("uint16"):
            raise ValueError("Cannot save 16-bit (uint16) image as 8-bit lossy JPEG. Use .tif or .png instead.")

        if arr.ndim == 3 and arr.shape[2] == 3:
            enc_arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            enc_arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        else:
            enc_arr = arr

        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
        ok, buf = cv2.imencode(ext, enc_arr, params)
        if not ok:
            raise IOError(f"Failed to encode output image: {fpath}")
        buf.tofile(fpath)

        sidecar_path = os.path.splitext(fpath)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as sf:
            sf.write(meta_json_str)
    else:
        raise ValueError(f"Unsupported output extension '{ext}'. Use .tif, .tiff, .png, or .jpg.")

    return sidecar_path


def resolve_sigma_px(
    image_shape: Tuple[int, ...],
    sigma_px: Optional[float] = None,
    deprecated_mag: Optional[str] = None,
) -> Tuple[float, str, List[Dict[str, str]]]:
    """
    Resolve background smoothing scale in pixels (`sigma_px`).
    - Rejects `--mag auto`.
    - Supports `--mag` only as a deprecated alias with warning.
    - Defaults to `max(15.0, round(min(H, W) / 8.0, 1))` when neither `--sigma-px` nor `--mag` is given.
    Returns `(effective_sigma_px, resolution_source, warnings_list)`.
    """
    warnings_list: List[Dict[str, str]] = []
    h, w = image_shape[:2]

    if deprecated_mag is not None:
        m_clean = deprecated_mag.strip().lower()
        if m_clean == "auto":
            raise ValueError(
                "'--mag auto' has been removed because inferring objective magnification from folder names is unreliable. "
                "Please specify '--sigma-px <pixels>' explicitly (or omit '--mag' to use the resolution-scaled default)."
            )
        if m_clean not in MAG_TO_SIGMA_PX:
            raise ValueError(
                f"Unsupported '--mag' value '{deprecated_mag}'. Valid deprecated aliases: {list(MAG_TO_SIGMA_PX.keys())}. "
                "Prefer '--sigma-px <pixels>'."
            )
        if sigma_px is not None:
            warnings_list.append(
                make_warning(
                    "DEPRECATED_MAG_ARG",
                    f"'--mag {deprecated_mag}' is deprecated and ignored because '--sigma-px {sigma_px}' was explicitly provided.",
                )
            )
        else:
            mapped = MAG_TO_SIGMA_PX[m_clean]
            msg = (
                f"'--mag {deprecated_mag}' is deprecated because objective magnification alone does not define pixel scale "
                f"(camera pixel pitch, binning, and ROI cropping alter pixel dimensions). Mapped to --sigma-px={mapped}."
            )
            print(f"[DEPRECATION WARNING] {msg}", file=sys.stderr)
            warnings_list.append(make_warning("DEPRECATED_MAG_ARG", msg))
            return mapped, f"deprecated_mag:{m_clean}", warnings_list

    if sigma_px is not None:
        if sigma_px <= 0 or not np.isfinite(sigma_px):
            raise ValueError(f"'--sigma-px' must be a positive finite number, got {sigma_px}.")
        return float(sigma_px), "explicit_sigma_px", warnings_list

    default_sigma = max(15.0, round(min(h, w) / 8.0, 1))
    return default_sigma, "default_short_edge_div_8", warnings_list
