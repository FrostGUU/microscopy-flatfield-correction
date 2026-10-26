"""
Microscopy Illumination Correction CLI & Library (`correct_illumination.py`)
----------------------------------------------------------------------------
Supports two distinct correction modes:
  1. Mode A (`--mode flatfield`): True instrument flat-field calibration using measured
     flat-field reference F and optional dark-field D:
       I_corr = ((I_raw - D) / (F - D)) * norm(F - D) [+ median(D) if --keep-offset]
  2. Mode B (`--mode pseudo-flatfield`, default): Approximate low-frequency background
     estimation (`masked_gaussian` default; `gaussian` optional with warning) for brightfield
     visualization when no physical flat-field reference F is available.
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, Any, List, Optional, Sequence, Union
import numpy as np
import cv2
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from io_utils import (
    LoadedImage,
    QuantizationResult,
    read_image,
    inspect_loaded_array,
    check_overwrite_safety,
    check_dir_overwrite_safety,
    quantize_and_clip,
    write_image_and_metadata,
    MAG_TO_SIGMA_PX,
)
from flatfield_mode_a import flatfield_correct_mode_a
from pseudo_flatfield_mode_b import pseudo_flatfield_correct_mode_b
from qc_report import build_qc_report


# =============================================================================
# Legacy Baseline Functions (Preserved for Stage 0 & Stage 5 Benchmark Tests)
# =============================================================================

def legacy_imread_unicode(fpath: str) -> Optional[np.ndarray]:
    """[Legacy Baseline] Read image as 8-bit 3-channel BGR via cv2.IMREAD_COLOR."""
    if not os.path.exists(fpath):
        return None
    data = np.fromfile(fpath, dtype=np.uint8)
    if len(data) == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def legacy_imwrite_unicode(fpath: str, img_bgr: np.ndarray, quality: int = 95) -> bool:
    """[Legacy Baseline] Write 8-bit BGR image."""
    ext = os.path.splitext(fpath)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    if ext in [".jpg", ".jpeg"]:
        success, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            enc.tofile(fpath)
            return True
    elif ext in [".tif", ".tiff"]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(fpath)
        return True
    else:
        success, enc = cv2.imencode(ext if ext else ".png", img_bgr)
        if success:
            enc.tofile(fpath)
            return True
    return False


def get_sigma_for_mag(mag_str: Optional[str], default_sigma: float = 50.0) -> float:
    """[Deprecated Helper] Map magnification string to pixel sigma."""
    if not mag_str:
        return default_sigma
    m = mag_str.lower().strip()
    for k, v in MAG_TO_SIGMA_PX.items():
        if k in m:
            return v
    return default_sigma


def legacy_pseudo_flatfield_correction(
    img_bgr: np.ndarray, sigma: float = 50.0, target_val: float = 220.0
) -> np.ndarray:
    """
    [Legacy Baseline] Per-channel unmasked Gaussian division clipped to [0, 255] uint8.
    Preserved strictly for characterization and benchmark comparison tests.
    """
    img_f = img_bgr.astype(np.float32)
    if img_f.ndim == 2:
        bg = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        corr = np.clip((img_f / (bg + 1e-5)) * target_val, 0, 255)
        return corr.astype(np.uint8)
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    bg_b = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_r = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corr_b = np.clip((b / (bg_b + 1e-5)) * target_val, 0, 255)
    corr_g = np.clip((g / (bg_g + 1e-5)) * target_val, 0, 255)
    corr_r = np.clip((r / (bg_r + 1e-5)) * target_val, 0, 255)
    return np.stack([corr_b, corr_g, corr_r], axis=2).astype(np.uint8)


# Backward-compatible symbol aliases for characterization tests & legacy callers
imread_unicode = legacy_imread_unicode
imwrite_unicode = legacy_imwrite_unicode
pseudo_flatfield_correction = legacy_pseudo_flatfield_correction


# =============================================================================
# Core Rigorous Pipeline (Single & Batch)
# =============================================================================

def correct_image_pipeline(
    sample: Union[str, np.ndarray, LoadedImage],
    mode: str = "pseudo-flatfield",
    # Mode A parameters
    flatfield: Optional[Union[str, Sequence[str], np.ndarray, LoadedImage]] = None,
    darkfield: Optional[Union[str, Sequence[str], np.ndarray, LoadedImage]] = None,
    norm_stat: str = "median",
    keep_offset: bool = False,
    denom_eps_rel: float = 0.01,
    allow_resize: bool = False,
    clip_negative: bool = True,
    # Mode B parameters
    sigma_px: Optional[float] = None,
    deprecated_mag: Optional[str] = None,
    bg_method: str = "masked_gaussian",
    per_channel: bool = False,
    target_val: Optional[float] = None,
    min_bg_rel: float = 0.01,
    max_gain: float = 4.0,
    border_mode: str = "reflect",
    # Common parameters
    modality: str = "brightfield",
    force: bool = False,
    float_output: bool = False,
    clip_warn_threshold: float = 0.001,
    ground_truth_bg_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run the complete illumination correction pipeline (Mode A or Mode B), quantization,
    and independent QC report generation on a file path or in-memory array.
    Returns dict with keys: `output_array`, `float_array`, `gain_map`, `report`, `loaded`.
    """
    if isinstance(sample, str):
        loaded = read_image(sample)
    elif isinstance(sample, LoadedImage):
        loaded = sample
    else:
        loaded = inspect_loaded_array(sample, source_path="<in_memory_array>")

    mode_norm = mode.strip().lower().replace("_", "-")
    if mode_norm in ("flatfield", "flat-field", "mode-a", "a"):
        if flatfield is None:
            raise ValueError("Mode A ('flatfield') requires a flat-field reference (--flatfield).")
        res_a = flatfield_correct_mode_a(
            sample=loaded,
            flatfield=flatfield,
            darkfield=darkfield,
            norm_stat=norm_stat,
            keep_offset=keep_offset,
            denom_eps_rel=denom_eps_rel,
            allow_resize=allow_resize,
            clip_negative=clip_negative,
            modality=modality,
        )
        quant = quantize_and_clip(
            res_a.corrected_float,
            orig_dtype=loaded.orig_dtype,
            alpha=loaded.alpha,
            float_output=float_output,
            clip_warn_threshold=clip_warn_threshold,
        )
        extra_diag = {
            "flatfield_reference": res_a.ref_diagnostics,
            "invalid_denominator_summary": res_a.invalid_spatial_summary,
            "negative_after_dark_count": res_a.negative_after_dark_count,
            "negative_after_dark_fraction": res_a.negative_after_dark_fraction,
        }
        report = build_qc_report(
            loaded=loaded,
            quant=quant,
            gain_map=res_a.gain_map,
            mode_params=res_a.parameters,
            mode_warnings=res_a.warnings,
            protected_fraction=res_a.invalid_fraction,
            capped_gain_fraction=0.0,
            extra_diagnostics=extra_diag,
            ground_truth_bg_mask=ground_truth_bg_mask,
        )
        return {
            "output_array": quant.array,
            "float_array": res_a.corrected_float,
            "gain_map": res_a.gain_map,
            "report": report,
            "loaded": loaded,
        }

    elif mode_norm in ("pseudo-flatfield", "pseudo-flat-field", "pseudo", "mode-b", "b"):
        res_b = pseudo_flatfield_correct_mode_b(
            sample=loaded,
            sigma_px=sigma_px,
            deprecated_mag=deprecated_mag,
            bg_method=bg_method,
            per_channel=per_channel,
            target_val=target_val,
            min_bg_rel=min_bg_rel,
            max_gain=max_gain,
            border_mode=border_mode,
            modality=modality,
            force=force,
        )
        quant = quantize_and_clip(
            res_b.corrected_float,
            orig_dtype=loaded.orig_dtype,
            alpha=loaded.alpha,
            float_output=float_output,
            clip_warn_threshold=clip_warn_threshold,
        )
        extra_diag = {
            "foreground_excluded_fraction": round(res_b.foreground_excluded_fraction, 5),
            "low_bg_protected_fraction": round(res_b.low_bg_fraction, 6),
            "capped_gain_fraction": round(res_b.capped_gain_fraction, 6),
        }
        report = build_qc_report(
            loaded=loaded,
            quant=quant,
            gain_map=res_b.gain_map,
            mode_params=res_b.parameters,
            mode_warnings=res_b.warnings,
            protected_fraction=res_b.low_bg_fraction,
            capped_gain_fraction=res_b.capped_gain_fraction,
            extra_diagnostics=extra_diag,
            ground_truth_bg_mask=ground_truth_bg_mask,
        )
        return {
            "output_array": quant.array,
            "float_array": res_b.corrected_float,
            "bg_estimate": res_b.bg_estimate,
            "gain_map": res_b.gain_map,
            "report": report,
            "loaded": loaded,
        }

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'flatfield' (Mode A) or 'pseudo-flatfield' (Mode B).")


def process_single_image(
    input_path: str,
    output_path: str,
    mode: str = "pseudo-flatfield",
    flatfield: Optional[str] = None,
    darkfield: Optional[str] = None,
    norm_stat: str = "median",
    keep_offset: bool = False,
    denom_eps_rel: float = 0.01,
    allow_resize: bool = False,
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
    float_output: bool = False,
    clip_warn_threshold: float = 0.001,
    overwrite: bool = False,
    quality: int = 95,
    report_json_path: Optional[str] = None,
    print_summary: bool = False,
) -> Dict[str, Any]:
    """Process a single image file, write output + metadata, and return the QC report."""
    check_overwrite_safety(input_path, output_path, overwrite=overwrite)

    result = correct_image_pipeline(
        sample=input_path,
        mode=mode,
        flatfield=flatfield,
        darkfield=darkfield,
        norm_stat=norm_stat,
        keep_offset=keep_offset,
        denom_eps_rel=denom_eps_rel,
        allow_resize=allow_resize,
        sigma_px=sigma_px,
        deprecated_mag=deprecated_mag,
        bg_method=bg_method,
        per_channel=per_channel,
        target_val=target_val,
        min_bg_rel=min_bg_rel,
        max_gain=max_gain,
        border_mode=border_mode,
        modality=modality,
        force=force,
        float_output=float_output,
        clip_warn_threshold=clip_warn_threshold,
    )

    report = result["report"]
    report["output_path"] = output_path
    sidecar = write_image_and_metadata(
        output_path,
        result["output_array"],
        metadata=report,
        quality=quality,
    )
    if sidecar:
        report["sidecar_metadata_path"] = sidecar

    if report_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_json_path)), exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2, ensure_ascii=False)

    if print_summary:
        _print_qc_summary(report)

    return report


def process_batch(
    input_dir: str,
    output_dir: str,
    mode: str = "pseudo-flatfield",
    flatfield: Optional[str] = None,
    darkfield: Optional[str] = None,
    norm_stat: str = "median",
    keep_offset: bool = False,
    denom_eps_rel: float = 0.01,
    allow_resize: bool = False,
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
    float_output: bool = False,
    clip_warn_threshold: float = 0.001,
    overwrite: bool = False,
    extensions: Optional[Sequence[str]] = None,
    quality: int = 95,
    report_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Process all supported images in a directory and write a batch QC report JSON."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    check_dir_overwrite_safety(input_dir, output_dir, overwrite=overwrite)
    if deprecated_mag is not None and deprecated_mag.strip().lower() == "auto":
        raise ValueError(
            "'--mag auto' has been removed because inferring objective magnification from folder names is unreliable. "
            "Specify '--sigma-px <pixels>' explicitly (or omit '--mag' to use the resolution-scaled default)."
        )

    if extensions is None:
        extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    os.makedirs(output_dir, exist_ok=True)
    all_files: List[str] = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        all_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    all_files = sorted(list(set(all_files)))

    if not all_files:
        print(f"Warning: No images found in {input_dir}")
        return {"processed_count": 0, "total_files": 0, "files": []}

    file_reports: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for idx, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        out_fpath = os.path.join(output_dir, fname)
        try:
            rep = process_single_image(
                input_path=fpath,
                output_path=out_fpath,
                mode=mode,
                flatfield=flatfield,
                darkfield=darkfield,
                norm_stat=norm_stat,
                keep_offset=keep_offset,
                denom_eps_rel=denom_eps_rel,
                allow_resize=allow_resize,
                sigma_px=sigma_px,
                deprecated_mag=deprecated_mag,
                bg_method=bg_method,
                per_channel=per_channel,
                target_val=target_val,
                min_bg_rel=min_bg_rel,
                max_gain=max_gain,
                border_mode=border_mode,
                modality=modality,
                force=force,
                float_output=float_output,
                clip_warn_threshold=clip_warn_threshold,
                overwrite=overwrite,
                quality=quality,
            )
            file_reports.append(rep)
        except Exception as e:
            errors.append({"file": fname, "error": str(e)})
            print(f"  [ERROR] Failed to process {fname}: {e}", file=sys.stderr)

    batch_summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "mode": mode,
        "total_files": len(all_files),
        "processed_count": len(file_reports),
        "error_count": len(errors),
        "errors": errors,
        "files": file_reports,
    }

    out_report_path = report_json_path or os.path.join(output_dir, "batch_qc_report.json")
    with open(out_report_path, "w", encoding="utf-8") as bf:
        json.dump(batch_summary, bf, indent=2, ensure_ascii=False)

    return batch_summary


def _print_qc_summary(report: Dict[str, Any]) -> None:
    in_s = report["input_summary"]
    out_s = report["output_summary"]
    unif = report["background_uniformity_independent_grid"]
    g_s = report["gain_distribution"]
    print(f"=== QC Summary ({report.get('source_path', 'image')}) ===")
    print(f"  Mode              : {report['mode']}")
    print(f"  Input -> Output   : {in_s['dtype']} {in_s['shape']} -> {out_s['dtype']} {out_s['shape']}")
    print(f"  Intensity Range   : [{in_s['min']:.1f}, {in_s['max']:.1f}] (med {in_s['median']:.1f}) -> [{out_s['min']:.1f}, {out_s['max']:.1f}] (med {out_s['median']:.1f})")
    print(f"  Clipped Pixels    : {out_s['clipped_total_fraction']*100:.4f}% (low={out_s['clipped_low_count']}, high={out_s['clipped_high_count']})")
    print(f"  Grid Uniformity CV: {unif['before']['cv']:.4f} -> {unif['after']['cv']:.4f} (max/min: {unif['before']['max_min_ratio']:.3f} -> {unif['after']['max_min_ratio']:.3f})")
    print(f"  Gain [min,med,max]: [{g_s['min']:.3f}, {g_s['median']:.3f}, {g_s['max']:.3f}]")
    if report["warnings"]:
        print(f"  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    - [{w['code']}] {w['message']}")


def _add_common_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mode",
        default="pseudo-flatfield",
        choices=["flatfield", "pseudo-flatfield"],
        help="Correction mode: 'flatfield' (Mode A, requires --flatfield reference) or 'pseudo-flatfield' (Mode B, approximate low-frequency background estimation).",
    )
    # Mode A options
    p.add_argument("--flatfield", "-f", default=None, help="[Mode A] Path to flat-field reference image or directory of frames.")
    p.add_argument("--darkfield", "-d", default=None, help="[Mode A] Optional path to dark-field image or directory of frames.")
    p.add_argument("--norm-stat", default="median", choices=["median", "mean"], help="[Mode A] Normalization statistic for (F - D) (default: median).")
    p.add_argument("--keep-offset", action="store_true", help="[Mode A] Add back median(D) after flat-field division to preserve dark offset.")
    p.add_argument("--denom-eps-rel", type=float, default=0.01, help="[Mode A] Relative threshold of max(F - D) below which pixels are left uncorrected (default: 0.01).")
    p.add_argument("--allow-resize", action="store_true", help="[Mode A] Explicitly allow resizing reference F/D if dimensions differ from sample.")
    # Mode B options
    p.add_argument("--sigma-px", "--sigma", dest="sigma_px", type=float, default=None, help="[Mode B] Background smoothing scale in pixels (default: short_edge / 8).")
    p.add_argument("--mag", default=None, help="[DEPRECATED] Legacy objective magnification alias ('4x','10x','20x','40x','100x'). '--mag auto' is removed; use --sigma-px instead.")
    p.add_argument(
        "--bg-method",
        default="masked_gaussian",
        choices=["masked_gaussian", "gaussian"],
        help="[Mode B] Background estimation method: 'masked_gaussian' (default, foreground-resistant) or 'gaussian' (naive unmasked blur, emits warning).",
    )
    p.add_argument("--per-channel", action="store_true", help="[Mode B] Estimate background independently per RGB channel (emits warning: alters stain color ratios).")
    p.add_argument("--target-val", type=float, default=None, help="[Mode B] Explicit normalization target intensity (default: None, preserves native median background).")
    p.add_argument("--min-bg-rel", type=float, default=0.01, help="[Mode B] Relative background floor below which gain is locked to 1.0 (default: 0.01).")
    p.add_argument("--max-gain", type=float, default=4.0, help="[Mode B] Maximum allowed multiplicative gain ceiling (default: 4.0).")
    p.add_argument("--border-mode", default="reflect", choices=["reflect", "nearest", "mirror", "wrap"], help="[Mode B] Boundary padding mode (default: reflect).")
    # Shared options
    p.add_argument("--modality", default="brightfield", choices=["brightfield", "fluorescence"], help="Imaging modality ('brightfield' or 'fluorescence'). Fluorescence requires Mode A unless --force is passed.")
    p.add_argument("--force", action="store_true", help="Force Mode B execution on fluorescence / dark-background images (not recommended for quantification).")
    p.add_argument("--float-output", action="store_true", help="Save output as 32-bit floating-point TIFF (.tif/.tiff only).")
    p.add_argument("--clip-warn-threshold", type=float, default=0.001, help="Fraction of clipped pixels above which CLIPPED_PIXELS warning is emitted (default: 0.001 = 0.1%%).")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files or same directory.")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality if output is .jpg (default: 95).")
    p.add_argument("--report-json", default=None, help="Optional file path to save the standalone QC report JSON.")
    p.add_argument("--print-summary", action="store_true", help="Print human-readable QC summary to stdout.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Microscopy Illumination Correction CLI: Mode A (instrument flat-field calibration) "
            "& Mode B (pseudo-flat-field low-frequency background approximation)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_single = subparsers.add_parser("single", help="Correct illumination for a single image file")
    p_single.add_argument("--input-file", "-i", required=True, help="Path to input image")
    p_single.add_argument("--output-file", "-o", required=True, help="Path to output corrected image")
    _add_common_arguments(p_single)

    p_batch = subparsers.add_parser("batch", help="Batch correct illumination for all images in a directory")
    p_batch.add_argument("--input-dir", "-i", required=True, help="Directory containing raw microscopy images")
    p_batch.add_argument("--output-dir", "-o", required=True, help="Directory to save corrected images")
    _add_common_arguments(p_batch)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "single":
        process_single_image(
            input_path=args.input_file,
            output_path=args.output_file,
            mode=args.mode,
            flatfield=args.flatfield,
            darkfield=args.darkfield,
            norm_stat=args.norm_stat,
            keep_offset=args.keep_offset,
            denom_eps_rel=args.denom_eps_rel,
            allow_resize=args.allow_resize,
            sigma_px=args.sigma_px,
            deprecated_mag=args.mag,
            bg_method=args.bg_method,
            per_channel=args.per_channel,
            target_val=args.target_val,
            min_bg_rel=args.min_bg_rel,
            max_gain=args.max_gain,
            border_mode=args.border_mode,
            modality=args.modality,
            force=args.force,
            float_output=args.float_output,
            clip_warn_threshold=args.clip_warn_threshold,
            overwrite=args.overwrite,
            quality=args.quality,
            report_json_path=args.report_json,
            print_summary=args.print_summary,
        )
        return 0

    elif args.command == "batch":
        process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            flatfield=args.flatfield,
            darkfield=args.darkfield,
            norm_stat=args.norm_stat,
            keep_offset=args.keep_offset,
            denom_eps_rel=args.denom_eps_rel,
            allow_resize=args.allow_resize,
            sigma_px=args.sigma_px,
            deprecated_mag=args.mag,
            bg_method=args.bg_method,
            per_channel=args.per_channel,
            target_val=args.target_val,
            min_bg_rel=args.min_bg_rel,
            max_gain=args.max_gain,
            border_mode=args.border_mode,
            modality=args.modality,
            force=args.force,
            float_output=args.float_output,
            clip_warn_threshold=args.clip_warn_threshold,
            overwrite=args.overwrite,
            quality=args.quality,
            report_json_path=args.report_json,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
