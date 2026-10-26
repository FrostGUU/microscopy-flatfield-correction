"""
Stage 5: Comprehensive Automated Pytest Suite (`test_rigorous_suite.py`)
------------------------------------------------------------------------
Covers all 11 required verification categories with pre-set thresholds and fixed random seeds.
"""

import os
import re
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

from io_utils import read_image, inspect_loaded_array
from correct_illumination import (
    correct_image_pipeline,
    process_single_image,
    process_batch,
    main as cli_main,
)
from run_synthetic_benchmark import run_all_benchmarks

# =============================================================================
# Pre-set Quantitative Thresholds (Fixed before test execution — DO NOT LOOSEN)
# =============================================================================
THRESH_UNIFORM_MAX_REL_ERR = 0.005          # <= 0.5%
THRESH_SYNTH_BG_MED_REL_ERR_PCT = 1.0       # <= 1.0%
THRESH_SYNTH_TARGET_MEAN_ERR_PCT = 2.0      # <= 2.0%
THRESH_SYNTH_CONTRAST_RETENTION_PCT = 95.0  # >= 95.0%
THRESH_MODE_A_NOISELESS_REL_ERR = 0.001     # <= 0.1%
THRESH_IDENTITY_16BIT_MAX_LSB = 1           # <= +/- 1 LSB
THRESH_RGB_BG_RATIO_SHIFT_PCT = 1.0         # <= 1.0%


def _warning_codes(report: dict) -> set:
    return {w["code"] for w in report.get("warnings", [])}


# =============================================================================
# 1. Uniform Background Test (<= 0.5% max relative deviation, noiseless)
# =============================================================================

@pytest.mark.parametrize("dtype,val", [(np.uint8, 180), (np.uint16, 36000)])
@pytest.mark.parametrize("is_rgb", [False, True])
def test_01_uniform_background_mode_a_and_b(dtype, val, is_rgb):
    shape = (128, 128, 3) if is_rgb else (128, 128)
    img = np.full(shape, val, dtype=dtype)
    flat = np.full(shape, int(val * 1.1), dtype=dtype)

    res_b = correct_image_pipeline(img, mode="pseudo-flatfield", sigma_px=24.0)
    out_b = res_b["output_array"].astype(np.float64)
    rel_err_b = np.max(np.abs(out_b - float(val)) / float(val))
    assert rel_err_b <= THRESH_UNIFORM_MAX_REL_ERR

    res_a = correct_image_pipeline(img, mode="flatfield", flatfield=flat)
    out_a = res_a["output_array"].astype(np.float64)
    rel_err_a = np.max(np.abs(out_a - float(val)) / float(val))
    assert rel_err_a <= THRESH_UNIFORM_MAX_REL_ERR


# =============================================================================
# 2. Known Non-Gaussian Illumination Synthetic Benchmark (cos^4 & tilted plane)
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_benchmark_data():
    return run_all_benchmarks()


@pytest.mark.parametrize("scenario", ["cos4_vignetting", "tilted_plane"])
def test_02_synthetic_non_gaussian_illumination_accuracy(synthetic_benchmark_data, scenario):
    metrics = synthetic_benchmark_data["scenarios"][scenario]["new_mode_b_masked_gaussian_16bit"]
    assert metrics["bg_median_rel_error_pct"] <= THRESH_SYNTH_BG_MED_REL_ERR_PCT
    assert metrics["target_mean_rel_error_pct"] <= THRESH_SYNTH_TARGET_MEAN_ERR_PCT
    assert metrics["contrast_retention_pct"] >= THRESH_SYNTH_CONTRAST_RETENTION_PCT


# =============================================================================
# 3. Mode A True Flat-Field Recovery (<= 0.1% noiseless error)
# =============================================================================

def test_03_mode_a_true_flatfield_noiseless_recovery():
    h, w = 160, 160
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h / 2.0) / (h / 2.0)) ** 2 + ((xx - w / 2.0) / (w / 2.0)) ** 2
    illum = 1.0 / ((1.0 + 0.35 * r2) ** 2)

    dark_val = 800.0
    flat_gain = 40000.0 * illum
    flat_u16 = np.rint(dark_val + flat_gain).astype(np.uint16)
    dark_u16 = np.full((h, w), int(dark_val), dtype=np.uint16)

    # True object signal (in float before illumination modulation)
    true_signal = np.full((h, w), 30000.0, dtype=np.float64)
    true_signal[40:110, 50:120] = 12000.0
    raw_u16 = np.rint(dark_val + true_signal * (flat_gain / np.median(flat_gain))).astype(np.uint16)

    res = correct_image_pipeline(raw_u16, mode="flatfield", flatfield=flat_u16, darkfield=dark_u16)
    recovered = res["output_array"].astype(np.float64)
    rel_err = np.median(np.abs(recovered - true_signal) / true_signal)
    max_rel_err = np.max(np.abs(recovered - true_signal) / true_signal)
    assert rel_err <= THRESH_MODE_A_NOISELESS_REL_ERR
    assert max_rel_err <= THRESH_MODE_A_NOISELESS_REL_ERR

    # Also verify --keep-offset adds back median(D) = 800
    res_offset = correct_image_pipeline(
        raw_u16, mode="flatfield", flatfield=flat_u16, darkfield=dark_u16, keep_offset=True
    )
    diff_offset = np.median(res_offset["output_array"].astype(np.float64) - recovered)
    assert abs(diff_offset - dark_val) <= 1.0


# =============================================================================
# 4. Negative Controls: Too-Small Sigma, Naive Gaussian Halo, Legacy Comparison
# =============================================================================

def test_04_negative_controls_small_sigma_and_naive_gaussian_halo(synthetic_benchmark_data):
    sc = synthetic_benchmark_data["scenarios"]["cos4_vignetting"]
    masked = sc["new_mode_b_masked_gaussian_16bit"]
    small_sigma = sc["negative_control_small_sigma_10px"]
    naive_gauss = sc["new_mode_b_naive_gaussian_16bit"]
    legacy = sc["legacy_8bit_per_channel_gaussian"]

    # 1. Too-small sigma (10px vs 38px disc) must severely attenuate contrast (< 65%) and induce halo
    assert small_sigma["contrast_retention_pct"] < 65.0
    assert small_sigma["target_mean_rel_error_pct"] > 40.0
    assert small_sigma["halo_overshoot_pct"] > 3.0

    # 2. Naive unmasked Gaussian must exhibit halo overshoot (> 10%) and worse target error than masked_gaussian
    assert naive_gauss["halo_overshoot_pct"] > 10.0
    assert naive_gauss["target_mean_rel_error_pct"] > 10.0
    assert masked["halo_overshoot_pct"] < 1.0

    # 3. Legacy 8-bit implementation loses 16-bit depth and exhibits large target/halo error
    assert legacy["output_dtype"] == "uint8"
    assert legacy["target_mean_rel_error_pct"] > 15.0
    assert legacy["halo_overshoot_pct"] > 10.0


# =============================================================================
# 5. 16-Bit TIFF Preservation, Identity <= 1 LSB, Histogram Bins & 12-Bit Check
# =============================================================================

def test_05_16bit_tiff_preservation_and_quantization():
    rng = np.random.default_rng(99)
    h, w = 128, 128
    # Create 16-bit image with > 2000 distinct intensity levels
    base_u16 = np.linspace(10000, 45000, h * w, dtype=np.uint16).reshape(h, w)
    flat_uniform = np.full((h, w), 30000, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_tif = os.path.join(tmpdir, "in_16bit.tif")
        out_tif = os.path.join(tmpdir, "out_16bit.tif")
        flat_tif = os.path.join(tmpdir, "flat_16bit.tif")
        tifffile.imwrite(in_tif, base_u16)
        tifffile.imwrite(flat_tif, flat_uniform)

        report = process_single_image(
            input_path=in_tif,
            output_path=out_tif,
            mode="flatfield",
            flatfield=flat_tif,
        )
        saved = tifffile.imread(out_tif)

        assert saved.dtype == np.dtype("uint16")
        assert report["output_summary"]["dtype"] == "uint16"
        max_lsb_diff = int(np.max(np.abs(saved.astype(np.int32) - base_u16.astype(np.int32))))
        assert max_lsb_diff <= THRESH_IDENTITY_16BIT_MAX_LSB

        # Verify no quantization staircasing (occupied histogram bins preserved)
        unique_in = len(np.unique(base_u16))
        unique_out = len(np.unique(saved))
        assert unique_out >= int(0.98 * unique_in)

        # Verify embedded TIFF ImageDescription JSON metadata
        with tifffile.TiffFile(out_tif) as tif:
            desc = tif.pages[0].description
            meta = json.loads(desc)
            assert meta["mode"] == "flatfield"
            assert meta["input_summary"]["bit_depth"] == 16

    # Check 12-bit data in 16-bit container detection (max <= 4095, not stretched to 65535)
    img_12bit = rng.integers(400, 3800, size=(64, 64), dtype=np.uint16)
    res_12 = correct_image_pipeline(img_12bit, mode="pseudo-flatfield", sigma_px=16.0)
    assert "CONTAINER_12BIT_IN_16BIT" in _warning_codes(res_12["report"])
    assert res_12["output_array"].max() <= 4500  # Not stretched to 65535


# =============================================================================
# 6. RGB Channel Order & Ratios (Shared Gain <= 1% shift vs --per-channel shift)
# =============================================================================

def test_06_rgb_channel_order_and_ratios(synthetic_benchmark_data):
    rgb_bench = synthetic_benchmark_data["rgb_color_ratio_benchmark"]
    shared = rgb_bench["new_mode_b_shared_luminance_default"]
    per_ch = rgb_bench["new_mode_b_per_channel_flag"]

    assert shared["bg_R_over_G_error_pct"] <= THRESH_RGB_BG_RATIO_SHIFT_PCT
    assert shared["bg_R_over_B_error_pct"] <= THRESH_RGB_BG_RATIO_SHIFT_PCT
    assert shared["fg_droplet_R_over_G_error_pct"] <= THRESH_RGB_BG_RATIO_SHIFT_PCT

    # --per-channel must produce a measurable color ratio shift and emit warning
    assert per_ch["bg_R_over_B_error_pct"] > 10.0

    # Verify warning code and channel order (R > G > B preserved across TIFF and PNG I/O)
    img_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = 220  # R
    img_rgb[:, :, 1] = 170  # G
    img_rgb[:, :, 2] = 130  # B
    res_per_ch = correct_image_pipeline(img_rgb, mode="pseudo-flatfield", per_channel=True)
    assert "PER_CHANNEL_COLOR_SHIFT" in _warning_codes(res_per_ch["report"])

    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in [".tif", ".png"]:
            in_p = os.path.join(tmpdir, f"color{ext}")
            out_p = os.path.join(tmpdir, f"color_out{ext}")
            if ext == ".tif":
                tifffile.imwrite(in_p, img_rgb)
            else:
                cv2.imwrite(in_p, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            process_single_image(in_p, out_p, mode="pseudo-flatfield")
            reloaded = read_image(out_p).data
            assert int(reloaded[32, 32, 0]) == 220
            assert int(reloaded[32, 32, 1]) == 170
            assert int(reloaded[32, 32, 2]) == 130


# =============================================================================
# 7. Edge Conditions: Zero BG, Near-Zero BG, Saturation, Dust/Hot-Pixels, NaN/Inf
# =============================================================================

def test_07_edge_conditions_zero_bg_saturation_outliers_nan():
    # 1. All-zero and near-zero background in Mode B -> no NaN/Inf, LOW_BG_REGION emitted
    zero_img = np.zeros((64, 64), dtype=np.uint16)
    res_zero = correct_image_pipeline(zero_img, mode="pseudo-flatfield", force=True)
    assert np.all(np.isfinite(res_zero["output_array"]))
    assert "LOW_BG_REGION" in _warning_codes(res_zero["report"])

    # Partial zero region (obscured corner)
    part_zero = np.full((64, 64), 200, dtype=np.uint8)
    part_zero[:16, :16] = 0
    res_part = correct_image_pipeline(part_zero, mode="pseudo-flatfield", sigma_px=4.0)
    assert np.all(np.isfinite(res_part["output_array"]))
    assert "LOW_BG_REGION" in _warning_codes(res_part["report"])

    # 2. Mode A with zero / dead region in flat-field F -> INVALID_FLATFIELD_DENOMINATOR
    sample = np.full((64, 64), 1000, dtype=np.uint16)
    flat_dead = np.full((64, 64), 2000, dtype=np.uint16)
    flat_dead[0:10, 0:10] = 5  # <= 1% of 2000
    res_dead = correct_image_pipeline(sample, mode="flatfield", flatfield=flat_dead)
    assert "INVALID_FLATFIELD_DENOMINATOR" in _warning_codes(res_dead["report"])
    assert res_dead["report"]["diagnostics"]["invalid_denominator_summary"]["invalid_pixel_count"] == 100

    # 3. Saturated flat-field reference F & dust mote high-frequency artifact
    flat_sat_dust = np.full((128, 128), 50000, dtype=np.uint16)
    flat_sat_dust[:20, :20] = 65535  # Saturated block
    flat_sat_dust[60:65, 60:65] = 8000  # Sharp dark dust speck
    sample_128 = np.full((128, 128), 30000, dtype=np.uint16)
    res_ref_warn = correct_image_pipeline(sample_128, mode="flatfield", flatfield=flat_sat_dust)
    codes = _warning_codes(res_ref_warn["report"])
    assert "REFERENCE_SATURATED" in codes
    assert "REFERENCE_HIGH_FREQUENCY_ARTIFACTS" in codes

    # 4. Hot-pixel outlier resilience in Mode B
    clean_bg = np.full((128, 128), 200, dtype=np.uint8)
    noisy_hot = clean_bg.copy()
    noisy_hot[64, 64] = 255
    noisy_hot[30, 90] = 255
    res_hot = correct_image_pipeline(noisy_hot, mode="pseudo-flatfield", sigma_px=20.0)
    assert abs(float(np.median(res_hot["output_array"])) - 200.0) <= 1.0

    # 5. NaN / Inf rejection
    bad_float = np.full((32, 32), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported data type"):
        correct_image_pipeline(bad_float)


# =============================================================================
# 8. Diverse Image Dimensions (Odd/Non-Square), Mode A Resize & Boundary Mode
# =============================================================================

@pytest.mark.parametrize("shape", [(63, 127), (151, 89, 3), (75, 75)])
def test_08_image_dimensions_and_mode_a_resize(shape):
    img = np.full(shape, 180, dtype=np.uint8)
    res = correct_image_pipeline(img, mode="pseudo-flatfield")
    assert res["output_array"].shape == shape

    # Mode A mismatched dimensions must fail by default and succeed with allow_resize=True
    wrong_flat_shape = (shape[0] + 12, shape[1] + 8) if len(shape) == 2 else (shape[0] + 12, shape[1] + 8, 3)
    wrong_flat = np.full(wrong_flat_shape, 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="Spatial resolution mismatch"):
        correct_image_pipeline(img, mode="flatfield", flatfield=wrong_flat, allow_resize=False)

    res_resized = correct_image_pipeline(img, mode="flatfield", flatfield=wrong_flat, allow_resize=True)
    assert res_resized["output_array"].shape == shape
    assert "RESIZED_REFERENCE" in _warning_codes(res_resized["report"])


# =============================================================================
# 9. Input/Output Consistency, RGBA Alpha Preservation & Fluorescence Rejection
# =============================================================================

def test_09_input_output_consistency_rgba_and_fluorescence_rejection():
    # 1. RGBA (H, W, 4) preserves Alpha channel untouched and emits ALPHA_CHANNEL_PRESERVED
    rgba = np.zeros((64, 64, 4), dtype=np.uint16)
    rgba[:, :, 0] = 30000
    rgba[:, :, 1] = 25000
    rgba[:, :, 2] = 20000
    alpha_pattern = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    rgba[:, :, 3] = alpha_pattern

    res_rgba = correct_image_pipeline(rgba, mode="pseudo-flatfield", sigma_px=16.0)
    out_rgba = res_rgba["output_array"]
    assert out_rgba.shape == (64, 64, 4)
    assert out_rgba.dtype == np.dtype("uint16")
    assert np.array_equal(out_rgba[:, :, 3], alpha_pattern)
    assert "ALPHA_CHANNEL_PRESERVED" in _warning_codes(res_rgba["report"])

    # 2. Fluorescence / dark-background rejection in Mode B unless --force is passed
    dark_fluo = np.full((128, 128), 800, dtype=np.uint16)
    dark_fluo[55:65, 55:65] = 45000  # Sparse bright fluorescent cells on dark background
    with pytest.raises(ValueError, match="Mode B .* rejected"):
        correct_image_pipeline(dark_fluo, mode="pseudo-flatfield")

    with pytest.raises(ValueError, match="Mode B .* rejected"):
        correct_image_pipeline(rgba[:, :, :3], mode="pseudo-flatfield", modality="fluorescence")

    res_forced = correct_image_pipeline(dark_fluo, mode="pseudo-flatfield", force=True)
    assert "FLUORESCENCE_OR_DARK_BG_FORCED" in _warning_codes(res_forced["report"])


# =============================================================================
# 10. Regression Tests: Chinese Paths, Overwrite Safety, Invalid TIFFs, --mag
# =============================================================================

def test_10_regression_unicode_no_overwrite_unsupported_tiff_deprecated_mag():
    with tempfile.TemporaryDirectory() as tmpdir:
        cn_dir = os.path.join(tmpdir, "中文实验目录_油红O_测试")
        os.makedirs(cn_dir, exist_ok=True)
        in_tif = os.path.join(cn_dir, "样品_01_原始图.tif")
        out_tif = os.path.join(cn_dir, "样品_01_校正后.tif")
        out_png = os.path.join(cn_dir, "样品_01_校正后.png")

        img = np.full((64, 64, 3), 190, dtype=np.uint8)
        tifffile.imwrite(in_tif, img)

        # 1. Chinese path read/write + sidecar JSON for PNG
        process_single_image(in_tif, out_tif, mode="pseudo-flatfield")
        process_single_image(in_tif, out_png, mode="pseudo-flatfield")
        assert os.path.exists(out_tif)
        assert os.path.exists(out_png)
        assert os.path.exists(os.path.splitext(out_png)[0] + ".json")

        # 2. Overwrite protection (same file or existing output file rejected without --overwrite)
        with pytest.raises(ValueError, match="Refusing to overwrite input file"):
            process_single_image(in_tif, in_tif, overwrite=False)
        with pytest.raises(FileExistsError, match="Output file already exists"):
            process_single_image(in_tif, out_tif, overwrite=False)
        with pytest.raises(ValueError, match="Refusing to use the same directory"):
            process_batch(cn_dir, cn_dir, overwrite=False)

        # Explicitly passing overwrite=True succeeds
        process_single_image(in_tif, out_tif, overwrite=True)

        # 3. Unsupported TIFFs: multi-page 3D Z-stack, channel-first (3, H, W), float32
        multipage_tif = os.path.join(cn_dir, "multipage.tif")
        tifffile.imwrite(multipage_tif, np.zeros((4, 32, 32), dtype=np.uint8))
        with pytest.raises(ValueError, match="Multi-page|Unsupported"):
            read_image(multipage_tif)

        ch_first_tif = os.path.join(cn_dir, "channel_first.tif")
        tifffile.imwrite(ch_first_tif, np.zeros((3, 64, 64), dtype=np.uint8), photometric="minisblack")
        with pytest.raises(ValueError, match="Multi-page|channel-first|Unsupported"):
            read_image(ch_first_tif)

        # Also test in-memory channel-first (3, 64, 64)
        with pytest.raises(ValueError, match="channel-first"):
            inspect_loaded_array(np.zeros((3, 64, 64), dtype=np.uint8))

        float_tif = os.path.join(cn_dir, "float_in.tif")
        tifffile.imwrite(float_tif, np.ones((32, 32), dtype=np.float32))
        with pytest.raises(ValueError, match="Unsupported data type"):
            read_image(float_tif)

        # 4. Deprecated --mag alias works with warning; --mag auto raises ValueError
        out_mag = os.path.join(cn_dir, "out_mag4x.tif")
        rc = cli_main(["single", "-i", in_tif, "-o", out_mag, "--mag", "4x"])
        assert rc == 0
        with tifffile.TiffFile(out_mag) as tif:
            meta = json.loads(tif.pages[0].description)
            assert any(w["code"] == "DEPRECATED_MAG_ARG" for w in meta["warnings"])
            assert meta["parameters"]["sigma_px"] == 75.0

        with pytest.raises(ValueError, match="--mag auto' has been removed"):
            process_single_image(in_tif, out_mag, deprecated_mag="auto", overwrite=True)


# =============================================================================
# 11. Documentation Claims Lint Test (`README.md` & `SKILL.md`)
# =============================================================================

def test_11_documentation_claims():
    """
    Scan README.md and SKILL.md to ensure:
      1. Prohibited absolute guarantee terms ('保留所有结构', '定量准确', '无光晕',
         'without halo artifacts', 'preserves true staining chromaticity') do not appear
         as positive claims.
      2. Every occurrence of '无损' or 'lossless' appears ONLY in explicitly qualified contexts
         such as '文件编码无损', '无损压缩/编码', 'lossless encoding', 'lossless file', 'lossless compression',
         or in explicit disclaimers explaining that lossless encoding != unaltered pixel values.
    """
    allowed_lossless_context = re.compile(
        r"(lossless\s+(file|compression|encoding|container|format|tiff|png)|"
        r"(file|compression|encoding)\s+is\s+lossless|"
        r"文件编码无损|无损(文件|编码|压缩|格式|容器)|"
        r"无损\s*[≠!=]|不等于无损|并非无损|不是无损)",
        re.IGNORECASE,
    )
    prohibited_positive_claims = [
        "保留所有结构",
        "定量准确",
        "without halo artifacts",
        "preserves true staining chromaticity",
        "without distortion",
    ]

    for doc_name in ("README.md", "SKILL.md"):
        doc_path = os.path.join(REPO_ROOT, doc_name)
        with open(doc_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            for bad_phrase in prohibited_positive_claims:
                if bad_phrase.lower() in line.lower():
                    # Allow only if explicitly negated on the same line (e.g., '不得声称...保留所有结构' or '不能保证无光晕')
                    if not re.search(r"(不|非|不能|不得|无法|not|never|cannot|do not|does not)", line, re.IGNORECASE):
                        pytest.fail(
                            f"{doc_name}:{line_no} contains prohibited unverified claim '{bad_phrase}': {line}"
                        )

            if "无光晕" in line:
                if not re.search(r"(不|非|不能|不得|无法|并非)", line):
                    pytest.fail(f"{doc_name}:{line_no} contains prohibited claim '无光晕': {line}")

            if re.search(r"(lossless|无损)", line, re.IGNORECASE):
                if not allowed_lossless_context.search(line):
                    pytest.fail(
                        f"{doc_name}:{line_no} uses 'lossless'/'无损' outside explicitly qualified file-encoding context: {line}"
                    )
