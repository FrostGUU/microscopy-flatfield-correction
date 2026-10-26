# Changelog & Version History (`CHANGELOG.md`)

All notable changes, architectural refactorings, and quantitative improvements to `microscopy-flatfield-correction` are documented in this file.

---

## [`v2.0.0`] — 2026-10-26 (Scientific Quantitative Refactoring Release)

### Summary of Improvements (v1.1.1 $\rightarrow$ v2.0.0)

| Dimension | `v1.x` (Legacy Baseline) | `v2.0.0` (Refactored Scientific Release) | Quantitative Improvement |
| :--- | :--- | :--- | :--- |
| **16-Bit TIFF Bit-Depth** | Silently downconverted to 8-bit (`uint8`) BGR via `cv2.IMREAD_COLOR` | Preserves native `uint16` / `uint8` bit-depth via `tifffile` (plus optional `--float-output`) | Eliminates **95.36%** unique intensity level loss on 16-bit images; identity error $\le \pm 1\text{ LSB}$ |
| **Calibration Modes** | Single unmasked Gaussian pseudo-flat-field only | **Mode A (`flatfield`)** true instrument calibration ($F, D$) + **Mode B (`pseudo-flatfield`)** | Mode A achieves **0.007%–0.024%** target intensity error & **99.99%–100.00%** contrast retention |
| **Mode B Background Estimation** | Unmasked full-image Gaussian blur (`cv2.GaussianBlur`) | Detrended IRLS polynomial + foreground-masked normalized Gaussian (`masked_gaussian`) | Reduces $\cos^4$ target intensity error from **20.79%** to **0.14%**; reduces halo overshoot from **+13.84%** to **+0.20%** |
| **RGB Color Channel Ratios** | Per-channel independent normalization to `target_val = 220` | **Shared luminance gain** by default (`--per-channel` optional with warning) | Reduces foreground stained droplet `R/G` ratio error from **18.69%** to **0.09%** (background `R/B` error from **21.53%** to **0.01%**) |
| **Spatial Scale Specification** | Hardcoded lookup table by objective (`--mag 4x/10x/20x/40x` or folder name `auto`) | Pixel-scale `--sigma-px` (default `short_edge / 8`); `--mag auto` removed, `--mag` deprecated | Prevents scale mismatch across different camera pixel pitches, C-mount adapters, binning, and ROI crops |
| **Quantization & Safety** | Silent `[0, 255]` clipping + `.astype(uint8)` floor truncation; overwrites files in-place | Round-to-nearest (`np.rint`), out-of-range pixel counting & warning (`>0.1%`), no-overwrite default (`--overwrite` required) | Eliminates `-1 LSB` truncation bias and prevents accidental raw data destruction |
| **Quality Control (QC)** | No metrics or metadata output | Embedded TIFF `ImageDescription` JSON / sidecar `.json` + independent $8\times 8$ coarse-grid QC report | Provides independent pre/post CV, `max/min` ratio, gain percentiles, channel ratio shifts, and structured warning codes |

---

### Detailed Stage-by-Stage Breakdown in `v2.0.0`

#### Stage 0 — Baseline Audit & Characterization (`0cdd096` / `v2.0.0-stage0`)
- **Added**: [`reports/STAGE0_BASELINE_AUDIT.md`](reports/STAGE0_BASELINE_AUDIT.md), [`tests/test_stage0_baseline.py`](tests/test_stage0_baseline.py), [`reports/stage0_characterization_metrics.json`](reports/stage0_characterization_metrics.json).
- **What Changed**:
  - Audited `v1.1.1` code line-by-line and created 4 executable characterization tests demonstrating:
    1. 16-bit TIFF silent conversion to 8-bit BGR (`95.36%` of distinct intensity levels lost upon read).
    2. Silent highlight clipping at `255` and `-1 LSB` floor truncation bias.
    3. Foreground contrast loss and `+14.69%` bright halo ring overshoot around dark structures under unmasked Gaussian blur.
    4. `>20%` shift in RGB channel ratios (`R/B` and `R/G`) caused by per-channel normalization.

#### Stage 1 — Bit-Depth Preserving I/O, Validation & Metadata (`6a53ceb` / `v2.0.0-stage1`)
- **Added / Modified**: [`scripts/io_utils.py`](scripts/io_utils.py), [`requirements.txt`](requirements.txt).
- **What Changed**:
  - Replaced `cv2.IMREAD_COLOR` with `tifffile` for `.tif`/`.tiff` (with clear error if `imagecodecs` is required) and `cv2.IMREAD_UNCHANGED` for `.png`/`.jpg`.
  - Added strict shape and dtype validation: supports 2D grayscale `(H, W)`, 3D RGB `(H, W, 3)`, and 3D RGBA `(H, W, 4)` (preserving the Alpha channel unmodified with `ALPHA_CHANNEL_PRESERVED` warning) in `uint8` and `uint16`.
  - Explicitly rejects multi-page 3D TIFF stacks, channel-first `(C, H, W)` layouts, and floating-point inputs without heuristic axis guessing.
  - Detects 12-bit sensor data stored in 16-bit containers (`max <= 4095`, `CONTAINER_12BIT_IN_16BIT`) without stretching to `65535`.
  - Implemented `quantize_and_clip` using `float32` internal precision, `np.rint` round-to-nearest integer conversion, and explicit clipping counters (`CLIPPED_PIXELS` warning when clipped pixels exceed `--clip-warn-threshold`, default `0.1%`).
  - Added overwrite protection (`check_overwrite_safety`, `check_dir_overwrite_safety`), requiring `--overwrite` to overwrite existing files or directories.
  - Embedded JSON metadata into TIFF `ImageDescription` tags (7-bit ASCII escaped) and companion `<stem>.json` sidecar files for PNG/JPEG.
  - Removed unreliable `--mag auto` folder-name inference; retained `--mag` strictly as a deprecated alias (`DEPRECATED_MAG_ARG`).

#### Stage 2 — Mode A: True Instrument Flat-Field Calibration (`60cac01` / `v2.0.0-stage2`)
- **Added**: [`scripts/flatfield_mode_a.py`](scripts/flatfield_mode_a.py).
- **What Changed**:
  - Implemented physical flat-field calibration $I_{\mathrm{corr}} = \frac{\max(I_{\mathrm{raw}} - D,\ 0)}{F - D} \times \mathrm{norm}(F - D) + I_{\mathrm{offset}}$ supporting single-frame or multi-frame (pixel-wise median) flat-field $F$ and dark-field $D$.
  - Enforced paired per-channel calibration and spatial resolution matching (raising `ValueError` on mismatch unless `--allow-resize` is passed, which logs `RESIZED_REFERENCE`).
  - Added denominator protection where $F - D \le \tau_{\mathrm{denom}} \cdot \max(F - D)$ (default `1%`, `--denom-eps-rel 0.01`), leaving protected pixels uncorrected (`gain = 1.0`) and reporting spatial quadrant and bounding-box summaries (`INVALID_FLATFIELD_DENOMINATOR`).
  - Added reference quality auditing for sensor saturation (`REFERENCE_SATURATED`), localized high-frequency dust/bubble anomalies (`REFERENCE_HIGH_FREQUENCY_ARTIFACTS`), and bit-depth mismatch (`REFERENCE_BIT_DEPTH_MISMATCH`).
  - Added `--keep-offset` to optionally restore $\mathrm{median}(D)$ after flat-field division.

#### Stage 3 — Mode B: Foreground-Resistant Pseudo-Flat-Field (`8b63275` / `v2.0.0-stage3`)
- **Added**: [`scripts/pseudo_flatfield_mode_b.py`](scripts/pseudo_flatfield_mode_b.py).
- **What Changed**:
  - Introduced pixel-scale smoothing parameter `--sigma-px` (defaulting to $\max(15.0,\ \min(H, W)/8)$).
  - Implemented foreground-resistant background estimation (`--bg-method masked_gaussian`, default) combining coarse-block upper-percentile pre-filtering, asymmetric IRLS 2D polynomial trend fitting, and scale-consistent foreground-masked normalized Gaussian convolution with `reflect` boundary padding. Retained naive unmasked Gaussian as `--bg-method gaussian` with `NAIVE_GAUSSIAN_BG` warning.
  - Changed RGB color handling to use a **single shared luminance gain field** by default, preserving RGB channel ratios ($R/G, B/G, R/B$). Retained `--per-channel` as an explicit option with `PER_CHANNEL_COLOR_SHIFT` warning.
  - Changed default normalization target $T$ to $\mathrm{median}(\hat{B})$, preserving native intensity scales for both `uint8` and `uint16` images unless `--target-val` is explicitly provided.
  - Added low-background noise protection (`--min-bg-rel 0.01`, `LOW_BG_REGION`) and maximum gain ceiling (`--max-gain 4.0`, `GAIN_CAPPED`).
  - Rejected fluorescence / sparse-bright-object dark-background images in Mode B by default unless `--force` is specified (`FLUORESCENCE_OR_DARK_BG_FORCED`).

#### Stage 4 — Independent Quality Control Report & Unified CLI (`1dfab6c` / `v2.0.0-stage4`)
- **Added / Modified**: [`scripts/qc_report.py`](scripts/qc_report.py), [`scripts/correct_illumination.py`](scripts/correct_illumination.py).
- **What Changed**:
  - Implemented `build_qc_report` to output machine-readable JSON reports containing input/output bit-depth, shape, range, saturation, clipping counts, gain percentiles (`min`, `p1`, `median`, `p99`, `max`), RGB foreground/background channel ratio shifts, and structured warning codes.
  - Implemented independent background uniformity evaluation (`compute_independent_grid_uniformity`) using an $8\times 8$ coarse grid of background-pixel block medians identified via independent Otsu + local percentile thresholding (never reusing the correction algorithm's own background estimate).
  - Unified Mode A and Mode B under `scripts/correct_illumination.py` while preserving `single` and `batch` subcommands.

#### Stage 5 — Automated Pytest Suite & Synthetic Benchmarks (`d1e2b53` / `v2.0.0-stage5`)
- **Added**: [`tests/test_rigorous_suite.py`](tests/test_rigorous_suite.py), [`scripts/run_synthetic_benchmark.py`](scripts/run_synthetic_benchmark.py), [`reports/synthetic_benchmark_old_vs_new.json`](reports/synthetic_benchmark_old_vs_new.json), [`reports/SYNTHETIC_BENCHMARK_OLD_VS_NEW.md`](reports/SYNTHETIC_BENCHMARK_OLD_VS_NEW.md).
- **What Changed**:
  - Added 17 automated pytest tests covering uniform background invariance ($\le 0.5\%$), non-Gaussian $\cos^4$ and tilted-plane recovery (BG error $\le 1\%$, target error $\le 2\%$, contrast retention $\ge 95\%$), Mode A noiseless recovery ($\le 0.1\%$), negative controls (too-small $\sigma_{\mathrm{px}}$ and naive Gaussian halo), 16-bit TIFF preservation ($\le \pm 1\text{ LSB}$), RGB channel ratio retention ($\le 1\%$), edge cases, odd dimensions, RGBA Alpha preservation, Unicode paths, overwrite safety, and documentation terminology linting.

#### Stage 6 — Documentation Revision & Reproducible Visual Benchmarks (`76036f2` / `v2.0.0-stage6`)
- **Added / Modified**: [`scripts/generate_benchmark_figures.py`](scripts/generate_benchmark_figures.py), [`reports/real_samples_qc_summary.json`](reports/real_samples_qc_summary.json), [`README.md`](README.md), [`SKILL.md`](SKILL.md), [`assets/flatfield_comparison_overview.png`](assets/flatfield_comparison_overview.png), [`assets/examples/`](assets/examples/).
- **What Changed**:
  - Rewrote `README.md` and `SKILL.md` to clearly separate Mode A vs. Mode B, define quantitative applicability boundaries, document limitations, and eliminate unverified absolute claims.
  - Regenerated all example images, benchmark figures, and `README.md` QC metrics automatically via `scripts/generate_benchmark_figures.py`.

---

## [`v1.1.1`] — 2026-10-26 (`db715dd`)
- **Fixed**: Escaped LaTeX subscripts and set braces in `README.md` and `SKILL.md` so GitHub MathJax renders formulas without `'_' allowed only in math mode` errors.

## [`v1.1.0`] — 2026-10-26 (`de1bfc6`)
- **Added**: Initial 4x and 20x brightfield before/after comparison overview figure and 3 anonymized sample image pairs under `assets/`.

## [`v1.0.0`] — 2026-10-26 (`b1a11ce`)
- **Added**: Initial release of `microscopy-flatfield-correction` skill (`SKILL.md`, `README.md`, `requirements.txt`, `.gitignore`, and 8-bit per-channel Gaussian `scripts/correct_illumination.py`).
