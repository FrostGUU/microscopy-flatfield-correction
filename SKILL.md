---
name: microscopy-flatfield-correction
description: >-
  Illumination correction for microscopy images supporting two distinct modes:
  Mode A (flatfield, instrument calibration using measured flat-field F and dark-field D)
  and Mode B (pseudo-flatfield, approximate foreground-masked low-frequency background
  estimation for brightfield visual uniformity). Preserves input bit depth (uint8/uint16)
  and outputs machine-readable quality control (QC) reports.
---

# Microscopy Flat-Field & Pseudo-Flat-Field Illumination Correction

## Overview & Core Principles

Microscopy images frequently exhibit spatial illumination non-uniformity caused by optical vignetting, uneven light source projection, or shading gradients.

> [!IMPORTANT]
> **Lossless File Encoding != Unaltered Pixel Values (`文件编码无损 ≠ 图像像素内容未改变`)**
> Saving an output image in a format with **lossless file encoding** (such as uncompressed TIFF or lossless compression PNG) only means no additional compression artifacts are introduced when writing to disk. Illumination correction itself mathematically modifies pixel intensities.

This skill provides two strictly separated modes via `scripts/correct_illumination.py`:

1. **Mode A (`--mode flatfield`) — True Instrument Flat-Field Calibration**:
   - Uses an experimentally measured flat-field reference image $F$ (and optional dark-field image $D$) acquired under identical optical and exposure settings.
   - Applicable to both brightfield and fluorescence microscopy when linear sensor response and matching calibration frames are available.
2. **Mode B (`--mode pseudo-flatfield`, default) — Pseudo-Flat-Field Approximation**:
   - Estimates a smooth low-frequency background surface $\hat{B}(x, y)$ directly from a brightfield sample image when no physical reference $F$ is available.
   - Intended for **brightfield visual presentation and background leveling**; it is **not** an instrument flat-field calibration and **must not** be used for fluorescence microscopy or unvalidated quantitative intensity claims.

---

## Suitability for Quantitative Analysis vs. Visual Presentation

| Application Scenario | Mode A (`flatfield` with $F, D$) | Mode B (`pseudo-flatfield` without $F$) | Technical Rationale |
| :--- | :---: | :---: | :--- |
| **Visual presentation & figure panels (brightfield)** | Suitable | Suitable | Levels large-scale vignetting and field shading for visual inspection. |
| **Fluorescence intensity quantification** | Suitable (with linear $F, D$) | **Prohibited** (rejected by default) | Dividing dark-background fluorescence images by smoothed self-background amplifies noise and distorts emitter flux. |
| **Color threshold segmentation & positive area % (Oil Red O, DAB-IHC, H&E)** | Caution required | **Not recommended for direct quantification** | Scaling intensities alters RGB values and optical density thresholds; `--per-channel` additionally shifts RGB channel ratios. |
| **Staining intensity / Optical Density (OD) comparison** | Suitable (if calibrated in linear space) | **Do not use without validation** | Mode B estimates background from the sample itself; dense staining can leak into $\hat{B}$ and alter local contrast. |

---

## Mathematical Formulation

### Mode A: Instrument Flat-Field Calibration (`--mode flatfield`)

Given raw sample image $I_{\mathrm{raw}}$, flat-field reference $F$, and optional dark-field reference $D$ (if multiple frames are supplied, the pixel-wise median is used):

$$
I_{\mathrm{corr}}(x, y) = \frac{\max\bigl(I_{\mathrm{raw}}(x, y) - D(x, y),\ 0\bigr)}{F(x, y) - D(x, y)} \times \mathrm{norm}(F - D) + I_{\mathrm{offset}}
$$

- **Per-channel calibration**: Computed per channel using paired reference channels.
- **Normalization factor** $\mathrm{norm}(F - D)$: Defaults to $\mathrm{median}(F - D)$ over valid pixels (or `mean` via `--norm-stat mean`).
- **Dark-field zero offset** $I_{\mathrm{offset}}$: Defaults to $0$ (dark offset subtracted); pass `--keep-offset` to add back $\mathrm{median}(D)$.
- **Denominator protection**: Pixels where $F(x, y) - D(x, y) \le \tau_{\mathrm{denom}} \cdot \max(F - D)$ (default $\tau_{\mathrm{denom}} = 1\%$) are left uncorrected ($\mathrm{gain} = 1.0$) and reported in the QC JSON under `invalid_denominator_summary`.

### Mode B: Pseudo-Flat-Field Approximation (`--mode pseudo-flatfield`)

Estimates a low-frequency illumination background $\hat{B}(x, y)$ at spatial scale $\sigma_{\mathrm{px}}$ (in pixels):

$$
g(x, y) = \min\left(\frac{T}{\hat{B}(x, y)},\ g_{\max}\right), \qquad I_{\mathrm{corr}}(x, y) = I_{\mathrm{raw}}(x, y) \times g(x, y)
$$

- **Foreground-resistant background estimation (`--bg-method masked_gaussian`, default)**: Fits a robust asymmetric IRLS polynomial trend and performs foreground-masked normalized Gaussian convolution at scale `--sigma-px` with `reflect` boundary padding, reducing foreground absorption bias compared to naive Gaussian blurring (`--bg-method gaussian`).
- **Shared luminance gain for RGB images (default)**: Estimates $\hat{B}(x, y)$ from mean luminance $L = \frac{1}{3}(R+G+B)$ and multiplies all three channels by the same scalar gain $g(x, y)$ to preserve RGB channel ratios ($R:G:B$). Note: shared luminance gain does **not** remove background color cast. Passing `--per-channel` normalizes each channel independently (emitting `PER_CHANNEL_COLOR_SHIFT` warning).
- **Scale preservation**: Target intensity $T$ defaults to $\mathrm{median}(\hat{B})$, preserving the native numeric scale for both `uint8` and `uint16` images unless `--target-val` is explicitly specified.
- **Low-background & gain protection**: Regions where $\hat{B}(x, y) \le \mathrm{min\_bg\_rel} \cdot \max(\hat{B})$ (default $1\%$) are locked to $g(x, y) = 1.0$ (`LOW_BG_REGION`), and gains above `--max-gain` (default $4.0$) are capped (`GAIN_CAPPED`).

---

## CLI Usage

### 1. Mode A: Instrument Flat-Field Correction

```bash
python scripts/correct_illumination.py single \
  --mode flatfield \
  --input-file "path/to/sample_16bit.tif" \
  --flatfield "path/to/flat_ref_16bit.tif" \
  --darkfield "path/to/dark_ref_16bit.tif" \
  --output-file "path/to/calibrated_16bit.tif" \
  --report-json "path/to/qc_report.json" \
  --print-summary
```

### 2. Mode B: Pseudo-Flat-Field Correction (Single & Batch)

```bash
# Single image with explicit pixel scale --sigma-px
python scripts/correct_illumination.py single \
  --mode pseudo-flatfield \
  --input-file "path/to/brightfield.tif" \
  --output-file "path/to/pseudo_corrected.tif" \
  --sigma-px 75 \
  --report-json "path/to/qc_report.json"

# Batch directory processing
python scripts/correct_illumination.py batch \
  --mode pseudo-flatfield \
  --input-dir "path/to/raw_dir" \
  --output-dir "path/to/corrected_dir" \
  --sigma-px 75
```

### Key Options

- `--mode {flatfield,pseudo-flatfield}`: Select Mode A or Mode B (default: `pseudo-flatfield`).
- `--sigma-px <float>`: [Mode B] Smoothing scale in pixels. Defaults to `short_edge / 8`. Must be chosen larger than biological structures of interest and verified visually and via QC reports.
- `--bg-method {masked_gaussian,gaussian}`: [Mode B] Foreground-masked estimation (default) or naive unmasked Gaussian (emits `NAIVE_GAUSSIAN_BG` warning).
- `--per-channel`: [Mode B] Independent RGB channel normalization (emits `PER_CHANNEL_COLOR_SHIFT` warning).
- `--target-val <float>`: [Mode B] Override target baseline $T$ (default: `median(bg)`).
- `--max-gain <float>`: [Mode B] Maximum gain cap (default: `4.0`).
- `--min-bg-rel <float>`: [Mode B] Relative background floor (default: `0.01`).
- `--overwrite`: Explicitly allow overwriting existing output files.
- `--mag`: **Deprecated** legacy alias (`4x`->75px, `10x`->60px, `20x`->45px, `40x`->40px). `--mag auto` has been removed.

---

## Limitations (`局限性`)

1. **Gamma / Nonlinearity in 8-Bit Images**: Standard 8-bit PNG/JPEG and color camera outputs are typically gamma-encoded (sRGB), whereas the multiplicative illumination model assumes linear photon response. Processing 8-bit inputs emits `GAMMA_ENCODED_8BIT`.
2. **Scale Ambiguity in Mode B**: If biological structures (e.g., confluent cell sheets, large tissue sections, or dense lipid colonies) span a spatial scale comparable to or larger than `--sigma-px`, Mode B cannot separate foreground absorption from illumination shading, resulting in contrast attenuation and boundary halos.
3. **Shared Gain vs. Background Color Cast**: In Mode B, default shared luminance gain preserves RGB channel ratios ($R/G, R/B$) but does not remove lamp color temperature casts; conversely, `--per-channel` neutralizes background tint at the cost of altering foreground stain channel ratios.
4. **Boundary Reflection Approximation**: Mode B uses `reflect` padding at image edges; while polynomial detrending mitigates global slope reflection bias, localized dark objects touching image corners can still perturb corner background estimates.
5. **Unsupported Container Layouts**: Multi-page Z-stack/time-series TIFFs, channel-first `(C, H, W)` arrays, and non-integer inputs are rejected with explicit errors rather than guessed heuristically.
