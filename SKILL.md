---
name: microscopy-flatfield-correction
description: >-
  Corrects non-uniform illumination, vignetting (dark corners), and shading in
  brightfield microscopy images (e.g., Oil Red O staining, histology, brightfield
  cell culture) using adaptive pseudo flat-field correction. Outputs lossless
  flat-field corrected images with normalized background baseline.
---

# Microscopy Flat-Field Illumination Correction

## Overview

Brightfield microscopy photographs (especially low-magnification 4x, 10x, and unevenly illuminated 20x/40x images) frequently suffer from optical vignetting (dark corners), uneven lamp intensity, and field shading gradients. 

This skill provides automated pseudo flat-field correction to:
- Eliminate center-to-corner vignetting (e.g. flattening a 39% brightness drop to within 2%).
- Normalize background baseline across different batches or wells to a clean white baseline (`target_val = 220.0`).
- Preserve original staining colors, biological structures, and resolution without distortion.
- Produce lossless TIFF or PNG files ready for downstream manual inspection, ImageJ/Fiji analysis, or publication.

## Dependencies

- Python 3.9+ with `numpy`, `opencv-python`, `Pillow`.
- No external cloud APIs required; runs 100% locally.

## Quick Start

Batch correct all images in a directory:
```bash
python scripts/correct_illumination.py batch \
  --input-dir "path/to/raw_images" \
  --output-dir "path/to/corrected_images" \
  --mag 4x
```

Correct a single image:
```bash
python scripts/correct_illumination.py single \
  --input-file "path/to/raw.tif" \
  --output-file "path/to/corrected.tif" \
  --mag 20x
```

## Utility Scripts

The skill provides the CLI script `scripts/correct_illumination.py` with two subcommands:

### 1. `batch` Subcommand

Processes an entire directory of microscopy images (`.tif`, `.tiff`, `.png`, `.jpg`):

```bash
python scripts/correct_illumination.py batch [OPTIONS]
```

**Options**:
- `--input-dir`, `-i` (required): Path to folder containing raw images.
- `--output-dir`, `-o` (required): Destination folder for corrected images.
- `--mag` (optional, default: `auto`): Objective magnification (`4x`, `10x`, `20x`, `40x`, `100x`, or `auto`).
- `--sigma` (optional): Explicit Gaussian blur radius override.
- `--target-val` (optional, default: `220.0`): Target uniform background brightness.
- `--quality` (optional, default: `95`): JPEG quality if output extension is `.jpg`.

### 2. `single` Subcommand

Processes a single image file:

```bash
python scripts/correct_illumination.py single [OPTIONS]
```

**Options**:
- `--input-file`, `-i` (required): Path to raw image file.
- `--output-file`, `-o` (required): Path to output image file.
- `--mag` (optional): Objective magnification (`4x`, `10x`, `20x`, `40x`).
- `--sigma` (optional): Filter radius override.
- `--target-val` (optional, default: `220.0`): Background baseline intensity.

## Parameter Tuning & Guidelines

| Objective | Magnification Scale | Default Sigma | Description |
| :--- | :---: | :---: | :--- |
| **4x** | Wide-field macro | `75` | Wide-scale optical vignetting spanning across full field. |
| **10x** | Low-power | `60` | Moderate corner illumination drop. |
| **20x** | Mid-power | `45` | Localized illumination slopes and subtle field gradients. |
| **40x / 100x** | High-power | `40` | Subtle lens center-to-edge falloff. |

### Formula

$$
I_{\mathrm{bg}}(x, y) = \mathrm{GaussianBlur}\bigl(I(x, y),\ \sigma\bigr)
$$

$$
I_{\mathrm{corrected}}(x, y) = \mathrm{clip}\left(\frac{I(x, y)}{I_{\mathrm{bg}}(x, y) + 10^{-5}} \times I_{\mathrm{target}},\ 0,\ 255\right)
$$

## Common Mistakes

1. **Using too small a sigma**: If sigma is too small (e.g. `< 20`), the filter will smooth over larger biological structures or dense cell clusters instead of only estimating the low-frequency illumination profile, causing artificial halo artifacts around dark cells.
2. **Lossy recompression**: Brightfield microscopy images should be saved as `.tif` or `.png` to avoid JPEG compression artifacts on cell/droplet boundaries.
3. **Non-ASCII path handling**: When calling OpenCV from Windows with paths containing Chinese characters, always use memory-buffer read/write (`imdecode`/`imencode`) as implemented in `correct_illumination.py`.
