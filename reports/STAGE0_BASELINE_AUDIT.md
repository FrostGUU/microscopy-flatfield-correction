# Stage 0 Baseline Audit Report (`db715dd`)

This report documents the baseline code audit of `scripts/correct_illumination.py`, `SKILL.md`, and `README.md` prior to refactoring, along with quantitative characterization test results from `tests/test_stage0_baseline.py` (`reports/stage0_characterization_metrics.json`).

---

## 1. Does reading convert 16-bit TIFF to 8-bit?

- **Conclusion**: **Yes — 16-bit TIFFs (and grayscale images) are silently downconverted to 8-bit 3-channel BGR** (**运行验证** / Verified by execution).
- **Code Locations**:
  - `scripts/correct_illumination.py:24` — `return cv2.imdecode(data, cv2.IMREAD_COLOR)`
  - `scripts/correct_illumination.py:37-38` — `img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)` followed by `Image.fromarray(img_rgb).save(fpath)`
- **Quantitative Verification (`test_q1_legacy_converts_16bit_tiff_to_8bit`)**:
  - A synthetic 2D `uint16` TIFF (`shape=(256, 256)`, range `[4000, 62000]`, containing `4,097` distinct intensity levels) read via `imread_unicode` is returned with `dtype=uint8`, `shape=(256, 256, 3)`, range `[15, 242]`, and only `190` unique intensity levels (**`95.36%` of distinct intensity levels lost immediately upon read**).
  - After `pseudo_flatfield_correction` and `imwrite_unicode`, the saved TIFF on disk has `dtype=uint8` with `182` unique levels.

---

## 2. Do division, clipping to `0–255`, and casting to `uint8` lose intensity information?

- **Conclusion**: **Yes — severe dynamic range clipping, quantization loss, and systematic `-1 LSB` truncation bias occur** (**运行验证** / Verified by execution).
- **Code Locations**:
  - `scripts/correct_illumination.py:75-77` — `corr_b = np.clip((b / (bg_b + 1e-5)) * target_val, 0, 255)` (and lines `76`, `77` for `g`, `r`)
  - `scripts/correct_illumination.py:79` — `corrected_bgr = np.stack([corr_b, corr_g, corr_r], axis=2).astype(np.uint8)`
  - `scripts/correct_illumination.py:151,160` — `target_val` defaults to `220.0` (an 8-bit constant incompatible with 16-bit dynamic range `0–65535`).
- **Quantitative Verification (`test_q2_legacy_clips_and_truncates_intensity`)**:
  1. **Silent highlight clipping**: Any pixel whose intensity exceeds `255 / 220 ≈ 1.159×` of the local Gaussian background is hard-clipped to `255` without counting or warning. In the synthetic test image with a bright patch (`62000` over `~30000` local background), **`100.0%` of pixels in the highlight patch (`3.271%` of total image pixels) are clipped at `255`**.
  2. **Floor truncation instead of rounding**: Line `79` uses `.astype(np.uint8)` without `np.rint()`. Because line `75` adds `+ 1e-5` to the denominator, a uniform image of intensity `10` corrected to `target_val = 220.0` evaluates to `10.0 / 10.00001 * 220.0 = 219.99978`, which `.astype(np.uint8)` truncates to **`219`** (`truncation_bias_lsb = 1`).

---

## 3. Does the Gaussian background mix in real biological structures, weakening droplets/edges or causing halos?

- **Conclusion**: **Yes — unmasked Gaussian filtering absorbs dark foreground structures into `bg`, inflating foreground brightness (weakening absorption contrast) and producing bright halo rings around dense structures** (**运行验证** / Verified by execution).
- **Code Locations**:
  - `scripts/correct_illumination.py:71-73` — `bg_b = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma, sigmaY=sigma)` (unmasked full-image Gaussian blur).
  - `scripts/correct_illumination.py:96-97` — Only warns in `SKILL.md:95` about `< 20` sigma, but even at the default `sigma = 50` (`scripts/correct_illumination.py:62`), dense cell clusters or lipid droplet colonies of radius `45 px` heavily contaminate the Gaussian estimate.
- **Quantitative Verification (`test_q3_legacy_gaussian_halo_and_contrast_loss`)**:
  - For a dark circular cluster (`I_fg = 60`, radius `45 px`) on a uniform background (`I_bg = 200`, true Weber contrast `(200 - 60)/200 = 0.7000`), ideal scaling to `target_val = 220.0` should yield `I_fg_ideal = 66.0`.
  - Under `pseudo_flatfield_correction(..., sigma=50, target_val=220.0)`, the foreground mean is inflated to **`81.90`** (**`+24.09%` foreground intensity distortion**), reducing Weber contrast to `0.6346` (`90.65%` retention, **`9.35%` contrast loss**).
  - Immediately outside the cluster (`48 < r <= 68 px`), the depressed `bg` estimate causes a bright **halo ring** with mean intensity **`252.33`** (**`+14.69%` overshoot above `220.0`**) and peak intensity **`255` (saturated)**.

---

## 4. Does per-channel independent correction alter staining colors and channel ratios?

- **Conclusion**: **Yes — dividing B, G, and R by three independent unmasked Gaussian backgrounds (`bg_b`, `bg_g`, `bg_r`) alters both background color balance and foreground stain chromatic ratios** (**运行验证** / Verified by execution).
- **Code Locations**:
  - `scripts/correct_illumination.py:69-78` — Separates `b, g, r` and divides each by its own `bg_b`, `bg_g`, `bg_r` scaled to the same `target_val = 220.0`.
- **Quantitative Verification (`test_q4_legacy_per_channel_color_shift`)**:
  - On a synthetic brightfield image with a warm lamp background (`B=176, G=198, R=220`, true `R/B = 1.2500`) and red-stained lipid droplets (`B=80, G=70, R=196`, true droplet `R/G = 2.8000`, `R - G = 126.0`) covering ~45% of a central region:
    - Because red droplets absorb green (`G=70`) and blue (`B=80`) strongly while transmitting red (`R=196`), `bg_g` and `bg_b` are pulled down much more strongly than `bg_r` inside and around the droplet region.
    - Consequently, `corr_g` and `corr_b` receive a much higher gain than `corr_r`: droplet mean BGR shifts from `[80.0, 70.0, 196.0]` to **`[113.51, 90.56, 200.43]`**.
    - Droplet **`R/G` ratio drops from `2.8000` to `2.2133` (`20.95%` shift)**, and chromatic difference **`R - G` drops from `126.0` to `109.87` (`-12.8%`)**, directly altering downstream color thresholding (`R - G` or HSV segmentation).
    - Background `R/B` ratio shifts from `1.2500` to `0.9612` (`23.11%` shift).

---

## 5. Is setting `sigma` solely by objective magnification (`4x`, `10x`, `20x`, `40x`) reasonable?

- **Conclusion**: **No — objective magnification alone does not determine image pixel scale or feature size in pixels** (**仅推断 + 代码核实** / Verified by code inspection & optical principles).
- **Code Locations**:
  - `scripts/correct_illumination.py:47-60` — `get_sigma_for_mag` hardcodes `4x -> 75`, `10x -> 60`, `20x -> 45`, `40x/100x -> 40` (in pixels).
  - `SKILL.md:82-87`, `README.md:42-47` — Presents these pixel values as universal constants per objective.
- **Technical Reason**:
  - A `75 px` Gaussian kernel on a `4096 × 3000` full-resolution camera frame spans `< 2.5%` of the image width (smaller than a single adipocyte cluster or organoid), whereas on a `512 × 512` 4×4-binned or cropped image it spans `14.6%` of the field. Camera sensor pixel pitch, relay lens magnification (`0.5x` vs `1.0x` C-mount), camera binning, and ROI cropping all change the pixel dimensions of biological structures and optical vignetting.

---

## 6. Is the current algorithm over-described as true flat-field correction or "lossless" correction?

- **Conclusion**: **Yes — both `SKILL.md` and `README.md` conflate pseudo-flat-field background division with true instrument flat-field calibration and make unverified "lossless / no halo / preserves true staining chromaticity" claims** (**仅推断 + 文本核实** / Verified by text inspection).
- **Locations**:
  - `SKILL.md:6-7` — *"Outputs lossless flat-field corrected images with normalized background baseline."*
  - `SKILL.md:19-20` — *"Preserve original staining colors, biological structures, and resolution without distortion. Produce lossless TIFF or PNG files ready for downstream manual inspection, ImageJ/Fiji analysis, or publication."*
  - `README.md:13-14` — *"Preserves true staining chromaticity, cellular morphology, and lipid droplet boundaries without halo artifacts or resolution loss. Lossless Output & Unicode Path Safe: Outputs baseline-compliant `.tif`/`.tiff` or `.png` ready for ImageJ/Fiji quantification..."*
  - As proven in Questions 1–4 above, the algorithm modifies pixel intensities non-uniformly, clips highlights, shifts RGB channel ratios by `>20%`, and induces `+14.7%` halo overshoot around dense dark structures.

---

## 7. Additional Audits

1. **Gamma / Linearity Issue**:
   - **Location**: `scripts/correct_illumination.py:68-78` (**仅推断 + 理论核实**).
   - 8-bit PNG/JPEG and color camera TIFFs are typically gamma-encoded ($I_{\text{sRGB}} \approx I_{\text{linear}}^{1/\gamma}$, $\gamma \approx 2.2$). The multiplicative illumination model $I(x,y) = L(x,y) \cdot T(x,y)$ holds in **linear photon flux space**; dividing gamma-encoded pixel values directly introduces nonlinear tone distortion in dark corners.
2. **Boundary Handling**:
   - **Location**: `scripts/correct_illumination.py:71-73` (**仅推断 + 代码核实**).
   - Calls `cv2.GaussianBlur(b, (0, 0), sigmaX=sigma, sigmaY=sigma)` without specifying `borderType` (relying on OpenCV's implicit default `BORDER_DEFAULT` / `BORDER_REFLECT_101`), and never documents how dark corners or objects touching image borders affect corner gain estimates.
3. **Unicode / Chinese Path Behavior (`imdecode`/`imencode`)**:
   - **Location**: `scripts/correct_illumination.py:17-45` (**运行验证**).
   - While `np.fromfile` + `cv2.imdecode` avoids Windows ANSI path failures, `imwrite_unicode` (`line 35-38`) branches to `Image.fromarray(img_rgb).save(fpath)` for `.tif`/`.tiff`, while using `cv2.imencode` for `.png`/`.jpg`, and `imread_unicode` (`line 24`) hardcodes `cv2.IMREAD_COLOR`, discarding 16-bit depth and single-channel grayscale structure, and converting RGBA `(H,W,4)` by dropping the Alpha channel.
4. **Unreliability of `--mag auto` Folder-Name Inference**:
   - **Location**: `scripts/correct_illumination.py:115-116` (`detect_str = mag if mag != "auto" else os.path.basename(os.path.abspath(input_dir))`) (**运行验证**).
   - Performs substring matching (`'4x' in m`, `'10x' in m`, `'20x' in m`, `'40x' in m` at `lines 52-59`). If a folder is named `exp_10x_diluted_4x_wash` (`'4x'` matched first because of `if/elif` order!) or `sample_2024_batch1` (no match -> silent fallback to `default_sigma=50`), the inferred sigma is arbitrary and unrecorded.
5. **Output Overwrite Risk**:
   - **Location**: `scripts/correct_illumination.py:82-92`, `94-136` (**运行验证**).
   - Neither `process_single_image` nor `process_batch` checks whether `output_path == input_path` or `os.path.abspath(output_dir) == os.path.abspath(input_dir)`. Running `batch -i ./raw -o ./raw` overwrites all original raw experimental images in-place without confirmation or backup.
