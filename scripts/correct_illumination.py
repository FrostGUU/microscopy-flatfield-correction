"""
Microscopy Flat-Field Illumination Correction CLI
-------------------------------------------------
Adaptive pseudo flat-field illumination correction for brightfield microscopy
(Oil Red O, HE staining, immunohistochemistry, brightfield cell culture).
Eliminates vignetting (dark corners), shading, and lighting gradients.
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
from PIL import Image

def imread_unicode(fpath):
    """Safely read image files with non-ASCII characters in filepath."""
    if not os.path.exists(fpath):
        return None
    data = np.fromfile(fpath, dtype=np.uint8)
    if len(data) == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(fpath, img_bgr, quality=95):
    """Safely save image files with non-ASCII characters in filepath."""
    ext = os.path.splitext(fpath)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    if ext in ['.jpg', '.jpeg']:
        success, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            enc.tofile(fpath)
            return True
    elif ext in ['.tif', '.tiff']:
        # Use PIL for 100% compliant baseline TIFF saving
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(fpath)
        return True
    else:
        success, enc = cv2.imencode(ext if ext else '.png', img_bgr)
        if success:
            enc.tofile(fpath)
            return True
    return False

def get_sigma_for_mag(mag_str, default_sigma=50):
    """Map magnification to recommended low-frequency background Gaussian filter sigma."""
    if not mag_str:
        return default_sigma
    m = mag_str.lower().strip()
    if '4x' in m:
        return 75
    elif '10x' in m:
        return 60
    elif '20x' in m:
        return 45
    elif '40x' in m or '100x' in m:
        return 40
    return default_sigma

def pseudo_flatfield_correction(img_bgr, sigma=50, target_val=220.0):
    """
    Perform pseudo flat-field correction by dividing each channel
    by a large-scale Gaussian-filtered background estimation.
    Eliminates center-to-corner vignetting and normalizes illumination baseline.
    """
    img_f = img_bgr.astype(np.float32)
    b, g, r = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    
    bg_b = cv2.GaussianBlur(b, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_r = cv2.GaussianBlur(r, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    corr_b = np.clip((b / (bg_b + 1e-5)) * target_val, 0, 255)
    corr_g = np.clip((g / (bg_g + 1e-5)) * target_val, 0, 255)
    corr_r = np.clip((r / (bg_r + 1e-5)) * target_val, 0, 255)
    
    corrected_bgr = np.stack([corr_b, corr_g, corr_r], axis=2).astype(np.uint8)
    return corrected_bgr

def process_single_image(input_path, output_path, sigma, target_val, quality=95):
    """Process a single image file."""
    img = imread_unicode(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    corrected = pseudo_flatfield_correction(img, sigma=sigma, target_val=target_val)
    success = imwrite_unicode(output_path, corrected, quality=quality)
    if not success:
        raise IOError(f"Could not write image to: {output_path}")
    return True

def process_batch(input_dir, output_dir, mag="auto", sigma=None, target_val=220.0, extensions=None, quality=95):
    """Process all images in a directory."""
    if extensions is None:
        extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        
    os.makedirs(output_dir, exist_ok=True)
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        all_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    all_files = sorted(list(set(all_files)))
    
    if not all_files:
        print(f"Warning: No images found in {input_dir}")
        return 0
        
    # Determine sigma
    if sigma is not None:
        effective_sigma = sigma
    else:
        # Detect from mag or folder name
        detect_str = mag if mag != "auto" else os.path.basename(os.path.abspath(input_dir))
        effective_sigma = get_sigma_for_mag(detect_str, default_sigma=50)
        
    print(f"Processing {len(all_files)} images from: {input_dir}")
    print(f"  - Target baseline: {target_val}")
    print(f"  - Smoothing sigma: {effective_sigma}")
    print(f"  - Destination: {output_dir}")
    
    count = 0
    for idx, fpath in enumerate(all_files, 1):
        fname = os.path.basename(fpath)
        out_fpath = os.path.join(output_dir, fname)
        try:
            process_single_image(fpath, out_fpath, sigma=effective_sigma, target_val=target_val, quality=quality)
            count += 1
            if idx % 10 == 0 or idx == len(all_files):
                print(f"  [{idx:03d}/{len(all_files):03d}] Processed {fname}")
        except Exception as e:
            print(f"  [ERROR] Failed to process {fname}: {e}", file=sys.stderr)
            
    print(f"Successfully processed and saved {count}/{len(all_files)} images to: {output_dir}")
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Microscopy Flat-Field Illumination Correction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: single
    p_single = subparsers.add_parser("single", help="Correct illumination for a single image file")
    p_single.add_argument("--input-file", "-i", required=True, help="Path to input image")
    p_single.add_argument("--output-file", "-o", required=True, help="Path to output corrected image")
    p_single.add_argument("--mag", default=None, choices=["4x", "10x", "20x", "40x", "100x"], help="Magnification (used to select sigma)")
    p_single.add_argument("--sigma", type=float, default=None, help="Explicit Gaussian filter sigma (overrides --mag)")
    p_single.add_argument("--target-val", type=float, default=220.0, help="Target uniform background baseline (default: 220.0)")
    p_single.add_argument("--quality", type=int, default=95, help="JPEG quality if output is JPG (default: 95)")
    
    # Subcommand: batch
    p_batch = subparsers.add_parser("batch", help="Batch correct illumination for all images in a directory")
    p_batch.add_argument("--input-dir", "-i", required=True, help="Directory containing raw microscopy images")
    p_batch.add_argument("--output-dir", "-o", required=True, help="Directory to save corrected images")
    p_batch.add_argument("--mag", default="auto", help="Magnification (auto, 4x, 10x, 20x, 40x, 100x)")
    p_batch.add_argument("--sigma", type=float, default=None, help="Explicit Gaussian filter sigma (overrides --mag)")
    p_batch.add_argument("--target-val", type=float, default=220.0, help="Target uniform background baseline (default: 220.0)")
    p_batch.add_argument("--quality", type=int, default=95, help="JPEG quality if output is JPG (default: 95)")
    
    args = parser.parse_args()
    
    if args.command == "single":
        sigma = args.sigma if args.sigma is not None else get_sigma_for_mag(args.mag, default_sigma=50)
        process_single_image(args.input_file, args.output_file, sigma=sigma, target_val=args.target_val, quality=args.quality)
        print(f"Success! Corrected image saved to: {args.output_file}")
    elif args.command == "batch":
        process_batch(args.input_dir, args.output_dir, mag=args.mag, sigma=args.sigma, target_val=args.target_val, quality=args.quality)

if __name__ == "__main__":
    main()
