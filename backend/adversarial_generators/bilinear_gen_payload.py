from __future__ import annotations

import argparse
from math import log10
from typing import Tuple

import numpy as np
import numpy.typing as npt
import cv2

ImageF32 = npt.NDArray[np.float32]
VecF32 = npt.NDArray[np.float32]


def srgb2lin(x: ImageF32) -> ImageF32:
    x = x / 255.0
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return y.astype(np.float32)

def lin2srgb(y: ImageF32) -> ImageF32:
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)

def bilinear_kernel(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Linear (triangle) kernel for bilinear interpolation"""
    ax = np.abs(x)
    return np.where(ax <= 1, 1 - ax, 0.0)

def weight_vector_bilinear(scale: int = 4) -> VecF32:
    """
    Compute bilinear weights for a 2x2 region when downsampling by `scale`.
    For scale=4, OpenCV's bilinear uses only the 2x2 pixels at the center.
    """
    # For bilinear with scale=4, we sample at position (1.5, 1.5) in the 4x4 block
    # This gives us a 2x2 kernel centered around that point
    weights = np.zeros((scale, scale), dtype=np.float32)
    
    # The sample point in the source image for pixel (i,j) in destination
    # is at (i*scale + (scale-1)/2, j*scale + (scale-1)/2)
    # For scale=4, this is (1.5, 1.5) relative to the top-left of the 4x4 block
    
    center = (scale - 1) / 2.0  # 1.5 for scale=4
    
    for y in range(scale):
        for x in range(scale):
            # Distance from sampling point to pixel center
            dy = abs(y - center)
            dx = abs(x - center)
            
            # Bilinear weight is product of 1D linear weights
            if dy < 1.0 and dx < 1.0:
                weights[y, x] = (1.0 - dy) * (1.0 - dx)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    return weights.astype(np.float32).reshape(-1)

# ---------- luma helpers ----------

def luma_linear(img: ImageF32) -> npt.NDArray[np.float32]:
    """
    Rec.709 luma in linear-light: Y = 0.2126 R + 0.7152 G + 0.0722 B
    Returns (H, W) float32.
    """
    return (
        0.2126 * img[..., 0]
        + 0.7152 * img[..., 1]
        + 0.0722 * img[..., 2]
    ).astype(np.float32)

def bottom_luma_mask(img: ImageF32, frac: float = 0.3) -> npt.NDArray[np.bool_]:
    """
    Boolean mask where luma is within the bottom `frac` of the image's observed luma range.
    Uses linear-light luma.
    """
    Y = luma_linear(img)
    y_min = float(Y.min())
    y_max = float(Y.max())
    thresh = y_min + frac * (y_max - y_min)
    return (Y <= thresh)

def embed_bilinear(
    decoy: ImageF32,
    target: ImageF32,
    lam: float = 0.25,
    eps: float = 0.0,
    gamma_target: float = 1.0,
    dark_frac: float = 0.3,
) -> ImageF32:
    """
    Adjust a high-res decoy so bilinear 4:1 downscale matches `target`,
    but only modify pixels whose luma lies in the bottom `dark_frac` of the
    image's observed luma range (computed on the original decoy, in linear-light).
    """
    s = 4
    w_full: VecF32 = weight_vector_bilinear(s)
    sum_w_full = float(w_full.sum())

    # Precompute where edits are allowed based on original decoy luma
    editable_mask = bottom_luma_mask(decoy, frac=dark_frac)

    adv = decoy.copy()
    tgt = (target ** gamma_target).astype(np.float32)

    H_t, W_t, _ = tgt.shape
    for j in range(H_t):
        for i in range(W_t):
            y0, x0 = j * s, i * s
            blk = adv[y0:y0 + s, x0:x0 + s]
            blk_mask = editable_mask[y0:y0 + s, x0:x0 + s]

            # Flatten once for convenience
            mask_flat = blk_mask.reshape(-1)
            idx = np.flatnonzero(mask_flat)
            if idx.size == 0:
                continue  # nothing editable in this block

            # Process only red channel to get red background with black text
            for c in (0,):  # Only modify red channel (channel 0)
                y_cur = float((w_full * blk[..., c].reshape(-1)).sum())
                diff = float(tgt[j, i, c] - y_cur)

                # Restrict solver to the editable subset
                w_sub = w_full[idx]
                M = float(w_sub.size)

                # Recompute solver terms for the subset
                sum_w_sub = float(w_sub.sum())
                w_norm2_sub = float(w_sub @ w_sub)

                # Denominator for the least-squares-with-mean term (subset version)
                denom = (M * w_norm2_sub + lam**2) - (sum_w_sub ** 2)
                if abs(denom) < 1e-12:
                    continue  # ill-conditioned; skip this block

                delta_sub = diff * (M * w_sub - lam * sum_w_sub) / denom

                # Optional null-space dither within the subset (only if enough DOF)
                if eps > 0.0 and w_sub.size >= 3:
                    C_sub = np.vstack([w_sub, np.ones_like(w_sub, dtype=np.float32)])
                    _, _, Vh_sub = np.linalg.svd(C_sub, full_matrices=True)
                    B_sub = Vh_sub[2:].astype(np.float32)
                    if B_sub.size > 0:
                        delta_sub = delta_sub + eps * (B_sub.T @ np.random.randn(B_sub.shape[0])).astype(np.float32)

                # Scatter the subset update back into the 4x4 block
                delta_vec = np.zeros_like(w_full, dtype=np.float32)
                delta_vec[idx] = delta_sub.astype(np.float32)
                blk[..., c] = blk[..., c] + delta_vec.reshape(s, s)

            adv[y0:y0 + s, x0:x0 + s] = blk

    return adv.astype(np.float32)

def mse_psnr(a: ImageF32, b: ImageF32) -> Tuple[float, float]:
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * log10(1.0 / mse)
    return mse, psnr

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--decoy", required=True, help="decoy image (PNG/JPEG)")
    ap.add_argument("--target", required=True, help="target image (PNG/JPEG)")
    ap.add_argument("--lam", type=float, default=0.25, help="mean-preservation weight")
    ap.add_argument("--eps", type=float, default=0.0, help="null-space dither")
    ap.add_argument("--gamma", type=float, default=1.0, help="target gamma pre-emphasis")
    ap.add_argument("--dark-frac", type=float, default=0.3,
                    help="fraction of luma range considered embeddable (bottom part)")
    ap.add_argument("--anti-alias", action="store_true", 
                    help="use anti-aliased bilinear (INTER_LINEAR) instead of INTER_LINEAR_EXACT")
    args = ap.parse_args()

    # Load images using OpenCV (loads as BGR)
    decoy_bgr = cv2.imread(args.decoy, cv2.IMREAD_COLOR).astype(np.float32)
    target_bgr = cv2.imread(args.target, cv2.IMREAD_COLOR).astype(np.float32)
    
    # Convert BGR to RGB
    decoy_srgb = cv2.cvtColor(decoy_bgr, cv2.COLOR_BGR2RGB)
    target_srgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to linear space
    decoy_lin = srgb2lin(decoy_srgb)
    target_lin = srgb2lin(target_srgb)

    adv_lin = embed_bilinear(
        decoy_lin,
        target_lin,
        lam=args.lam,
        eps=args.eps,
        gamma_target=args.gamma,
        dark_frac=args.dark_frac,
    )

    # Convert back to sRGB
    adv_srgb = lin2srgb(adv_lin).round().astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV saving
    adv_bgr = cv2.cvtColor(adv_srgb, cv2.COLOR_RGB2BGR)
    
    # Save the image
    name_stub = f"adv_bilinear_{args.lam:g}_{args.eps:g}_{args.gamma:g}"
    cv2.imwrite(f"{name_stub}.png", adv_bgr)
    print(f"✓ saved {name_stub}.png")

    # Choose interpolation method based on anti-alias flag
    interp_method = cv2.INTER_LINEAR if args.anti_alias else cv2.INTER_LINEAR_EXACT
    interp_name = "INTER_LINEAR" if args.anti_alias else "INTER_LINEAR_EXACT"
    
    # Downsample using OpenCV bilinear for verification
    downsampled_bgr = cv2.resize(
        adv_bgr,
        (target_srgb.shape[1], target_srgb.shape[0]),
        interpolation=interp_method
    )
    
    # Convert back to RGB and linear space for comparison
    downsampled_rgb = cv2.cvtColor(downsampled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    downsampled_lin = srgb2lin(downsampled_rgb)

    mse, psnr = mse_psnr(target_lin, downsampled_lin)
    print(f"4×→1× OpenCV {interp_name}  |  MSE {mse:.6f}   PSNR {psnr:.2f} dB")

if __name__ == "__main__":
    main()