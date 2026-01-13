from __future__ import annotations

import argparse
from math import log10

import numpy as np
import numpy.typing as npt
from PIL import Image

ImageF32 = npt.NDArray[np.float32]
VecF32 = npt.NDArray[np.float32]


def srgb2lin(x: ImageF32) -> ImageF32:
    x = x / 255.0
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return y.astype(np.float32)


def lin2srgb(y: ImageF32) -> ImageF32:
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)


def cubic_kernel(x: npt.NDArray[np.floating], a: float = -0.5) -> npt.NDArray[np.floating]:
    ax = np.abs(x)
    return np.where(
        ax <= 1,
        (a + 2) * ax**3 - (a + 3) * ax**2 + 1,
        np.where(ax < 2, a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a, 0.0),
    )


def weight_vector(scale: int = 4) -> VecF32:
    d = (scale / 2 - 1) - np.arange(scale, dtype=np.float32)
    w1d = cubic_kernel(d).astype(np.float32)
    return np.outer(w1d, w1d).astype(np.float32).reshape(-1)


# ---------- NEW: luma helpers ----------


def luma_linear(img: ImageF32) -> npt.NDArray[np.float32]:
    """
    Rec.709 luma in linear-light: Y = 0.2126 R + 0.7152 G + 0.0722 B
    Returns (H, W) float32.
    """
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def bottom_luma_mask(img: ImageF32, frac: float = 0.3) -> npt.NDArray[np.bool_]:
    """
    Boolean mask where luma is within the bottom `frac` of the image's observed luma range.
    Uses linear-light luma.
    """
    Y = luma_linear(img)
    y_min = float(Y.min())
    y_max = float(Y.max())
    thresh = y_min + frac * (y_max - y_min)
    return thresh >= Y


def embed(
    decoy: ImageF32,
    target: ImageF32,
    lam: float = 0.25,
    eps: float = 0.0,
    gamma_target: float = 1.0,
    dark_frac: float = 0.3,  # NEW: fraction of luma range to allow edits (bottom part)
) -> ImageF32:
    """
    Adjust a high-res decoy so bicubic 4:1 downscale matches `target`,
    but only modify pixels whose luma lies in the bottom `dark_frac` of the
    image's observed luma range (computed on the original decoy, in linear-light).

    Note:
        This v1 implementation updates channel 0 only, to match original behavior.
    """
    s = 4
    w_full: VecF32 = weight_vector(s)

    # Precompute where edits are allowed based on original decoy luma
    editable_mask = bottom_luma_mask(decoy, frac=dark_frac)

    adv = decoy.copy()
    tgt = (target**gamma_target).astype(np.float32)

    H_t, W_t, _ = tgt.shape
    for j in range(H_t):
        for i in range(W_t):
            y0, x0 = j * s, i * s
            blk = adv[y0 : y0 + s, x0 : x0 + s]
            blk_mask = editable_mask[y0 : y0 + s, x0 : x0 + s]

            # Flatten once for convenience
            mask_flat = blk_mask.reshape(-1)
            idx = np.flatnonzero(mask_flat)
            if idx.size == 0:
                continue  # nothing editable in this block

            # Compute current downscaled value using full weights (as the resampler would)
            # This is the quantity we try to match to tgt[j, i, c].
            for c in (0,):
                y_cur = float((w_full * blk[..., c].reshape(-1)).sum())
                diff = float(tgt[j, i, c] - y_cur)

                # Restrict solver to the editable subset
                w_sub = w_full[idx]
                M = float(w_sub.size)

                # Recompute solver terms for the subset
                sum_w_sub = float(w_sub.sum())
                w_norm2_sub = float(w_sub @ w_sub)

                # Denominator for the least-squares-with-mean term (subset version)
                denom = (M * w_norm2_sub + lam**2) - (sum_w_sub**2)
                if abs(denom) < 1e-12:
                    continue  # ill-conditioned; skip this block

                delta_sub = diff * (M * w_sub - lam * sum_w_sub) / denom

                # Optional null-space dither within the subset (only if enough DOF)
                if eps > 0.0 and w_sub.size >= 3:
                    C_sub = np.vstack(
                        [w_sub, np.ones_like(w_sub, dtype=np.float32)]
                    )  # shape (2, M)
                    _, _, Vh_sub = np.linalg.svd(C_sub, full_matrices=True)
                    # Null space spans rows 2.. of Vh
                    B_sub = Vh_sub[2:].astype(np.float32)  # ((M-2), M)
                    if B_sub.size > 0:
                        delta_sub = delta_sub + eps * (
                            B_sub.T @ np.random.randn(B_sub.shape[0])
                        ).astype(np.float32)

                # Scatter the subset update back into the 4x4 block (channel c)
                delta_vec = np.zeros_like(w_full, dtype=np.float32)
                delta_vec[idx] = delta_sub.astype(np.float32)
                blk[..., c] = blk[..., c] + delta_vec.reshape(s, s)

            adv[y0 : y0 + s, x0 : x0 + s] = blk

    return adv.astype(np.float32)


def mse_psnr(a: ImageF32, b: ImageF32) -> tuple[float, float]:
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
    ap.add_argument(
        "--dark-frac",
        type=float,
        default=0.3,
        help="fraction of luma range considered embeddable (bottom part)",
    )
    args = ap.parse_args()

    decoy_srgb = np.asarray(Image.open(args.decoy).convert("RGB"), dtype=np.float32)
    target_srgb = np.asarray(Image.open(args.target).convert("RGB"), dtype=np.float32)
    decoy_lin = srgb2lin(decoy_srgb)
    target_lin = srgb2lin(target_srgb)

    adv_lin = embed(
        decoy_lin,
        target_lin,
        lam=args.lam,
        eps=args.eps,
        gamma_target=args.gamma,
        dark_frac=args.dark_frac,
    )

    name_stub = f"adv_{args.lam:g}_{args.eps:g}_{args.gamma:g}"
    adv_img = Image.fromarray(lin2srgb(adv_lin).round().astype(np.uint8))
    adv_img.save(f"{name_stub}.png")
    print(f"✓ saved {name_stub}.png")

    mse, psnr = mse_psnr(
        target_lin,
        srgb2lin(
            np.asarray(
                adv_img.resize(
                    (target_srgb.shape[1], target_srgb.shape[0]),
                    resample=Image.BICUBIC,
                ),
                dtype=np.float32,
            )
        ),
    )
    print(f"4×→1× bicubic  |  MSE {mse:.6f}   PSNR {psnr:.2f} dB")


if __name__ == "__main__":
    main()
