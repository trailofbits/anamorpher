from __future__ import annotations

import argparse
from math import log10

import numpy as np
import numpy.typing as npt
from PIL import Image

# ---------------------------------------------------------------------------
# Types
ImageF32 = npt.NDArray[np.float32]
VecF32 = npt.NDArray[np.float32]

# ---------------------------------------------------------------------------
# Color transforms


def srgb2lin(x: ImageF32) -> ImageF32:
    """
    Convert sRGB (0..255) to linear-light (~0..1), elementwise.

    Args:
        x: (H, W, 3) float32 array interpreted as sRGB in the 0..255 range.

    Returns:
        Linear-light image of the same shape, float32.
    """
    x = x / 255.0
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return y.astype(np.float32)


def lin2srgb(y: ImageF32) -> ImageF32:
    """
    Convert linear-light (~0..1) to sRGB (0..255), elementwise.

    Args:
        y: (H, W, 3) float32 array in linear-light.

    Returns:
        sRGB image in the same shape, float32 in 0..255 (not rounded).
    """
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Core


def embed_nn(
    decoy: ImageF32,
    target: ImageF32,
    lam: float = 0.25,
    eps: float = 0.0,
    gamma_target: float = 1.0,
    offset: int = 2,
) -> ImageF32:
    """
    Adjust a high-res decoy so 4:1 nearest-neighbor downscale matches `target`.

    For PIL's NEAREST, shrinking by an exact factor of 4 samples the CENTER
    of each 4×4 block. That's (offset, offset) with offset=2 (0-based).
    This solver enforces that selected sample to equal the target, and
    optionally spreads energy to the other 15 pixels to roughly preserve the
    block mean controlled by λ. A small null-space term (ε) can be added to
    spread energy without changing the selected sample or the block mean.

    Note:
        v1 updates channel 0 only, to mirror the original behavior.

    Args:
        decoy: Source image at 4× the target resolution, linear-light float32.
        target: Desired appearance after 4:1 NN downscale, linear-light float32.
        lam: Mean-preservation weight (λ). λ=0 touches only the selected sample;
             λ→∞ approximately preserves block mean.
        eps: Dither magnitude along the null space (0 disables).
        gamma_target: Exponent applied to `target` before fitting.
        offset: Which pixel inside each 4×4 block NN will sample (default 2).

    Returns:
        Modified decoy in linear-light.
    """
    s = 4
    n = s * s
    k = offset * s + offset  # index of selected sample within the 4×4 block

    # Null space for constraints [e_k ; ones], so we can add dither that keeps
    # both the chosen pixel and the block mean unchanged.
    e = np.zeros(n, dtype=np.float32)
    e[k] = 1.0
    C = np.vstack([e, np.ones(n, dtype=np.float32)])  # (2,16)
    _, _, Vh = np.linalg.svd(C, full_matrices=True)
    B = Vh[2:].astype(np.float32)  # (14,16), basis for null space

    adv = decoy.copy()
    tgt = (target**gamma_target).astype(np.float32)

    H_t, W_t, _ = tgt.shape
    for j in range(H_t):
        for i in range(W_t):
            y0, x0 = j * s, i * s
            blk = adv[y0 : y0 + s, x0 : x0 + s]

            for c in (0,):  # v1: channel 0 only
                cur = float(blk[offset, offset, c])
                diff = float(tgt[j, i, c] - cur)

                if lam <= 0.0:
                    # Simple: set selected sample to the target value.
                    blk[offset, offset, c] = cur + diff
                else:
                    # Closed-form solution of:
                    #   minimize ||δ||^2 + λ^2 (sum δ)^2  s.t.  δ_k = diff
                    # yields:
                    #   δ_k = diff
                    #   δ_j (j!=k) = -diff * λ^2 / (1 + 15 λ^2)
                    denom = 1.0 + 15.0 * (lam**2)
                    delta_other = -diff * (lam**2) / denom
                    blk[..., c] = blk[..., c] + delta_other
                    # Fix center to hit target exactly after the uniform shift:
                    # (because we just added delta_other to all 16 entries)
                    blk[offset, offset, c] = cur + diff  # ensures δ_k = diff

                if eps > 0.0:
                    # Dither in null space: doesn't change selected sample or mean.
                    z = (B.T @ np.random.randn(B.shape[0]).astype(np.float32)).reshape(s, s)
                    blk[..., c] = blk[..., c] + eps * z

            adv[y0 : y0 + s, x0 : x0 + s] = blk

    return adv.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics


def mse_psnr(a: ImageF32, b: ImageF32) -> tuple[float, float]:
    """
    Mean squared error and PSNR (peak = 1.0) between two linear-light images.
    """
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * log10(1.0 / mse)
    return mse, psnr


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    """
    CLI: embed target into decoy and verify via nearest-neighbor downscale.

    Example:
        python3 gen_payload_nn.py --target images/target.png --decoy images/decoy.png \
            --eps 0.0 --lam 0.25 --gamma 1.0 --offset 2
    """
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--decoy", required=True, help="decoy image (PNG/JPEG)")
    ap.add_argument("--target", required=True, help="target image (PNG/JPEG)")
    ap.add_argument("--lam", type=float, default=0.25, help="mean-preservation weight")
    ap.add_argument("--eps", type=float, default=0.0, help="null-space dither")
    ap.add_argument("--gamma", type=float, default=1.0, help="target gamma pre-emphasis")
    ap.add_argument(
        "--offset",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="which pixel inside the 4×4 block NN selects (PIL uses 2)",
    )
    args = ap.parse_args()

    decoy_srgb = np.asarray(Image.open(args.decoy).convert("RGB"), dtype=np.float32)
    target_srgb = np.asarray(Image.open(args.target).convert("RGB"), dtype=np.float32)

    # Basic shape check: decoy must be 4× the target resolution
    Ht, Wt = target_srgb.shape[:2]
    Hd, Wd = decoy_srgb.shape[:2]
    assert Hd == 4 * Ht and Wd == 4 * Wt, (
        f"decoy must be 4× target size (got decoy {Wd}×{Hd}, target {Wt}×{Ht})"
    )

    decoy_lin = srgb2lin(decoy_srgb)
    target_lin = srgb2lin(target_srgb)

    adv_lin = embed_nn(
        decoy_lin,
        target_lin,
        lam=args.lam,
        eps=args.eps,
        gamma_target=args.gamma,
        offset=args.offset,
    )

    name_stub = f"advNN_{args.lam:g}_{args.eps:g}_{args.gamma:g}_off{args.offset}"
    adv_img = Image.fromarray(lin2srgb(adv_lin).round().astype(np.uint8))
    adv_img.save(f"{name_stub}.png")
    print(f"✓ saved {name_stub}.png")

    # Verify with PIL's NEAREST (uses the center pixel for 4:1)
    try:
        Resampling = Image.Resampling  # Pillow ≥9.1
    except AttributeError:
        Resampling = Image  # fallback for older Pillow

    down_img = adv_img.resize(
        (target_srgb.shape[1], target_srgb.shape[0]), resample=Resampling.NEAREST
    )
    down_img.save(f"{name_stub}_down.png")
    print(f"✓ saved {name_stub}_down.png")

    mse, psnr = mse_psnr(
        target_lin,
        srgb2lin(np.asarray(down_img, dtype=np.float32)),
    )
    print(f"4×→1× nearest  |  MSE {mse:.6f}   PSNR {psnr:.2f} dB")


if __name__ == "__main__":
    main()
