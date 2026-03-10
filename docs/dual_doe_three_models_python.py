"""Python port of compare_dual_doe_three_models_rigorous.m

Usage:
  python docs/dual_doe_three_models_python.py \
      --doe1 "圆形DOE1_入射场整形.dat" \
      --doe2 "圆形DOE2_精细调控.dat"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


@dataclass
class DOEInfo:
    N: int
    dx_m: float


def read_doe_phase_dat(filename: str) -> tuple[np.ndarray, DOEInfo]:
    with open(filename, "r", encoding="utf-8") as f:
        raw = np.fromstring(f.read(), sep=" ")
    if raw.size < 7:
        raise ValueError(f"DAT header invalid: {filename}")

    header = raw[:7]
    N1, N2 = int(round(header[0])), int(round(header[1]))
    if N1 != N2:
        raise ValueError(f"Only square DOE is supported, got {N1}x{N2}")

    N = N1
    dx_m = float(header[2]) * 1e-3  # mm -> m

    body = raw[7:]
    if body.size < 5 * N * N:
        raise ValueError(f"DAT body insufficient, need >= {5*N*N} floats")

    data = body.reshape(-1, 5)
    phase_vec = data[: N * N, 0]
    phase = phase_vec.reshape(N, N)
    return phase, DOEInfo(N=N, dx_m=dx_m)


def frequency_grid(N: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    f = (np.arange(N) - (N // 2)) / (N * dx)
    fx, fy = np.meshgrid(f, f)
    return fx, fy


def propagate_asm_bandlimited(Uin: np.ndarray, dx: float, wl: float, z: float) -> np.ndarray:
    N = Uin.shape[0]
    k = 2 * np.pi / wl
    fx, fy = frequency_grid(N, dx)
    arg = 1 - (wl * fx) ** 2 - (wl * fy) ** 2
    Hc = np.zeros_like(Uin, dtype=np.complex128)
    mask = arg >= 0
    Hc[mask] = np.exp(1j * k * z * np.sqrt(arg[mask]))
    H = np.fft.ifftshift(Hc)
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(Uin)) * H))


def propagate_fftrs_transfer(Uin: np.ndarray, dx: float, wl: float, z: float, zero_evanescent: bool = True) -> np.ndarray:
    N = Uin.shape[0]
    k = 2 * np.pi / wl
    fx, fy = frequency_grid(N, dx)
    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    kz = np.sqrt((k**2 - kx**2 - ky**2).astype(np.complex128))
    Hc = np.exp(1j * z * kz)
    if zero_evanescent:
        Hc[np.abs(np.imag(kz)) > 1e-14] = 0
    H = np.fft.ifftshift(Hc)
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(Uin)) * H))


def propagate_fresnel_transfer(Uin: np.ndarray, dx: float, wl: float, z: float) -> np.ndarray:
    N = Uin.shape[0]
    k = 2 * np.pi / wl
    fx, fy = frequency_grid(N, dx)
    Hc = np.exp(1j * k * z) * np.exp(-1j * np.pi * wl * z * (fx**2 + fy**2))
    H = np.fft.ifftshift(Hc)
    return np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(Uin)) * H))


def propagate_spectral_to_custom_grid(Uin: np.ndarray, dx: float, wl: float, z: float, x_out: np.ndarray, y_out: np.ndarray, mode: str, zero_evanescent: bool = True) -> np.ndarray:
    N = Uin.shape[0]
    Gu = np.fft.fft2(np.fft.ifftshift(Uin))

    f_center = (np.arange(N) - (N // 2)) / (N * dx)
    fx = np.fft.ifftshift(f_center)
    fy = fx
    FX, FY = np.meshgrid(fx, fy)

    mode = mode.lower()
    if mode == "fftrs":
        k = 2 * np.pi / wl
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY
        kz = np.sqrt((k**2 - kx**2 - ky**2).astype(np.complex128))
        H = np.exp(1j * z * kz)
        if zero_evanescent:
            H[np.abs(np.imag(kz)) > 1e-14] = 0
    elif mode == "fresnel":
        k = 2 * np.pi / wl
        H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wl * z * (FX**2 + FY**2))
    else:
        raise ValueError(f"Unknown mode={mode}")

    Gpu = Gu * H
    Ex = np.exp(1j * 2 * np.pi * np.outer(fx, x_out))
    Ey = np.exp(1j * 2 * np.pi * np.outer(y_out, fy))
    return (Ey @ Gpu @ Ex) / (N**2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe1", required=True)
    ap.add_argument("--doe2", required=True)
    ap.add_argument("--lambda-nm", type=float, default=1080)
    ap.add_argument("--f-cm", type=float, default=500)
    ap.add_argument("--d12-cm", type=float, default=15)
    ap.add_argument("--sigma-mm", type=float, default=5)
    ap.add_argument("--pad-factor", type=float, default=4)
    ap.add_argument("--obs-n", type=int, default=501)
    args = ap.parse_args()

    phi1, info1 = read_doe_phase_dat(args.doe1)
    phi2, info2 = read_doe_phase_dat(args.doe2)
    if info1.N != info2.N:
        raise ValueError("DOE sizes mismatch")

    N = info1.N
    dx = info1.dx_m
    wl = args.lambda_nm * 1e-9
    f = args.f_cm * 1e-2
    d12 = args.d12_cm * 1e-2
    sigma = args.sigma_mm * 1e-3

    Nsim = int(round(N * args.pad_factor))
    if Nsim % 2 == 0:
        Nsim += 1
    obs_N = args.obs_n + (args.obs_n % 2 == 0)

    x_eff = (np.arange(N) - (N // 2)) * dx
    X_eff, Y_eff = np.meshgrid(x_eff, x_eff)

    x_sim = (np.arange(Nsim) - (Nsim // 2)) * dx
    X_sim, Y_sim = np.meshgrid(x_sim, x_sim)

    s = (Nsim - N) // 2
    e = s + N
    aperture = np.zeros((Nsim, Nsim), dtype=bool)
    aperture[s:e, s:e] = True

    U1_eff = np.exp(-(X_eff**2 + Y_eff**2) / (2 * sigma**2))
    U1_eff /= U1_eff.max()
    U1_eff *= np.exp(1j * phi1)

    U1 = np.zeros((Nsim, Nsim), dtype=np.complex128)
    U1[s:e, s:e] = U1_eff
    phi2_pad = np.zeros((Nsim, Nsim))
    phi2_pad[s:e, s:e] = phi2
    phi_lens = -(np.pi / (wl * f)) * (X_sim**2 + Y_sim**2)

    U2_asm = propagate_asm_bandlimited(U1, dx, wl, d12)
    U2_fftrs = propagate_fftrs_transfer(U1, dx, wl, d12)
    U2_fres = propagate_fresnel_transfer(U1, dx, wl, d12)

    U_after_asm = U2_asm * aperture * np.exp(1j * phi2_pad)
    U_after_fftrs = U2_fftrs * aperture * np.exp(1j * phi2_pad) * np.exp(1j * phi_lens)
    U_after_fres = U2_fres * aperture * np.exp(1j * phi2_pad) * np.exp(1j * phi_lens)

    Uf_asm_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_after_asm)))
    I_asm_full = np.abs(Uf_asm_full) ** 2
    I_asm_full /= I_asm_full.max() + np.finfo(float).eps

    c0 = (Nsim - obs_N) // 2
    c1 = c0 + obs_N
    I_focus_asm = I_asm_full[c0:c1, c0:c1]

    dy = wl * f / (Nsim * dx)
    u = (np.arange(obs_N) - (obs_N // 2)) * dy
    v = u
    Uf_fftrs = propagate_spectral_to_custom_grid(U_after_fftrs, dx, wl, f, u, v, "fftrs")
    Uf_fres = propagate_spectral_to_custom_grid(U_after_fres, dx, wl, f, u, v, "fresnel", False)
    I_focus_fftrs = np.abs(Uf_fftrs) ** 2
    I_focus_fres = np.abs(Uf_fres) ** 2
    I_focus_fftrs /= I_focus_fftrs.max() + np.finfo(float).eps
    I_focus_fres /= I_focus_fres.max() + np.finfo(float).eps

    rel_fftrs = np.linalg.norm(I_focus_fftrs - I_focus_asm) / (np.linalg.norm(I_focus_asm) + np.finfo(float).eps)
    rel_fres = np.linalg.norm(I_focus_fres - I_focus_asm) / (np.linalg.norm(I_focus_asm) + np.finfo(float).eps)
    print(f"N={N}, Nsim={Nsim}, obs_N={obs_N}, dy={dy*1e6:.3f} um")
    print(f"Focal relative error: FFT-RS vs ASM={rel_fftrs:.6e}")
    print(f"Focal relative error: Fresnel vs ASM={rel_fres:.6e}")


if __name__ == "__main__":
    main()
