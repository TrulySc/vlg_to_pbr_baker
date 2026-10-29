# SPDX-License-Identifier: GPL-3.0-or-later
"""
VertexLitGeneric -> PBR conversion core.

Pure numpy, no bpy import, so it can be unit-tested outside Blender.

All image arrays are float32, shape (H, W, C), in Blender's pixel order
(row 0 = bottom of the image).  "Encoded" means the raw stored texture
value in 0..1 (sRGB-encoded for colour textures, raw data for normal /
mask textures).

The maths follows the Source SDK 2013 VertexLitGeneric pixel shader:
  * diffuse  = albedo * 2 * lightwarp(halflambert(N.L))      (when warped)
  * phong    = mask * boost * fresnelRanges(N.V) * (R.L)^n * N.L
  * rim      = rimmask * rimboost * (1-N.V)^4 * ambient * saturate(N.z)
and converts each term to the closest per-texel PBR quantity:
  * phong exponent  -> GGX roughness (Phong n ~ Blinn 4n ~ Beckmann a^2 = 2/(4n+2))
  * phong intensity -> F0, matched on highlight peak radiance, Schlick-fitted
                       to the $phongfresnelranges curve
  * rimlight        -> sheen weight/roughness (fitted against Cycles' sheen lobe)
  * lightwarp       -> albedo brightness/hue calibration + exported ramp
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# colour helpers
# --------------------------------------------------------------------------

def srgb_to_linear(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def gamma_to_linear(c):
    """Source converts VMT colour vars with a pure 2.2 power (GammaToLinear)."""
    return np.power(np.clip(np.asarray(c, dtype=np.float32), 0, None), 2.2)


def luminance(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def resample(img: np.ndarray, h: int, w: int, tile: float = 1.0) -> np.ndarray:
    """Bilinear resample of a UV-space texture onto an h x w grid (wrap addressing).
    `tile` repeats the texture (used for $detailscale)."""
    sh, sw = img.shape[:2]
    if sh == h and sw == w and tile == 1.0:
        return img.astype(np.float32, copy=False)
    v = (np.arange(h, dtype=np.float32) + 0.5) / h * tile * sh - 0.5
    u = (np.arange(w, dtype=np.float32) + 0.5) / w * tile * sw - 0.5
    v0 = np.floor(v).astype(np.int64); fv = (v - v0)[:, None, None]
    u0 = np.floor(u).astype(np.int64); fu = (u - u0)[None, :, None]
    v1 = (v0 + 1) % sh; u1 = (u0 + 1) % sw
    v0 %= sh; u0 %= sw
    a = img[v0][:, u0]; b = img[v0][:, u1]
    c = img[v1][:, u0]; d = img[v1][:, u1]
    top = a + (b - a) * fu
    bot = c + (d - c) * fu
    return (top + (bot - top) * fv).astype(np.float32)


def sample_ramp(ramp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ramp: (N, 3) ; x in 0..1 -> (..., 3), linear filtering, clamp addressing."""
    n = ramp.shape[0]
    p = np.clip(x, 0, 1) * (n - 1)
    i0 = np.floor(p).astype(np.int64); i1 = np.minimum(i0 + 1, n - 1)
    f = (p - i0)[..., None]
    return ramp[i0] + (ramp[i1] - ramp[i0]) * f


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class VLGParams:
    # albedo
    color2: tuple = (1.0, 1.0, 1.0)            # gamma-space, as written in the VMT
    blendtintbybasealpha: bool = False
    blendtintcoloroverbase: float = 0.0
    translucent: bool = False
    alphatest: bool = False
    alphatestreference: float = 0.5
    # phong
    phong: bool = False
    phongboost: float = 1.0
    phongexponent: Optional[float] = None       # None -> texture / default 5
    phongexponentfactor: Optional[float] = None
    phongfresnelranges: tuple = (0.0, 0.5, 1.0)
    phongtint: Optional[tuple] = None           # gamma-space colour
    phongalbedotint: bool = False
    phongalbedoboost: float = 1.0
    basemapalphaphongmask: bool = False
    invertphongmask: bool = False
    # diffuse
    halflambert: bool = False
    # rim
    rimlight: bool = False
    rimlightexponent: float = 4.0
    rimlightboost: float = 1.0
    rimmask: bool = False
    # envmap
    envmap: bool = False
    envmaptint: tuple = (1.0, 1.0, 1.0)         # gamma-space colour
    envmapfresnel: float = 0.0
    basealphaenvmapmask: bool = False
    normalmapalphaenvmapmask: bool = False
    # self illum
    selfillum: bool = False
    selfillumtint: tuple = (1.0, 1.0, 1.0)      # gamma-space colour
    selfillummaskscale: float = 1.0
    # detail
    detailscale: float = 4.0
    detailblendmode: int = 0
    detailblendfactor: float = 1.0
    detailtint: tuple = (1.0, 1.0, 1.0)


@dataclass
class ConvertSettings:
    normal_input: str = "AUTO"          # AUTO | GL | DX
    spec_strength: float = 1.0          # multiplier on matched F0
    max_f0: float = 0.16                # cap on F0 (~diamond, the most reflective non-metal)
    matte_unmasked: bool = True         # push roughness -> 1 where there is no phong
    roughness_bias: float = 0.0         # added to roughness after conversion
    env_roughness: float = 0.12         # roughness used for $envmap reflections
    env_strength: float = 1.0
    env_to_metallic: float = 0.0        # 0 keeps envmap as dielectric F0
    rim_strength: float = 1.0
    rim_use_rimexponent: bool = False   # Source uses a fixed ^4 fresnel for the ambient rim
    lightwarp_brightness: bool = False
    lightwarp_hue: float = 0.5
    lightwarp_srgb: bool = False        # Source samples lightwarps without sRGB decode
    selfillum_darken_base: bool = True


# --------------------------------------------------------------------------
# conversion constants
# --------------------------------------------------------------------------

def phong_to_roughness(n):
    """Source's reflect-vector Phong exponent n -> Blender/glTF perceptual roughness.
    Phong (R.L)^n ~ Blinn (N.H)^(4n);  Beckmann/GGX alpha^2 = 2/(N_blinn+2); r = sqrt(alpha)."""
    n = np.maximum(np.asarray(n, dtype=np.float32), 0.0)
    alpha2 = 2.0 / (4.0 * n + 2.0)
    return np.power(alpha2, 0.25).astype(np.float32)


def peak_calibration(n):
    """F0 needed (per unit of Source spec intensity) so a GGX highlight peak equals a
    Source Phong peak relative to Lambert diffuse: F0 = 4*alpha^2*S (verified in Cycles)."""
    n = np.maximum(np.asarray(n, dtype=np.float32), 0.0)
    return (8.0 / (4.0 * n + 2.0)).astype(np.float32)


def source_fresnel(c, ranges):
    """Source's 3-point fresnel: (min, mid, max) at N.V = 1, (1-N.V)^2=0.5, N.V = 0."""
    lo, mid, hi = ranges
    f = np.clip(1.0 - c, 0, 1) ** 2 - 0.5
    return np.where(f >= 0, mid + (hi - mid) * 2 * f, mid + (mid - lo) * 2 * f)


def fresnel_fit_coeffs(ranges, samples=512):
    """Least-squares Schlick fit coefficients.
    Minimises  int w(c) [ Schlick(F0,c) - k*Fsrc(c) ]^2 dc   with w = 2c (projected solid angle)
    =>  F0 = (k*A - B) / C."""
    c = (np.arange(samples) + 0.5) / samples
    w = 2 * c / samples
    g = (1 - c) ** 5
    fs = source_fresnel(c, ranges)
    A = float(np.sum(w * (1 - g) * fs))
    B = float(np.sum(w * (1 - g) * g))
    C = float(np.sum(w * (1 - g) ** 2))
    avg = float(np.sum(w * fs))
    return A, B, C, avg


# Sheen calibration measured in Cycles (Blender 5.2, Principled sheen, uniform white
# environment, ortho sphere).  For a rim falloff (1-N.V)^e the best-fit sheen roughness
# and sheen weight per unit of rim intensity:
_SHEEN_TABLE = [  # e, roughness, weight per unit
    (1.0, 0.65, 1.642), (1.5, 0.55, 1.609), (2.0, 0.50, 1.485), (3.0, 0.40, 1.428),
    (4.0, 0.35, 1.293), (5.0, 0.35, 1.008), (6.0, 0.30, 1.006), (8.0, 0.25, 0.878),
    (10.0, 0.25, 0.614), (12.0, 0.20, 0.607), (16.0, 0.20, 0.328), (24.0, 0.15, 0.154),
    (32.0, 0.10, 0.080),
]
# Source's ambient rim is masked by saturate(N.z) (upward-facing normals only). Around a
# silhouette ring seen from the side the mean of saturate(N.z) is 1/pi.
_RIM_UP_FACTOR = 1.0 / math.pi


def sheen_from_rim(exponent):
    e = float(np.clip(exponent, _SHEEN_TABLE[0][0], _SHEEN_TABLE[-1][0]))
    xs = [t[0] for t in _SHEEN_TABLE]
    rough = float(np.interp(e, xs, [t[1] for t in _SHEEN_TABLE]))
    k = float(np.interp(e, xs, [t[2] for t in _SHEEN_TABLE]))
    return rough, k


# --------------------------------------------------------------------------
# lightwarp
# --------------------------------------------------------------------------

def lightwarp_analysis(ramp_encoded: Optional[np.ndarray], halflambert: bool, srgb: bool):
    """Returns dict with the diffuse response curve and albedo calibration.
    ramp_encoded: (N,3) raw texel values or None."""
    c = np.linspace(-1, 1, 257, dtype=np.float32)
    if ramp_encoded is not None:
        ramp = srgb_to_linear(ramp_encoded) if srgb else ramp_encoded.astype(np.float32)
        x = np.clip(c * 0.5 + 0.5, 0, 1)
        resp = 2.0 * sample_ramp(ramp, x)                     # Source: 2 * tex1D(warp, halfLambert)
    elif halflambert:
        x = np.clip(c * 0.5 + 0.5, 0, 1)
        resp = np.repeat((x * x)[:, None], 3, 1)
    else:
        resp = np.repeat(np.clip(c, 0, 1)[:, None], 3, 1)
    lit = c >= 0
    lam = np.clip(c, 0, 1)
    peak = resp[-1]                                           # response at N.L = 1
    brightness = float(np.clip(luminance(peak), 0.25, 4.0))
    # hue of the lit hemisphere, weighted by Lambert so the terminator counts less
    wl = lam[lit][:, None]
    mean_rgb = (resp[lit] * wl).sum(0) / max(wl.sum(), 1e-6)
    hue = mean_rgb / max(float(luminance(mean_rgb)), 1e-6)
    shadow = resp[c <= 0].mean(0)                             # what the unlit side gets
    terminator = resp[np.argmin(np.abs(c))]
    return dict(
        curve_ndotl=c.tolist(),
        response=resp.tolist(),
        brightness=brightness,
        hue=hue.tolist(),
        shadow_color=shadow.tolist(),
        terminator_color=terminator.tolist(),
        peak_color=peak.tolist(),
        has_warp=ramp_encoded is not None,
    )


# --------------------------------------------------------------------------
# normal map convention detection
# --------------------------------------------------------------------------

def detect_normal_convention(nrm_encoded: np.ndarray, tile=32) -> str:
    """A true tangent-space normal map comes from a height field, so its slope field is
    curl-free.  Flipping green breaks that, so whichever convention gives the lower curl
    per tile wins.  Expects Blender pixel order (row 0 = bottom, +V up)."""
    n = nrm_encoded[..., :3] * 2 - 1
    nz = np.clip(n[..., 2], 0.3, None)
    hx = -n[..., 0] / nz
    votes = [0, 0]
    h, w = n.shape[:2]
    for k, sy in enumerate((1.0, -1.0)):
        hy = -sy * n[..., 1] / nz
        d1 = (np.roll(hx, -1, 0) - np.roll(hx, 1, 0)) * 0.5
        d2 = (np.roll(hy, -1, 1) - np.roll(hy, 1, 1)) * 0.5
        curl = (d1 - d2) ** 2
        if k == 0:
            curl_gl = curl
        else:
            curl_dx = curl
    slope = np.hypot(hx, n[..., 1] / nz)
    for i in range(1, h // tile - 1):
        for j in range(1, w // tile - 1):
            s = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            m = slope[s] < 3
            if not m.any() or slope[s][m].mean() < 0.05:
                continue
            votes[0 if curl_gl[s][m].mean() < curl_dx[s][m].mean() else 1] += 1
    if votes[0] == votes[1]:
        return "GL"
    return "GL" if votes[0] > votes[1] else "DX"


# --------------------------------------------------------------------------
# main conversion
# --------------------------------------------------------------------------

@dataclass
class Textures:
    base: Optional[np.ndarray] = None        # RGBA encoded (sRGB)
    bump: Optional[np.ndarray] = None        # RGBA raw
    exponent: Optional[np.ndarray] = None    # RGBA raw
    envmask: Optional[np.ndarray] = None     # RGB raw
    selfillummask: Optional[np.ndarray] = None
    detail: Optional[np.ndarray] = None      # RGBA encoded
    lightwarp: Optional[np.ndarray] = None   # (N,3) raw


def convert(tex: Textures, p: VLGParams, s: ConvertSettings, size=None, log=print):
    """Returns (maps: dict[str, ndarray], info: dict)."""
    if tex.base is None:
        raise ValueError("A $basetexture image is required")
    if size is None:
        h, w = tex.base.shape[:2]
    else:
        w, h = size
    info = {"warnings": []}

    base = resample(tex.base, h, w)
    if base.shape[2] == 3:
        base = np.concatenate([base, np.ones_like(base[..., :1])], 2)
    base_a = base[..., 3]
    base_lin = srgb_to_linear(base[..., :3])

    # ---------------------------------------------------------------- albedo
    tint = gamma_to_linear(p.color2)[None, None, :]
    if p.blendtintbybasealpha:
        tinted = lerp(base_lin * tint, np.broadcast_to(tint, base_lin.shape), float(p.blendtintcoloroverbase))
        albedo = lerp(base_lin, tinted, base_a[..., None])
    else:
        albedo = base_lin * tint
    emission = np.zeros_like(albedo)

    if tex.detail is not None and p.detailblendfactor > 0:
        det = resample(tex.detail, h, w, tile=float(p.detailscale))
        if det.shape[2] == 3:
            det = np.concatenate([det, np.ones_like(det[..., :1])], 2)
        dtint = gamma_to_linear(p.detailtint)[None, None, :]
        f = float(p.detailblendfactor)
        mode = int(p.detailblendmode)
        dlin = srgb_to_linear(det[..., :3]) * dtint
        if mode == 0:      # mod2x (detail is read raw, 0.5 = no change)
            albedo = albedo * lerp(1.0, 2.0 * det[..., :3] * dtint, f)
        elif mode == 1:    # additive
            albedo = albedo + f * dlin
        elif mode == 2:    # translucent detail
            albedo = lerp(albedo, dlin, (f * det[..., 3])[..., None])
        elif mode == 3:    # blend factor fade
            albedo = lerp(albedo, dlin, f)
        elif mode == 4:    # translucent base
            albedo = lerp(dlin, albedo, base_a[..., None]) if f > 0 else albedo
        elif mode in (5, 6):  # additive self-illum / unlit additive threshold fade
            emission = emission + f * dlin
        elif mode == 8:    # multiply
            albedo = albedo * lerp(1.0, dlin, f)
        else:
            info["warnings"].append(f"$detailblendmode {mode} is not converted")
    elif tex.detail is not None:
        info["warnings"].append("$detail present but $detailblendfactor is 0 (runtime proxy effect) - skipped")

    # ---------------------------------------------------------------- normal
    if tex.bump is not None:
        bump = resample(tex.bump, h, w)
        has_bump_alpha = bump.shape[2] >= 4
        conv = s.normal_input
        if conv == "AUTO":
            conv = detect_normal_convention(bump)
            info["normal_detected"] = conv
        n = bump[..., :3] * 2 - 1
        if conv == "DX":
            n[..., 1] *= -1
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
        normal_gl = n * 0.5 + 0.5
        normal_dx = normal_gl.copy(); normal_dx[..., 1] = 1 - normal_dx[..., 1]
        bump_a = bump[..., 3] if has_bump_alpha else np.ones((h, w), np.float32)
    else:
        normal_gl = np.broadcast_to(np.array([0.5, 0.5, 1.0], np.float32), (h, w, 3)).copy()
        normal_dx = normal_gl.copy()
        bump_a = np.ones((h, w), np.float32)

    # ---------------------------------------------------------------- phong
    expo = resample(tex.exponent, h, w) if tex.exponent is not None else None
    if expo is not None and expo.shape[2] == 3:
        expo = np.concatenate([expo, np.ones_like(expo[..., :1])], 2)

    if p.phongexponent is not None and p.phongexponent > 0:
        n_exp = np.full((h, w), float(p.phongexponent), np.float32)
        if expo is not None:
            info["warnings"].append("$phongexponent overrides $phongexponenttexture red channel (Source behaviour)")
    elif expo is not None:
        factor = float(p.phongexponentfactor) if p.phongexponentfactor else 149.0
        n_exp = 1.0 + expo[..., 0] * factor
    else:
        n_exp = np.full((h, w), 5.0, np.float32)

    if p.basemapalphaphongmask:
        mask = base_a.copy()
    else:
        mask = bump_a.copy()
    if p.invertphongmask:
        mask = 1 - mask

    if p.phongalbedotint and expo is not None:
        spec_tint = lerp(np.ones_like(albedo), albedo * float(p.phongalbedoboost), expo[..., 1:2])
    elif p.phongtint is not None:
        spec_tint = np.broadcast_to(gamma_to_linear(p.phongtint)[None, None, :], albedo.shape).copy()
    else:
        spec_tint = np.ones_like(albedo)

    rough = phong_to_roughness(n_exp)
    if p.phong:
        A, B, C, avg = fresnel_fit_coeffs(p.phongfresnelranges)
        k = mask * float(p.phongboost) * peak_calibration(n_exp)
        f0 = np.clip((k * A - B) / C, 0.0, None) * float(s.spec_strength)
        info["fresnel_fit"] = dict(A=A, B=B, C=C, avg_source_fresnel=avg)
    else:
        f0 = np.zeros((h, w), np.float32)
        info["warnings"].append("$phong is off - surface converted as matte (F0 = 0)")
    f0_rgb = f0[..., None] * spec_tint

    if s.matte_unmasked:
        rough = lerp(np.ones_like(rough), rough, smoothstep(0.0, 0.04, f0))

    # ---------------------------------------------------------------- envmap
    metallic = np.zeros((h, w), np.float32)
    if p.envmap:
        if tex.envmask is not None:
            em = resample(tex.envmask, h, w)[..., :3]
        elif p.basealphaenvmapmask:
            em = np.repeat(base_a[..., None], 3, 2)
        elif p.normalmapalphaenvmapmask:
            em = np.repeat(bump_a[..., None], 3, 2)
        else:
            em = np.ones((h, w, 3), np.float32)
        E = em * gamma_to_linear(p.envmaptint)[None, None, :] * float(s.env_strength)
        if p.envmapfresnel > 0:
            # $envmapfresnel reuses $phongfresnelranges; fold in its average so the
            # Schlick term from the renderer is not counted twice
            _, _, _, avg = fresnel_fit_coeffs(p.phongfresnelranges)
            E = E * lerp(1.0, avg, float(p.envmapfresnel))
        e_l = luminance(E); f_l = luminance(f0_rgb)
        wgt = e_l / np.maximum(e_l + f_l, 1e-6)
        rough = lerp(rough, np.full_like(rough, s.env_roughness), wgt * (e_l > 1e-4))
        f0_rgb = np.maximum(f0_rgb, E)
        if s.env_to_metallic > 0:
            metallic = np.clip(e_l * float(s.env_to_metallic), 0, 1).astype(np.float32)

    f0_rgb = np.clip(f0_rgb, 0.0, float(s.max_f0))
    rough = np.clip(rough + float(s.roughness_bias), 0.02, 1.0)

    # ---------------------------------------------------------------- rim -> sheen
    sheen = np.zeros((h, w), np.float32)
    sheen_rough = 0.5
    if p.rimlight:
        e = p.rimlightexponent if s.rim_use_rimexponent else 4.0
        sheen_rough, kk = sheen_from_rim(e)
        rm = expo[..., 3] if (p.rimmask and expo is not None) else 1.0
        sheen = np.clip(kk * _RIM_UP_FACTOR * float(p.rimlightboost) * rm * float(s.rim_strength), 0, 1)
        sheen = np.broadcast_to(sheen, (h, w)).astype(np.float32)

    # ---------------------------------------------------------------- self illum
    if p.selfillum:
        if tex.selfillummask is not None:
            si = resample(tex.selfillummask, h, w)[..., 0]
        else:
            si = base_a
        si = np.clip(si * float(p.selfillummaskscale), 0, 1)
        emission = emission + albedo * gamma_to_linear(p.selfillumtint)[None, None, :] * si[..., None]
        if s.selfillum_darken_base:
            albedo = albedo * (1 - si[..., None])

    # ---------------------------------------------------------------- lightwarp calibration
    lw = lightwarp_analysis(tex.lightwarp, bool(p.halflambert or tex.lightwarp is not None), s.lightwarp_srgb)
    calib = np.ones(3, np.float32)
    if s.lightwarp_brightness:
        calib *= lw["brightness"]
    calib *= lerp(np.ones(3, np.float32), np.array(lw["hue"], np.float32), float(s.lightwarp_hue))
    albedo = np.clip(albedo * calib[None, None, :], 0, 1)
    info["albedo_calibration"] = calib.tolist()
    info["lightwarp"] = lw

    # ---------------------------------------------------------------- alpha
    alpha_used_for_other = p.blendtintbybasealpha or p.basemapalphaphongmask or \
        (p.selfillum and tex.selfillummask is None) or (p.envmap and p.basealphaenvmapmask)
    if (p.translucent or p.alphatest) and not alpha_used_for_other:
        alpha = base_a
        if p.alphatest:
            alpha = (alpha >= float(p.alphatestreference)).astype(np.float32)
        info["alpha_mode"] = "CLIP" if p.alphatest else "BLEND"
    else:
        alpha = np.ones((h, w), np.float32)
        info["alpha_mode"] = "OPAQUE"
        if p.translucent or p.alphatest:
            info["warnings"].append("$translucent/$alphatest ignored: base alpha is already used as a mask")

    # ---------------------------------------------------------------- outputs
    f0_l = luminance(f0_rgb)
    spec_tint_norm = f0_rgb / np.maximum(f0_l[..., None], 1e-6)
    spec_tint_norm = np.where(f0_l[..., None] > 1e-5, spec_tint_norm, 1.0)
    level = f0_l / 0.08
    info["max_f0"] = float(f0_l.max())
    info["specular_level_max"] = float(level.max())
    info["specular_level_clipped_fraction"] = float((level > 1.0).mean())
    info["sheen_roughness"] = sheen_rough
    info["roughness_mean"] = float(rough.mean())

    maps = {
        "basecolor": np.concatenate([linear_to_srgb(albedo), alpha[..., None]], 2),   # sRGB
        "normal_gl": normal_gl.astype(np.float32),
        "normal_dx": normal_dx.astype(np.float32),
        "roughness": rough.astype(np.float32),
        "metallic": metallic,
        "specular": np.clip(level, 0, 1).astype(np.float32),                       # 0.5 = 4% F0
        "specular_color": linear_to_srgb(np.clip(f0_rgb, 0, 1)),                    # F0 (spec/gloss)
        "specular_tint": linear_to_srgb(np.clip(spec_tint_norm, 0, 1)),
        "emission": linear_to_srgb(emission),
        "sheen": sheen,
        "opacity": alpha.astype(np.float32),
        # linear helpers (not written unless requested)
        "_albedo_linear": albedo, "_f0_linear": f0_rgb,
    }
    return maps, info


def params_to_dict(p: VLGParams):
    return asdict(p)
