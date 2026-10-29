# SPDX-License-Identifier: GPL-3.0-or-later
"""Tiny PNG writer (8/16-bit gray, gray+alpha, RGB, RGBA). Writes exactly the values
given - no colour management - so maps land on disk byte-for-byte as computed."""

import struct
import zlib

import numpy as np


def _chunk(tag, data):
    c = struct.pack(">I", len(data)) + tag + data
    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path, img, bits=8, flip_vertical=True, compress=6):
    """img: float array (H,W) or (H,W,C) in 0..1, Blender pixel order (row 0 = bottom)
    when flip_vertical=True."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if flip_vertical:
        a = a[::-1]
    h, w, c = a.shape
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    a = np.clip(a, 0, 1)
    if bits == 16:
        q = np.round(a * 65535).astype(">u2")
    else:
        q = np.round(a * 255).astype(np.uint8)
    raw = q.reshape(h, w * c * (2 if bits == 16 else 1)) if bits == 8 else q.view(np.uint8).reshape(h, -1)
    rows = np.concatenate([np.zeros((h, 1), np.uint8), raw], 1)
    data = zlib.compress(rows.tobytes(), compress)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, color_type, 0, 0, 0)))
        f.write(_chunk(b"IDAT", data))
        f.write(_chunk(b"IEND", b""))
