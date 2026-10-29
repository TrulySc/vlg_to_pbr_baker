# SPDX-License-Identifier: GPL-3.0-or-later
"""
Minimal readers for Source engine files, no bpy dependency:
  * VMT / KeyValues text
  * VTF textures (uncompressed formats, DXT1/3/5, RGBA16F)
  * VPK v1/v2 directory archives
  * Steam / TF2 install discovery
"""

from __future__ import annotations

import os
import re
import struct
import sys
from typing import Optional

import numpy as np

# =============================================================================
# KeyValues / VMT
# =============================================================================

_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|(\{)|(\})|([^\s{}"]+)')


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        # remove // comments that are not inside quotes
        q = False
        for i, ch in enumerate(line):
            if ch == '"':
                q = not q
            elif ch == '/' and not q and line[i:i + 2] == '//':
                line = line[:i]
                break
        out.append(line)
    return "\n".join(out)


def parse_keyvalues(text: str) -> dict:
    """Parse KeyValues into nested dicts with lower-cased keys.
    Returns {"shader": name, "params": {...}} for a VMT."""
    toks = []
    for m in _TOKEN.finditer(_strip_comments(text)):
        if m.group(1) is not None:
            toks.append(("s", m.group(1)))
        elif m.group(2):
            toks.append(("{", None))
        elif m.group(3):
            toks.append(("}", None))
        else:
            toks.append(("s", m.group(4)))
    pos = 0

    def block():
        nonlocal pos
        d = {}
        while pos < len(toks):
            kind, val = toks[pos]
            if kind == "}":
                pos += 1
                return d
            if kind != "s":
                pos += 1
                continue
            key = val.lower()
            pos += 1
            if pos >= len(toks):
                break
            k2, v2 = toks[pos]
            if k2 == "{":
                pos += 1
                d[key] = block()
            elif k2 == "s":
                # a bare value could be "{ 128 86 61 }" written unquoted -> handled by tokens
                d[key] = v2
                pos += 1
            else:
                pos += 1
        return d

    # first token is the shader name
    shader = None
    while pos < len(toks):
        kind, val = toks[pos]
        if kind == "s":
            shader = val.lower()
            pos += 1
            if pos < len(toks) and toks[pos][0] == "{":
                pos += 1
                return {"shader": shader, "params": block()}
        else:
            pos += 1
    return {"shader": shader, "params": {}}


_DX9_BLOCKS = (">=dx90", ">dx90", "dx90", ">=dx90_20b", "dx9", ">=dx95", "hdr_dx9", ">=hdr_dx9", "srgb")


def flatten_vmt_params(params: dict) -> dict:
    """Merge DX9+ fallback blocks into the top level, drop proxies."""
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            continue
        flat[k.lower()] = v
    for k, v in params.items():
        if isinstance(v, dict) and k.lower() in _DX9_BLOCKS:
            for k2, v2 in v.items():
                if not isinstance(v2, dict):
                    flat[k2.lower()] = v2
    return flat


def parse_number(v, default=None):
    try:
        return float(str(v).strip().strip('"'))
    except (TypeError, ValueError):
        vec = parse_vector(v)
        return vec[0] if vec else default


def parse_vector(v, n=3):
    """'[.25 .5 1]' -> floats ; '{128 86 61}' -> /255 ; '0.5' -> broadcast."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        vals = [float(x) for x in v]
    else:
        s = str(v).strip()
        scale = 1.0
        if s.startswith("{"):
            scale = 1.0 / 255.0
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", s)
        if not nums:
            return None
        vals = [float(x) * scale for x in nums]
    if len(vals) == 1:
        vals = vals * n
    return tuple(vals[:n])


def parse_bool(v, default=False):
    if v is None:
        return default
    f = parse_number(v, None)
    if f is None:
        return str(v).strip().lower() in ("true", "yes")
    return f != 0.0


# =============================================================================
# VTF
# =============================================================================

VTF_FORMATS = {
    0: "RGBA8888", 1: "ABGR8888", 2: "RGB888", 3: "BGR888", 4: "RGB565", 5: "I8", 6: "IA88",
    7: "P8", 8: "A8", 9: "RGB888_BLUESCREEN", 10: "BGR888_BLUESCREEN", 11: "ARGB8888",
    12: "BGRA8888", 13: "DXT1", 14: "DXT3", 15: "DXT5", 16: "BGRX8888", 17: "BGR565",
    18: "BGRX5551", 19: "BGRA4444", 20: "DXT1_ONEBITALPHA", 21: "BGRA5551", 22: "UV88",
    23: "UVWQ8888", 24: "RGBA16161616F", 25: "RGBA16161616", 26: "UVLX8888",
}
_BPP = {0: 4, 1: 4, 2: 3, 3: 3, 4: 2, 5: 1, 6: 2, 7: 1, 8: 1, 9: 3, 10: 3, 11: 4, 12: 4,
        16: 4, 17: 2, 18: 2, 19: 2, 21: 2, 22: 2, 23: 4, 24: 8, 25: 8, 26: 4}
_DXT = {13: 8, 14: 16, 15: 16, 20: 8}


def _img_size(fmt, w, h):
    if fmt in _DXT:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * _DXT[fmt]
    if fmt in _BPP:
        return w * h * _BPP[fmt]
    raise ValueError(f"Unsupported VTF format {VTF_FORMATS.get(fmt, fmt)}")


def _decode_565(v):
    r = ((v >> 11) & 31) / 31.0
    g = ((v >> 5) & 63) / 63.0
    b = (v & 31) / 31.0
    return np.stack([r, g, b], -1).astype(np.float32)


def _decode_dxt(data, w, h, fmt):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    bs = _DXT[fmt]
    blocks = np.frombuffer(data, np.uint8, bw * bh * bs).reshape(bh * bw, bs)
    if fmt in (13, 20):
        cblk = blocks
        alpha = None
    else:
        cblk = blocks[:, 8:]
        ablk = blocks[:, :8]
    c0 = cblk[:, 0].astype(np.uint32) | (cblk[:, 1].astype(np.uint32) << 8)
    c1 = cblk[:, 2].astype(np.uint32) | (cblk[:, 3].astype(np.uint32) << 8)
    idx = (cblk[:, 4].astype(np.uint32) | (cblk[:, 5].astype(np.uint32) << 8) |
           (cblk[:, 6].astype(np.uint32) << 16) | (cblk[:, 7].astype(np.uint32) << 24))
    p0 = _decode_565(c0); p1 = _decode_565(c1)
    four = (c0 > c1) | (fmt in (14, 15))
    pal = np.zeros((len(c0), 4, 4), np.float32)
    pal[:, 0, :3] = p0; pal[:, 1, :3] = p1
    pal[:, 2, :3] = np.where(four[:, None], (2 * p0 + p1) / 3, (p0 + p1) / 2)
    pal[:, 3, :3] = np.where(four[:, None], (p0 + 2 * p1) / 3, 0)
    pal[:, :, 3] = 1.0
    if fmt in (13, 20):
        pal[:, 3, 3] = np.where(four, 1.0, 0.0 if fmt == 20 else 1.0)
    shifts = np.arange(16, dtype=np.uint32) * 2
    ci = (idx[:, None] >> shifts[None, :]) & 3
    px = np.take_along_axis(pal, ci[:, :, None].astype(np.int64).repeat(4, 2), 1)  # (n,16,4)
    if fmt == 14:
        a = ablk.astype(np.uint16)
        nib = np.stack([(a & 15), (a >> 4)], -1).reshape(-1, 16) / 15.0
        px[:, :, 3] = nib
    elif fmt == 15:
        a0 = ablk[:, 0].astype(np.float32) / 255; a1 = ablk[:, 1].astype(np.float32) / 255
        bits = np.zeros(len(a0), np.uint64)
        for i in range(6):
            bits |= ablk[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
        ai = ((bits[:, None] >> (np.arange(16, dtype=np.uint64) * np.uint64(3))[None, :]) & np.uint64(7)).astype(np.int64)
        ap = np.zeros((len(a0), 8), np.float32)
        ap[:, 0] = a0; ap[:, 1] = a1
        eight = a0 > a1
        for i in range(1, 7):
            ap[:, 1 + i] = np.where(eight, ((7 - i) * a0 + i * a1) / 7, 0)
        for i in range(1, 5):
            ap[:, 1 + i] = np.where(eight, ap[:, 1 + i], ((5 - i) * a0 + i * a1) / 5)
        ap[:, 6] = np.where(eight, ap[:, 6], 0.0)
        ap[:, 7] = np.where(eight, ap[:, 7], 1.0)
        px[:, :, 3] = np.take_along_axis(ap, ai, 1)
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w]


def _decode_raw(data, w, h, fmt):
    n = w * h
    if fmt == 24:
        a = np.frombuffer(data, np.float16, n * 4).reshape(h, w, 4).astype(np.float32)
        return a
    if fmt == 25:
        return np.frombuffer(data, np.uint16, n * 4).reshape(h, w, 4).astype(np.float32) / 65535.0
    if fmt in (4, 17):
        v = np.frombuffer(data, np.uint16, n).astype(np.uint32)
        rgb = _decode_565(v).reshape(h, w, 3)
        if fmt == 17:
            rgb = rgb[..., ::-1]
        return np.concatenate([rgb, np.ones((h, w, 1), np.float32)], 2)
    if fmt in (18, 21):
        v = np.frombuffer(data, np.uint16, n).astype(np.uint32)
        b = (v & 31) / 31; g = ((v >> 5) & 31) / 31; r = ((v >> 10) & 31) / 31
        a = ((v >> 15) & 1).astype(np.float32) if fmt == 21 else np.ones(n, np.float32)
        return np.stack([r, g, b, a], -1).reshape(h, w, 4).astype(np.float32)
    if fmt == 19:
        v = np.frombuffer(data, np.uint16, n).astype(np.uint32)
        b = (v & 15) / 15; g = ((v >> 4) & 15) / 15; r = ((v >> 8) & 15) / 15; a = ((v >> 12) & 15) / 15
        return np.stack([r, g, b, a], -1).reshape(h, w, 4).astype(np.float32)
    bpp = _BPP[fmt]
    a = np.frombuffer(data, np.uint8, n * bpp).reshape(h, w, bpp).astype(np.float32) / 255.0
    one = np.ones((h, w, 1), np.float32)
    if fmt in (0, 23, 26):
        return a
    if fmt == 1:
        return a[..., [3, 2, 1, 0]]
    if fmt in (2, 9):
        return np.concatenate([a, one], 2)
    if fmt in (3, 10):
        return np.concatenate([a[..., ::-1], one], 2)
    if fmt == 5 or fmt == 7:
        return np.concatenate([a.repeat(3, 2), one], 2)
    if fmt == 6:
        return np.concatenate([a[..., :1].repeat(3, 2), a[..., 1:2]], 2)
    if fmt == 8:
        return np.concatenate([np.zeros((h, w, 3), np.float32), a], 2)
    if fmt == 11:
        return a[..., [1, 2, 3, 0]]
    if fmt == 12:
        return a[..., [2, 1, 0, 3]]
    if fmt == 16:
        return np.concatenate([a[..., [2, 1, 0]], one], 2)
    if fmt == 22:
        return np.concatenate([a, np.zeros((h, w, 1), np.float32), one], 2)
    raise ValueError(f"Unsupported VTF format {fmt}")


def read_vtf(data: bytes, frame: int = 0):
    """Decode the largest mip of a VTF. Returns float32 RGBA (H, W, 4), top row first,
    plus a header dict."""
    if data[:4] != b"VTF\0":
        raise ValueError("Not a VTF file")
    vmaj, vmin, hsize = struct.unpack_from("<3I", data, 4)
    w, h, flags, frames, first = struct.unpack_from("<HHIHH", data, 16)
    fmt, = struct.unpack_from("<i", data, 52)
    mips = data[56]
    lofmt, = struct.unpack_from("<i", data, 57)
    low_w, low_h = data[61], data[62]
    depth = 1
    if (vmaj, vmin) >= (7, 2):
        depth = max(1, struct.unpack_from("<H", data, 63)[0])
    faces = 1
    if flags & 0x4000:  # TEXTUREFLAGS_ENVMAP
        faces = 6 if (vmaj, vmin) >= (7, 5) or first != 0xFFFF else 7
    hi_offset = None
    if (vmaj, vmin) >= (7, 3):
        nres, = struct.unpack_from("<I", data, 68)
        for i in range(nres):
            tag, rflags, off = struct.unpack_from("<3sBI", data, 80 + i * 8)
            if tag == b"\x30\x00\x00":
                hi_offset = off
    if hi_offset is None:
        low_size = _img_size(lofmt, low_w, low_h) if (lofmt >= 0 and low_w and low_h) else 0
        hi_offset = hsize + low_size
    # mips are stored smallest -> largest; skip to mip 0
    off = hi_offset
    for m in range(mips - 1, 0, -1):
        mw, mh = max(1, w >> m), max(1, h >> m)
        md = max(1, depth >> m)
        off += _img_size(fmt, mw, mh) * frames * faces * md
    size0 = _img_size(fmt, w, h)
    off += size0 * faces * depth * min(frame, frames - 1)
    chunk = data[off:off + size0]
    if len(chunk) < size0:
        raise ValueError("VTF truncated")
    img = _decode_dxt(chunk, w, h, fmt) if fmt in _DXT else _decode_raw(chunk, w, h, fmt)
    header = dict(version=f"{vmaj}.{vmin}", width=w, height=h, format=VTF_FORMATS.get(fmt, fmt),
                  flags=flags, frames=frames, mips=mips)
    return img.astype(np.float32), header


# =============================================================================
# VPK
# =============================================================================

class VPK:
    """Read-only access to a *_dir.vpk archive (v1 and v2)."""

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.entries = {}
        with open(dir_path, "rb") as f:
            data = f.read()
        sig, ver, tree = struct.unpack_from("<3I", data, 0)
        if sig != 0x55AA1234:
            raise ValueError("Not a VPK")
        hdr = 12 if ver == 1 else 28
        self.data_start = hdr + tree
        self._dir_data = data
        pos = hdr

        def cstr():
            nonlocal pos
            end = data.index(b"\0", pos)
            s = data[pos:end].decode("utf-8", "replace")
            pos = end + 1
            return s

        while True:
            ext = cstr()
            if not ext:
                break
            while True:
                path = cstr()
                if not path:
                    break
                while True:
                    name = cstr()
                    if not name:
                        break
                    crc, pre, arch, eoff, elen, term = struct.unpack_from("<IHHIIH", data, pos)
                    pos += 18
                    preload = data[pos:pos + pre]
                    pos += pre
                    full = (f"{path.strip()}/" if path.strip() else "") + name + ("." + ext if ext.strip() else "")
                    self.entries[full.lower()] = (arch, eoff, elen, preload)

    def read(self, path: str) -> Optional[bytes]:
        e = self.entries.get(path.lower().replace("\\", "/"))
        if e is None:
            return None
        arch, off, ln, pre = e
        if ln == 0:
            return pre
        if arch == 0x7FFF:
            body = self._dir_data[self.data_start + off:self.data_start + off + ln]
        else:
            base = self.dir_path[:-len("_dir.vpk")]
            with open(f"{base}_{arch:03d}.vpk", "rb") as f:
                f.seek(off)
                body = f.read(ln)
        return pre + body


# =============================================================================
# Steam / TF2 discovery
# =============================================================================

def _steam_roots():
    roots = []
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        try:
            import winreg
            for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                              (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        for val in ("SteamPath", "InstallPath"):
                            try:
                                roots.append(winreg.QueryValueEx(k, val)[0])
                            except OSError:
                                pass
                except OSError:
                    pass
        except ImportError:
            pass
        roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]
    elif sys.platform == "darwin":
        roots.append(os.path.join(home, "Library/Application Support/Steam"))
    else:
        roots += [os.path.join(home, ".local/share/Steam"), os.path.join(home, ".steam/steam"),
                  os.path.join(home, ".steam/root"),
                  os.path.join(home, ".var/app/com.valvesoftware.Steam/.local/share/Steam"),
                  os.path.join(home, "snap/steam/common/.local/share/Steam")]
    out = []
    for r in roots:
        r = os.path.normpath(r)
        if os.path.isdir(r) and r not in out:
            out.append(r)
    return out


def find_steam_libraries():
    libs = []
    for root in _steam_roots():
        libs.append(root)
        vdf = os.path.join(root, "steamapps", "libraryfolders.vdf")
        if os.path.isfile(vdf):
            try:
                with open(vdf, encoding="utf-8", errors="replace") as f:
                    for m in re.finditer(r'"path"\s+"([^"]+)"', f.read()):
                        libs.append(os.path.normpath(m.group(1).replace("\\\\", "\\")))
            except OSError:
                pass
    seen, out = set(), []
    for l in libs:
        if l not in seen and os.path.isdir(l):
            seen.add(l); out.append(l)
    return out


def find_tf2():
    for lib in find_steam_libraries():
        p = os.path.join(lib, "steamapps", "common", "Team Fortress 2")
        if os.path.isdir(os.path.join(p, "tf")):
            return p
    return ""


class SourceFS:
    """Looks a game-relative path up in loose files, tf/custom and the game VPKs."""

    VPK_NAMES = ("tf/tf2_textures_dir.vpk", "tf/tf2_misc_dir.vpk",
                 "hl2/hl2_textures_dir.vpk", "hl2/hl2_misc_dir.vpk")

    def __init__(self, game_root: str, extra_dirs=()):
        self.roots = []
        self.vpks = []
        game_root = os.path.normpath(game_root) if game_root else ""
        if game_root and os.path.basename(game_root).lower() == "tf":
            game_root = os.path.dirname(game_root)
        for d in extra_dirs:
            if d and os.path.isdir(d):
                self.roots.append(d)
        if game_root:
            tf = os.path.join(game_root, "tf")
            self.roots.append(tf)
            custom = os.path.join(tf, "custom")
            if os.path.isdir(custom):
                for c in sorted(os.listdir(custom)):
                    full = os.path.join(custom, c)
                    if os.path.isdir(full):
                        self.roots.append(full)
                    elif c.lower().endswith("_dir.vpk"):
                        self._add_vpk(full)
            self.roots.append(os.path.join(game_root, "hl2"))
            for v in self.VPK_NAMES:
                self._add_vpk(os.path.join(game_root, v))

    def _add_vpk(self, p):
        if os.path.isfile(p):
            try:
                self.vpks.append(VPK(p))
            except Exception as e:  # noqa
                print(f"[VLG→PBR] could not read {p}: {e}")

    def read(self, rel: str) -> Optional[bytes]:
        rel = rel.replace("\\", "/").lstrip("/")
        for r in self.roots:
            p = os.path.join(r, *rel.split("/"))
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    return f.read()
            # case-insensitive fallback for Linux
            d = r
            ok = True
            for part in rel.split("/"):
                if not os.path.isdir(d) and part != rel.split("/")[-1]:
                    ok = False; break
                try:
                    match = next((x for x in os.listdir(d) if x.lower() == part.lower()), None)
                except OSError:
                    match = None
                if match is None:
                    ok = False; break
                d = os.path.join(d, match)
            if ok and os.path.isfile(d):
                with open(d, "rb") as f:
                    return f.read()
        for v in self.vpks:
            b = v.read(rel)
            if b is not None:
                return b
        return None

    def read_texture(self, vmt_path: str):
        rel = vmt_path.replace("\\", "/").lower()
        if not rel.startswith("materials/"):
            rel = "materials/" + rel
        if not rel.endswith(".vtf"):
            rel += ".vtf"
        b = self.read(rel)
        if b is None:
            return None, None
        return read_vtf(b)


# =============================================================================
# .vmt files on disk
# =============================================================================


def materials_root(path: str) -> Optional[str]:
    """For a file inside a materials/ tree, the folder that contains 'materials' (the game/addon root)."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.basename(d).lower() == "materials":
            return os.path.dirname(d)
        nd = os.path.dirname(d)
        if nd == d:
            return None
        d = nd


def find_file_ci(root: str, rel: str) -> Optional[str]:
    """root/rel, matching each path part case-insensitively (Source paths are case-insensitive)."""
    if not root:
        return None
    rel = re.sub(r"/+", "/", rel.replace("\\", "/")).strip("/")
    p = os.path.join(root, *rel.split("/"))
    if os.path.isfile(p):
        return p
    d = root
    parts = rel.split("/")
    for i, part in enumerate(parts):
        try:
            match = next((x for x in os.listdir(d) if x.lower() == part.lower()), None)
        except OSError:
            return None
        if match is None:
            return None
        d = os.path.join(d, match)
    return d if os.path.isfile(d) else None


def game_rel(path: str, ext: str) -> str:
    """'models\\x\\y.vtf' / 'materials/models/x/y' → 'materials/models/x/y<ext>'."""
    rel = re.sub(r"/+", "/", str(path).replace("\\", "/")).strip().lstrip("/")
    low = rel.lower()
    for e in (".vtf", ".vmt"):
        if low.endswith(e):
            rel, low = rel[:-4], low[:-4]
    if not low.startswith("materials/"):
        rel = "materials/" + rel
    return rel + ext


def read_vmt_file(path: str, fs=None):
    """Read a .vmt from disk, following 'patch' includes (looked up next to it, then in the game).
    Returns (shader, flat params, materials root or None, notes)."""
    root = materials_root(path)
    notes = []
    with open(path, encoding="utf-8", errors="replace") as f:
        kv = parse_keyvalues(f.read())
    shader, params = kv["shader"], kv["params"]
    depth = 0
    while shader == "patch" and depth < 8:
        depth += 1
        inc = params.get("include")
        if not inc:
            notes.append("patch material without an 'include'")
            break
        rel = game_rel(inc, ".vmt")
        text = None
        local = find_file_ci(root, rel) if root else None
        if local:
            with open(local, encoding="utf-8", errors="replace") as f:
                text = f.read()
        elif fs is not None:
            b = fs.read(rel)
            if b is not None:
                text = b.decode("utf-8", errors="replace")
        if text is None:
            notes.append(f"patch include '{inc}' not found – only the patched values are used")
            merged = {}
            for blk in ("insert", "replace"):
                if isinstance(params.get(blk), dict):
                    merged.update(params[blk])
            params = merged
            shader = "vertexlitgeneric"
            break
        base = parse_keyvalues(text)
        merged = dict(base["params"])
        for blk in ("insert", "replace"):
            if isinstance(params.get(blk), dict):
                merged.update(params[blk])
        notes.append(f"patch: included {inc}")
        shader, params = base["shader"], merged
    return shader, flatten_vmt_params(params), root, notes
