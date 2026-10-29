# SPDX-License-Identifier: GPL-3.0-or-later
"""
VLG → PBR Baker

One click: bakes Source Engine VertexLitGeneric materials (as imported by SourceIO, or read from a .vmt) into
ordinary Principled BSDF materials in the .blend, with packed PBR textures. Optionally also saves the texture
set to disk.
"""

bl_info = {
    "name": "VLG → PBR Baker",
    "author": "VLG PBR Baker contributors",
    "version": (2, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D View ▸ Sidebar ▸ VLG → PBR, and Properties ▸ Material ▸ VLG → PBR",
    "description": "Bake Source Engine VertexLitGeneric materials into PBR materials in one click",
    "category": "Material",
}

import json
import os
import time

import bpy
import numpy as np
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)

from . import core
from . import pngio
from . import source_formats as sf

ADDON_ID = __package__

# =============================================================================
# helpers
# =============================================================================


def log(*a):
    print("[VLG→PBR]", *a)


def addon_prefs(context=None):
    context = context or bpy.context
    a = context.preferences.addons.get(ADDON_ID)
    return a.preferences if a else None


def image_to_array(img: bpy.types.Image, role="color"):
    """Return float32 (H,W,4) of stored texel values, Blender order (row 0 = bottom).
    role 'color' -> sRGB-encoded values ; 'data' -> raw values."""
    if img is None:
        return None
    w, h = img.size
    if w == 0 or h == 0:
        try:
            img.reload()
        except Exception:  # noqa
            pass
        w, h = img.size
        if w == 0 or h == 0:
            raise RuntimeError(f"Image '{img.name}' has no pixel data (missing file?)")
    arr = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(arr)
    arr = arr.reshape(h, w, 4)
    if img.is_float and role == "color":
        cs = img.colorspace_settings.name.lower()
        if "srgb" not in cs and "non-color" not in cs and "raw" not in cs:
            arr[..., :3] = core.linear_to_srgb(arr[..., :3])
    return arr


def array_to_image(name, arr, colorspace="sRGB", pack=True):
    """arr: (H,W,4) float, Blender order. Creates/overwrites a byte image and packs it."""
    h, w = arr.shape[:2]
    img = bpy.data.images.get(name)
    if img is not None and (img.size[0] != w or img.size[1] != h or img.is_float):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=False)
    img.colorspace_settings.name = colorspace
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), np.float32)], 2)
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
    img.update()
    if pack:
        try:
            img.pack()
        except Exception as e:  # noqa
            log("pack failed", e)
    return img


def ramp_from_image(img: bpy.types.Image):
    a = image_to_array(img, "data")
    if a is None:
        return None
    row = a[a.shape[0] // 2, :, :3]
    return row.astype(np.float32)


def material_vmt_params(mat):
    if "vmt_parameters" not in mat:
        return None
    v = mat["vmt_parameters"]
    d = v.to_dict() if hasattr(v, "to_dict") else dict(v)
    return sf.flatten_vmt_params({k.lower(): val for k, val in d.items()})


def find_vlg_group_node(mat):
    if not mat or not mat.node_tree:
        return None
    for n in mat.node_tree.nodes:
        if n.type == "GROUP" and n.node_tree and "vertexlitgeneric" in n.node_tree.name.lower():
            return n
    return None


def linked_image(sock):
    if sock is None or not sock.is_linked:
        return None
    n = sock.links[0].from_node
    seen = 0
    while n.type == "REROUTE" and seen < 32:
        if not n.inputs[0].is_linked:
            return None
        n = n.inputs[0].links[0].from_node
        seen += 1
    return n.image if n.type == "TEX_IMAGE" else None


def linked_image_output(sock):
    """(image, output socket name) feeding a socket, through reroutes."""
    if sock is None or not sock.is_linked:
        return None, None
    link = sock.links[0]
    seen = 0
    while link.from_node.type == "REROUTE" and seen < 32:
        if not link.from_node.inputs[0].is_linked:
            return None, None
        link = link.from_node.inputs[0].links[0]
        seen += 1
    n = link.from_node
    return (n.image, link.from_socket.name) if n.type == "TEX_IMAGE" else (None, None)


def find_image_by_vmt_path(path):
    if not path:
        return None
    stem = str(path).replace("\\", "/").split("/")[-1].lower()
    if stem.endswith(".vtf"):
        stem = stem[:-4]
    for img in bpy.data.images:
        nm = img.name.lower()
        base = os.path.splitext(nm)[0]
        if (nm == stem or base == stem) and img.size[0] > 1:
            return img
    return None


def source_fs(context):
    prefs = addon_prefs(context)
    root = prefs.tf2_path if prefs else ""
    if not root:
        root = sf.find_tf2()
        if prefs and root:
            prefs.tf2_path = root
    extra = []
    if prefs and prefs.extra_content:
        extra = [p.strip() for p in prefs.extra_content.split(";") if p.strip()]
    if not root and not extra:
        return None
    return sf.SourceFS(root, extra)


def load_vtf_as_image(fs, vmt_path, role):
    img_arr, hdr = fs.read_texture(vmt_path)
    if img_arr is None:
        return None
    name = str(vmt_path).replace("\\", "/").split("/")[-1]
    arr = img_arr[::-1].copy()  # VTF is top-down
    return array_to_image(name, arr, "sRGB" if role == "color" else "Non-Color")


def abspath_dir(p):
    return os.path.normpath(bpy.path.abspath(p))


# =============================================================================
# property groups
# =============================================================================

MAP_ITEMS = [
    ("BASECOLOR", "Base Color", "Albedo (sRGB, alpha = opacity when used)", 1),
    ("NORMAL_GL", "Normal (OpenGL)", "Blender / Unity / Godot / glTF", 2),
    ("NORMAL_DX", "Normal (DirectX)", "Unreal / Source / Substance DX", 4),
    ("ROUGHNESS", "Roughness", "From $phongexponent", 8),
    ("METALLIC", "Metallic", "From envmap (0 unless Envmap → Metallic is used)", 16),
    ("SPECULAR", "Specular Level", "0.5 = 4% F0 (Blender IOR Level / Unreal Specular)", 32),
    ("SPECULAR_COLOR", "Specular Color (F0)", "F0 for spec/gloss workflows, sRGB", 64),
    ("SPECULAR_TINT", "Specular Tint", "Normalised F0 colour ($phongalbedotint / $phongtint)", 128),
    ("EMISSION", "Emission", "$selfillum and additive detail", 256),
    ("SHEEN", "Sheen", "$rimlight converted to sheen weight", 512),
    ("AO", "Ambient Occlusion", "Cycles AO bake (requires 'Bake AO')", 1024),
    ("OPACITY", "Opacity", "Separate alpha mask", 2048),
]
PACK_ITEMS = [
    ("ORM", "ORM (AO·Rough·Metal)", "glTF / Unreal / Godot packed map", 1),
    ("UNITY_METALLIC", "Unity Metallic+Smoothness", "RGB metallic, A smoothness", 2),
    ("UNITY_SPECULAR", "Unity Specular+Smoothness", "RGB F0, A smoothness", 4),
    ("HDRP_MASK", "HDRP Mask Map", "R metal, G AO, B detail(0), A smoothness", 8),
]


def _upd_noop(self, context):
    pass


class VLGDL_MaterialSettings(bpy.types.PropertyGroup):
    imported: BoolProperty(default=False)
    vmt_lightwarp_path: StringProperty(name="$lightwarptexture", description="Game path of the lightwarp texture")
    vmt_texture_paths: StringProperty(default="{}")

    # textures
    base_image: PointerProperty(name="$basetexture", type=bpy.types.Image)
    bump_image: PointerProperty(name="$bumpmap", type=bpy.types.Image)
    exponent_image: PointerProperty(name="$phongexponenttexture", type=bpy.types.Image)
    envmask_image: PointerProperty(name="$envmapmask", type=bpy.types.Image)
    selfillummask_image: PointerProperty(name="$selfillummask", type=bpy.types.Image)
    detail_image: PointerProperty(name="$detail", type=bpy.types.Image)
    lightwarp_image: PointerProperty(name="$lightwarptexture", type=bpy.types.Image)
    # alpha channels that come from separate images (a VTF exported as colour + mask PNGs)
    base_alpha_image: PointerProperty(name="Base Alpha", type=bpy.types.Image,
                                      description="Separate image holding $basetexture's alpha (tint / phong / "
                                                  "envmap mask). Empty = the base image's own alpha")
    base_alpha_from_alpha: BoolProperty(default=False)
    bump_alpha_image: PointerProperty(name="Normal Alpha", type=bpy.types.Image,
                                      description="Separate image holding $bumpmap's alpha (phong mask). "
                                                  "Empty = the normal map's own alpha")
    bump_alpha_from_alpha: BoolProperty(default=False)

    # albedo
    color2: FloatVectorProperty(name="$color2", subtype="COLOR_GAMMA", size=3, min=0, soft_max=1, default=(1, 1, 1))
    blendtintbybasealpha: BoolProperty(name="$blendtintbybasealpha")
    blendtintcoloroverbase: FloatProperty(name="$blendtintcoloroverbase", min=0, max=1)
    translucent: BoolProperty(name="$translucent")
    alphatest: BoolProperty(name="$alphatest")
    alphatestreference: FloatProperty(name="$alphatestreference", min=0, max=1, default=0.5)
    # phong
    phong: BoolProperty(name="$phong")
    phongboost: FloatProperty(name="$phongboost", min=0, soft_max=20, default=1)
    use_phongexponent: BoolProperty(name="Use $phongexponent", description="When set, overrides the exponent texture (Source behaviour)")
    phongexponent: FloatProperty(name="$phongexponent", min=0, soft_max=150, default=5)
    use_phongexponentfactor: BoolProperty(name="Use $phongexponentfactor")
    phongexponentfactor: FloatProperty(name="$phongexponentfactor", min=0, soft_max=300, default=149)
    phongfresnelranges: FloatVectorProperty(name="$phongfresnelranges", size=3, default=(0, 0.5, 1), min=0, soft_max=4)
    use_phongtint: BoolProperty(name="Use $phongtint")
    phongtint: FloatVectorProperty(name="$phongtint", subtype="COLOR_GAMMA", size=3, min=0, soft_max=1, default=(1, 1, 1))
    phongalbedotint: BoolProperty(name="$phongalbedotint")
    phongalbedoboost: FloatProperty(name="$phongalbedoboost", min=0, soft_max=10, default=1)
    basemapalphaphongmask: BoolProperty(name="$basemapalphaphongmask")
    invertphongmask: BoolProperty(name="$invertphongmask")
    halflambert: BoolProperty(name="$halflambert", description="Always on for $phong materials in Source")
    # rim
    rimlight: BoolProperty(name="$rimlight")
    rimlightexponent: FloatProperty(name="$rimlightexponent", min=0, soft_max=64, default=4)
    rimlightboost: FloatProperty(name="$rimlightboost", min=0, soft_max=10, default=1)
    rimmask: BoolProperty(name="$rimmask", description="Exponent texture alpha masks the rim")
    # envmap
    envmap: BoolProperty(name="$envmap")
    envmaptint: FloatVectorProperty(name="$envmaptint", subtype="COLOR_GAMMA", size=3, min=0, soft_max=1, default=(1, 1, 1))
    envmapfresnel: FloatProperty(name="$envmapfresnel", min=0, max=1)
    basealphaenvmapmask: BoolProperty(name="$basealphaenvmapmask")
    normalmapalphaenvmapmask: BoolProperty(name="$normalmapalphaenvmapmask")
    # selfillum
    selfillum: BoolProperty(name="$selfillum")
    selfillumtint: FloatVectorProperty(name="$selfillumtint", subtype="COLOR_GAMMA", size=3, min=0, soft_max=1, default=(1, 1, 1))
    selfillummaskscale: FloatProperty(name="$selfillummaskscale", min=0, soft_max=4, default=1)
    # detail
    detailscale: FloatProperty(name="$detailscale", min=0.001, soft_max=64, default=4)
    detailblendmode: IntProperty(name="$detailblendmode", min=0, max=11, default=0)
    detailblendfactor: FloatProperty(name="$detailblendfactor", min=0, max=1, default=1)
    detailtint: FloatVectorProperty(name="$detailtint", subtype="COLOR_GAMMA", size=3, min=0, soft_max=1, default=(1, 1, 1))

    # conversion
    normal_input: EnumProperty(name="Normal Map Input", items=[
        ("AUTO", "Auto-detect", "Detect green-channel convention from the map itself"),
        ("GL", "OpenGL (Y+)", ""), ("DX", "DirectX (Y-)", "")], default="AUTO")
    spec_strength: FloatProperty(name="Specular Intensity", min=0, soft_max=2, default=1.0,
                                 description="1 = highlight peak matches Source phong. Lower it if PBR "
                                             "environment reflections look too strong")
    max_f0: FloatProperty(name="Max F0", min=0.0, max=1.0, default=0.16,
                          description="Cap on reflectance at normal incidence. Source phong never reflects the "
                                      "environment, PBR does - above ~0.16 (diamond, the most reflective "
                                      "non-metal) dark phong areas start to read as chrome. 1.0 = exact peak match")
    matte_unmasked: BoolProperty(name="Matte Where Unmasked", default=True,
                                 description="Push roughness to 1 where the phong mask is black, so engines "
                                             "that ignore the specular map still read those areas as matte")
    roughness_bias: FloatProperty(name="Roughness Bias", min=-1, max=1, default=0.0)
    env_roughness: FloatProperty(name="Envmap Roughness", min=0, max=1, default=0.12)
    env_strength: FloatProperty(name="Envmap Strength", min=0, soft_max=2, default=1.0)
    env_to_metallic: FloatProperty(name="Envmap → Metallic", min=0, max=4, default=0.0)
    rim_strength: FloatProperty(name="Rim → Sheen Strength", min=0, soft_max=2, default=1.0)
    rim_use_rimexponent: BoolProperty(name="Rim Falloff From $rimlightexponent", default=False,
                                      description="Source uses a fixed (1-N.V)^4 falloff for ambient rim; "
                                                  "enable to follow $rimlightexponent instead (SourceIO behaviour)")
    lightwarp_brightness: BoolProperty(name="Match Lightwarp Brightness", default=False,
                                       description="Scale albedo by 2*lightwarp(1) (the fully lit value). Off by "
                                                   "default: hue transfer alone is independent of ramp brightness")
    lightwarp_hue: FloatProperty(name="Lightwarp Hue → Albedo", min=0, max=1, default=0.5, subtype="FACTOR")
    lightwarp_srgb: BoolProperty(name="Lightwarp is sRGB", default=False,
                                 description="Source samples lightwarps without sRGB decode; leave off")
    selfillum_darken_base: BoolProperty(name="Remove Self-Illum From Base", default=True)

    # output
    output_dir: StringProperty(name="Folder", subtype="DIR_PATH", default="//pbr_textures/")
    file_prefix: StringProperty(name="Prefix", description="Defaults to the material name")
    maps: EnumProperty(name="Maps", items=MAP_ITEMS, options={"ENUM_FLAG"},
                       default={"BASECOLOR", "NORMAL_GL", "NORMAL_DX", "ROUGHNESS", "METALLIC", "SPECULAR",
                                "SPECULAR_COLOR", "SPECULAR_TINT", "EMISSION", "SHEEN", "AO"})
    packs: EnumProperty(name="Packed", items=PACK_ITEMS, options={"ENUM_FLAG"}, default={"ORM"})
    data_bits: EnumProperty(name="Data Map Depth", items=[("8", "8-bit", ""), ("16", "16-bit", "")], default="8",
                            description="Bit depth for normal / roughness / masks")
    resolution: EnumProperty(name="Resolution", items=[
        ("BASE", "Match $basetexture", ""), ("1024", "1024", ""), ("2048", "2048", ""),
        ("4096", "4096", ""), ("CUSTOM", "Custom", "")], default="BASE")
    custom_res: IntProperty(name="Size", min=16, max=16384, default=1024)
    write_json: BoolProperty(name="Look Data (.json + ramp)", default=True,
                             description="Write the lightwarp ramp, rim and conversion data for stylised engine shaders")
    bake_ao: BoolProperty(name="Bake AO", default=True)
    ao_samples: IntProperty(name="AO Samples", min=1, max=4096, default=128)
    ao_distance: FloatProperty(name="AO Distance", min=0.0, soft_max=10, default=0.0,
                               description="0 = use the World's AO distance")
    ao_margin: IntProperty(name="Margin", min=0, max=64, default=8)
    assign_material: BoolProperty(name="Use On Mesh", default=True,
                                  description="Put the PBR material on the mesh in place of the VLG one "
                                              "(the VLG material is kept – Swap VLG ⇄ PBR switches back)")
    ao_to_base: BoolProperty(name="AO Into Base Color", default=False,
                             description="Multiply the baked AO into the material's base colour. Off keeps AO as "
                                         "its own (unconnected) image, as PBR engines expect")
    save_to_disk: BoolProperty(name="Save Textures To Disk", default=False,
                               description="Off: the textures are packed into the .blend. On: they are written "
                                           "as PNG files to the folder below and the material uses those files")

    # report
    last_report: StringProperty(default="")


class VLGDL_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    tf2_path: StringProperty(name="Team Fortress 2 Folder", subtype="DIR_PATH",
                             description="…/steamapps/common/Team Fortress 2 – used to pull lightwarps and "
                                         "other textures straight from the game VPKs")
    extra_content: StringProperty(name="Extra Content Folders",
                                  description="Optional ';'-separated folders that contain a materials/ tree "
                                              "(e.g. a Workshop item folder)")

    def draw(self, context):
        col = self.layout.column()
        row = col.row(align=True)
        row.prop(self, "tf2_path")
        row.operator("vlgdl.detect_tf2", text="", icon="VIEWZOOM")
        col.prop(self, "extra_content")


# =============================================================================
# import parameters
# =============================================================================

_TEXTURE_KEYS = {
    "$basetexture": ("base_image", "color"),
    "$bumpmap": ("bump_image", "data"),
    "$phongexponenttexture": ("exponent_image", "data"),
    "$envmapmask": ("envmask_image", "data"),
    "$selfillummask": ("selfillummask_image", "data"),
    "$detail": ("detail_image", "color"),
    "$lightwarptexture": ("lightwarp_image", "data"),
}


def apply_vmt_dict(st: VLGDL_MaterialSettings, d: dict):
    g = d.get

    def num(k, default):
        return float(sf.parse_number(g(k), default)) if g(k) is not None else default

    def vec(k, default):
        v = sf.parse_vector(g(k)) if g(k) is not None else None
        return tuple(v) if v else default

    def b(k, default=False):
        return sf.parse_bool(g(k), default)

    st.color2 = vec("$color2", (1, 1, 1))
    st.blendtintbybasealpha = b("$blendtintbybasealpha")
    st.blendtintcoloroverbase = num("$blendtintcoloroverbase", 0.0)
    st.translucent = b("$translucent")
    st.alphatest = b("$alphatest")
    st.alphatestreference = num("$alphatestreference", 0.5)
    st.phong = b("$phong")
    st.phongboost = num("$phongboost", 1.0)
    st.use_phongexponent = g("$phongexponent") is not None
    st.phongexponent = num("$phongexponent", 5.0)
    st.use_phongexponentfactor = g("$phongexponentfactor") is not None
    st.phongexponentfactor = num("$phongexponentfactor", 149.0)
    st.phongfresnelranges = vec("$phongfresnelranges", (0, 0.5, 1))
    st.use_phongtint = g("$phongtint") is not None
    st.phongtint = vec("$phongtint", (1, 1, 1))
    st.phongalbedotint = b("$phongalbedotint")
    st.phongalbedoboost = num("$phongalbedoboost", 1.0)
    st.basemapalphaphongmask = b("$basemapalphaphongmask")
    st.invertphongmask = b("$invertphongmask")
    st.halflambert = b("$halflambert") or st.phong
    st.rimlight = b("$rimlight")
    st.rimlightexponent = num("$rimlightexponent", 4.0)
    st.rimlightboost = num("$rimlightboost", 1.0)
    st.rimmask = b("$rimmask")
    st.envmap = bool(g("$envmap")) and str(g("$envmap")).strip() not in ("", "0")
    st.envmaptint = vec("$envmaptint", (1, 1, 1))
    st.envmapfresnel = num("$envmapfresnel", 0.0)
    st.basealphaenvmapmask = b("$basealphaenvmapmask")
    st.normalmapalphaenvmapmask = b("$normalmapalphaenvmapmask")
    st.selfillum = b("$selfillum")
    st.selfillumtint = vec("$selfillumtint", (1, 1, 1))
    st.selfillummaskscale = num("$selfillummaskscale", 1.0)
    st.detailscale = num("$detailscale", 4.0)
    st.detailblendmode = int(num("$detailblendmode", 0))
    st.detailblendfactor = num("$detailblendfactor", 1.0)
    st.detailtint = vec("$detailtint", (1, 1, 1))
    st.vmt_lightwarp_path = str(g("$lightwarptexture") or "")
    st.vmt_texture_paths = json.dumps({k: str(g(k)) for k in _TEXTURE_KEYS if g(k)})


def apply_node_values(st, node):
    """Read the *current* values of a SourceIO VertexLitGeneric group node."""
    ins = {i.name.split(" [")[0].strip().lower(): i for i in node.inputs}

    def val(name, default=None):
        s = ins.get(name)
        if s is None or not hasattr(s, "default_value"):
            return default
        v = s.default_value
        try:
            return tuple(v)
        except TypeError:
            return v

    def to_gamma(c):
        return tuple(max(0.0, float(x)) ** (1 / 2.2) for x in c[:3])

    if val("$color2") is not None: st.color2 = val("$color2")[:3]
    if val("$blendtintbybasealpha") is not None: st.blendtintbybasealpha = val("$blendtintbybasealpha") > 0.5
    if val("$blendtintcoloroverbase") is not None: st.blendtintcoloroverbase = val("$blendtintcoloroverbase")
    if val("$phong") is not None: st.phong = val("$phong") > 0.01
    if val("$phongboost") is not None: st.phongboost = val("$phongboost")
    if val("$phongexponent") is not None:
        st.use_phongexponent = True; st.phongexponent = val("$phongexponent")
    if val("$phongfresnelranges") is not None: st.phongfresnelranges = val("$phongfresnelranges")[:3]
    if val("$phongtint") is not None:  # the SourceIO group uses phongtint as a linear multiplier
        st.use_phongtint = True; st.phongtint = to_gamma(val("$phongtint"))
    if val("$rimlight") is not None: st.rimlight = val("$rimlight") > 0.01
    if val("$rimlightboost") is not None: st.rimlightboost = val("$rimlightboost")
    if val("$rimlightexponent") is not None: st.rimlightexponent = val("$rimlightexponent")
    if val("$envmap") is not None: st.envmap = val("$envmap") > 0.01
    if val("$envmaptint") is not None: st.envmaptint = val("$envmaptint")[:3]
    if val("$envmapfresnel") is not None: st.envmapfresnel = val("$envmapfresnel")
    if val("$selfillum") is not None: st.selfillum = val("$selfillum") > 0.01
    if val("$selfillumtint") is not None: st.selfillumtint = to_gamma(val("$selfillumtint"))
    if val("$selfillummaskscale") is not None: st.selfillummaskscale = val("$selfillummaskscale")
    st.halflambert = st.halflambert or st.phong


def detect_alpha_inputs(mat, st, reset=False):
    """SourceIO-style VLG groups take the base / normal alpha channels on sockets of their own, often fed by
    separate mask images (a VTF exported as colour + mask PNGs)."""
    node = find_vlg_group_node(mat)
    if node is None:
        return
    if reset:
        st.base_alpha_image = None
        st.bump_alpha_image = None
    for sock in node.inputs:
        nm = sock.name.lower()
        img, out = linked_image_output(sock)
        if img is None or img.size[0] <= 1:
            continue
        from_alpha = out == "Alpha"
        if "basemap texture alpha" in nm or "basetexture alpha" in nm:
            if st.base_alpha_image is None and not (img == st.base_image and from_alpha):
                st.base_alpha_image, st.base_alpha_from_alpha = img, from_alpha
        elif "bumpmap texture alpha" in nm or "normal map alpha" in nm:
            if st.bump_alpha_image is None and not (img == st.bump_image and from_alpha):
                st.bump_alpha_image, st.bump_alpha_from_alpha = img, from_alpha
        elif nm.startswith("$selfillummask") and st.selfillummask_image is None and not from_alpha:
            st.selfillummask_image = img


def assign_textures(context, mat, st, report):
    """Fill image slots: node links first, then names from the VMT, then the game files."""
    node = find_vlg_group_node(mat)
    if node is not None:
        for sock in node.inputs:
            nm = sock.name.lower()
            img = linked_image(sock)
            if img is None:
                continue
            if nm.startswith("$basetexture"):
                st.base_image = img
            elif nm.startswith("$bumpmap"):
                st.bump_image = img
        detect_alpha_inputs(mat, st, reset=True)
    if mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.type != "TEX_IMAGE" or n.image is None or n.image.size[0] <= 1:
                continue
            lab = (n.label or n.name).lower().strip()
            if lab in _TEXTURE_KEYS:
                attr = _TEXTURE_KEYS[lab][0]
                if getattr(st, attr) is None:
                    setattr(st, attr, n.image)
    paths = json.loads(st.vmt_texture_paths or "{}")
    fs = None
    for key, (attr, role) in _TEXTURE_KEYS.items():
        if getattr(st, attr) is not None or key not in paths:
            continue
        if key == "$detail" and st.detailblendfactor <= 0:
            continue
        img = find_image_by_vmt_path(paths[key])
        if img is None:
            if fs is None:
                fs = source_fs(context) or False
            if fs:
                try:
                    img = load_vtf_as_image(fs, paths[key], role)
                    if img is not None:
                        report.append(f"Loaded {key} from game files: {paths[key]}")
                except Exception as e:  # noqa
                    report.append(f"Could not decode {paths[key]}: {e}")
        if img is not None:
            setattr(st, attr, img)
        else:
            report.append(f"{key} '{paths[key]}' not found – set it manually if it matters")


# =============================================================================
# operators
# =============================================================================


class VLGDL_OT_detect_tf2(bpy.types.Operator):
    bl_idname = "vlgdl.detect_tf2"
    bl_label = "Find Team Fortress 2"
    bl_description = "Search Steam libraries for Team Fortress 2"

    def execute(self, context):
        p = sf.find_tf2()
        prefs = addon_prefs(context)
        if not p:
            self.report({"WARNING"}, "TF2 not found – set the folder manually")
            return {"CANCELLED"}
        if prefs:
            prefs.tf2_path = p
        self.report({"INFO"}, f"TF2: {p}")
        return {"FINISHED"}


class VLGDL_OT_import(bpy.types.Operator):
    bl_idname = "vlgdl.import_params"
    bl_label = "Read VLG Parameters"
    bl_description = "Fill the VLG settings from the material (SourceIO VMT data or node values)"
    bl_options = {"REGISTER", "UNDO"}

    source: EnumProperty(name="Source", items=[
        ("VMT", "VMT (game-accurate)", "Use the original .vmt values stored by SourceIO"),
        ("NODES", "Node values (as seen in Blender)", "Use the current values on the VertexLitGeneric node, "
                                                       "including any tweaks you made")], default="VMT")

    @classmethod
    def poll(cls, context):
        return vlg_source_of(getattr(context, "material", None)) is not None

    def execute(self, context):
        mat = vlg_source_of(getattr(context, "material", None))
        st = mat.vlgdl
        rep = []
        d = material_vmt_params(mat)
        if d is not None:
            apply_vmt_dict(st, d)
        elif self.source == "VMT":
            rep.append("No SourceIO VMT data on this material – used node values")
        node = find_vlg_group_node(mat)
        if self.source == "NODES" or d is None:
            if node is not None:
                apply_node_values(st, node)
            elif d is None:
                self.report({"ERROR"}, "No VMT data or VertexLitGeneric node found – use 'Load .vmt'")
                return {"CANCELLED"}
        assign_textures(context, mat, st, rep)
        if st.lightwarp_image is None and st.vmt_lightwarp_path:
            rep.append("Lightwarp not loaded – use 'Find in Game' or pick the file")
        st.imported = True
        st.last_report = "\n".join(rep)
        for r in rep:
            log(r)
        self.report({"INFO"}, "VLG parameters read" + (f" ({len(rep)} notes)" if rep else ""))
        return {"FINISHED"}


class VLGDL_OT_load_vmt(bpy.types.Operator):
    bl_idname = "vlgdl.load_vmt"
    bl_label = "Load .vmt"
    bl_description = "Read VLG parameters from a .vmt file"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.vmt", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return vlg_source_of(getattr(context, "material", None)) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8", errors="replace") as f:
                kv = sf.parse_keyvalues(f.read())
        except OSError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if kv["shader"] and "vertexlitgeneric" not in kv["shader"]:
            self.report({"WARNING"}, f"Shader is '{kv['shader']}', not VertexLitGeneric – converting anyway")
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        apply_vmt_dict(st, sf.flatten_vmt_params(kv["params"]))
        rep = []
        assign_textures(context, vlg_source_of(getattr(context, "material", None)), st, rep)
        st.imported = True
        st.last_report = "\n".join(rep)
        return {"FINISHED"}


class VLGDL_OT_find_lightwarp(bpy.types.Operator):
    bl_idname = "vlgdl.find_lightwarp"
    bl_label = "Find in Game"
    bl_description = "Extract the $lightwarptexture from your TF2 install (VPK or loose files)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return vlg_source_of(getattr(context, "material", None)) is not None

    def execute(self, context):
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        path = st.vmt_lightwarp_path
        if not path:
            self.report({"ERROR"}, "This material has no $lightwarptexture")
            return {"CANCELLED"}
        img = find_image_by_vmt_path(path)
        if img is None:
            fs = source_fs(context)
            if fs is None:
                self.report({"ERROR"}, "Set the Team Fortress 2 folder in the add-on preferences")
                return {"CANCELLED"}
            try:
                img = load_vtf_as_image(fs, path, "data")
            except Exception as e:  # noqa
                self.report({"ERROR"}, f"Could not decode {path}: {e}")
                return {"CANCELLED"}
            if img is None:
                self.report({"ERROR"}, f"{path}.vtf not found in the game folders")
                return {"CANCELLED"}
        st.lightwarp_image = img
        self.report({"INFO"}, f"Lightwarp: {img.name} ({img.size[0]}×{img.size[1]})")
        return {"FINISHED"}


class VLGDL_OT_load_texture_file(bpy.types.Operator):
    bl_idname = "vlgdl.load_texture_file"
    bl_label = "Load Texture File"
    bl_description = "Load a .vtf / .png / .tga into one of the texture slots"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.vtf;*.png;*.tga;*.jpg;*.jpeg;*.exr;*.tif;*.tiff", options={"HIDDEN"})
    slot: StringProperty(default="lightwarp_image")

    @classmethod
    def poll(cls, context):
        return vlg_source_of(getattr(context, "material", None)) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        role = "color" if self.slot in ("base_image", "detail_image") else "data"
        if self.filepath.lower().endswith(".vtf"):
            try:
                with open(self.filepath, "rb") as f:
                    arr, _ = sf.read_vtf(f.read())
            except Exception as e:  # noqa
                self.report({"ERROR"}, f"VTF: {e}")
                return {"CANCELLED"}
            img = array_to_image(os.path.splitext(os.path.basename(self.filepath))[0], arr[::-1].copy(),
                                 "sRGB" if role == "color" else "Non-Color")
        else:
            img = bpy.data.images.load(self.filepath, check_existing=True)
            img.colorspace_settings.name = "sRGB" if role == "color" else "Non-Color"
        setattr(st, self.slot, img)
        if self.slot in ("base_alpha_image", "bump_alpha_image"):
            setattr(st, self.slot.replace("_image", "_from_alpha"), False)
        return {"FINISHED"}


# ----------------------------------------------------------------------------- bake


def settings_to_core(st):
    p = core.VLGParams(
        color2=tuple(st.color2), blendtintbybasealpha=st.blendtintbybasealpha,
        blendtintcoloroverbase=st.blendtintcoloroverbase, translucent=st.translucent,
        alphatest=st.alphatest, alphatestreference=st.alphatestreference,
        phong=st.phong, phongboost=st.phongboost,
        phongexponent=st.phongexponent if st.use_phongexponent else None,
        phongexponentfactor=st.phongexponentfactor if st.use_phongexponentfactor else None,
        phongfresnelranges=tuple(st.phongfresnelranges),
        phongtint=tuple(st.phongtint) if st.use_phongtint else None,
        phongalbedotint=st.phongalbedotint, phongalbedoboost=st.phongalbedoboost,
        basemapalphaphongmask=st.basemapalphaphongmask, invertphongmask=st.invertphongmask,
        halflambert=st.halflambert, rimlight=st.rimlight, rimlightexponent=st.rimlightexponent,
        rimlightboost=st.rimlightboost, rimmask=st.rimmask, envmap=st.envmap,
        envmaptint=tuple(st.envmaptint), envmapfresnel=st.envmapfresnel,
        basealphaenvmapmask=st.basealphaenvmapmask, normalmapalphaenvmapmask=st.normalmapalphaenvmapmask,
        selfillum=st.selfillum, selfillumtint=tuple(st.selfillumtint), selfillummaskscale=st.selfillummaskscale,
        detailscale=st.detailscale, detailblendmode=st.detailblendmode,
        detailblendfactor=st.detailblendfactor, detailtint=tuple(st.detailtint))
    s = core.ConvertSettings(
        normal_input=st.normal_input, spec_strength=st.spec_strength, max_f0=st.max_f0,
        matte_unmasked=st.matte_unmasked, roughness_bias=st.roughness_bias,
        env_roughness=st.env_roughness, env_strength=st.env_strength, env_to_metallic=st.env_to_metallic,
        rim_strength=st.rim_strength, rim_use_rimexponent=st.rim_use_rimexponent,
        lightwarp_brightness=st.lightwarp_brightness, lightwarp_hue=st.lightwarp_hue,
        lightwarp_srgb=st.lightwarp_srgb, selfillum_darken_base=st.selfillum_darken_base)
    return p, s


def objects_using(mat):
    return [o for o in bpy.data.objects if o.type == "MESH" and any(sl.material == mat for sl in o.material_slots)]


def bake_ao(context, mat, w, h, st):
    """Cycles AO bake into a temporary image; returns (H,W) float or None."""
    objs = objects_using(mat)
    if not objs:
        return None, "no mesh uses this material"
    scene = context.scene
    view_layer = context.view_layer
    prev = dict(engine=scene.render.engine, samples=scene.cycles.samples,
                active=view_layer.objects.active,
                sel=[o for o in view_layer.objects if o.select_get()])
    prev_dist = None
    img = bpy.data.images.new("__vlgdl_ao", w, h, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    # texels no face covers stay white, so meshes sharing the sheet are not darkened
    img.pixels.foreach_set(np.ones(w * h * 4, np.float32))
    dummy = bpy.data.images.new("__vlgdl_dummy", 8, 8, alpha=False, float_buffer=True)
    temp_nodes = []
    hidden_mod = []
    try:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scene.render.engine = "CYCLES"
        scene.cycles.samples = st.ao_samples
        if st.ao_distance > 0 and scene.world:
            prev_dist = scene.world.light_settings.distance
            scene.world.light_settings.distance = st.ao_distance
        mats = {sl.material for o in objs for sl in o.material_slots if sl.material}
        for m in mats:
            if m.node_tree is None:
                continue
            n = m.node_tree.nodes.new("ShaderNodeTexImage")
            n.image = img if m == mat else dummy
            prev_active = m.node_tree.nodes.active
            for other in m.node_tree.nodes:
                other.select = False
            n.select = True
            m.node_tree.nodes.active = n
            temp_nodes.append((m, n, prev_active))
        for o in view_layer.objects:
            o.select_set(False)
        for o in objs:
            if o.name in view_layer.objects:
                o.select_set(True)
        view_layer.objects.active = objs[0]
        bpy.ops.object.bake(type="AO", margin=st.ao_margin, use_clear=False, target="IMAGE_TEXTURES")
        arr = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(arr)
        return arr.reshape(h, w, 4)[..., 0].copy(), None
    except Exception as e:  # noqa
        return None, str(e)
    finally:
        for m, n, pa in temp_nodes:
            m.node_tree.nodes.remove(n)
            if pa is not None:
                m.node_tree.nodes.active = pa
        scene.render.engine = prev["engine"]
        scene.cycles.samples = prev["samples"]
        if prev_dist is not None:
            scene.world.light_settings.distance = prev_dist
        for o in view_layer.objects:
            o.select_set(o in prev["sel"])
        view_layer.objects.active = prev["active"]
        bpy.data.images.remove(img)
        bpy.data.images.remove(dummy)


def safe_name(s):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)


def images_from_files(files, prefix):
    """Load saved PNGs (relative to the .blend when possible)."""
    out = {}
    for key, path in files.items():
        img = bpy.data.images.load(path, check_existing=True)
        if bpy.data.filepath:
            try:
                img.filepath = bpy.path.relpath(path)
            except ValueError:  # different drive on Windows
                pass
        if img.packed_file:
            img.unpack(method="REMOVE")
        img.reload()
        out[key] = img
    return out


def images_from_arrays(arrays, prefix):
    """Packed images in the .blend – nothing on disk."""
    out = {}
    for key, arr in arrays.items():
        a = np.asarray(arr, np.float32)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, 2)
        if a.shape[2] == 3:
            a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
        name = f"{prefix}_{key}"
        old = bpy.data.images.get(name)
        if old is not None and old.filepath and not old.packed_file:
            old.name = name + "_old"            # a saved file from an earlier bake: leave it alone
        out[key] = array_to_image(name, a, "sRGB" if key in SRGB_KEYS else "Non-Color", pack=True)
    return out


SRGB_KEYS = {"basecolor", "specular_color", "specular_tint", "emission"}


def build_pbr_material(mat, images, info, st):
    name = f"{mat.name}_PBR"
    pm = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    try:
        pm.use_nodes = True
    except Exception:  # noqa
        pass
    nt = pm.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (900, 0)
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.location = (500, 0)
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])

    def tex(key, cs, y, label):
        if key not in images:
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        img = images[key]
        img.colorspace_settings.name = cs
        n.image = img; n.label = label; n.location = (-500, y)
        return n

    y = 600
    bc = tex("basecolor", "sRGB", y, "Base Color"); y -= 280
    ao = tex("ao", "Non-Color", y, "Ambient Occlusion" if st.ao_to_base else "AO (not connected)"); y -= 280
    if bc:
        col = bc.outputs["Color"]
        if ao and st.ao_to_base:
            mix = nt.nodes.new("ShaderNodeMix"); mix.location = (-150, 600)
            mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"; mix.label = "× AO"
            mix.inputs["Factor"].default_value = 1.0
            nt.links.new(bc.outputs["Color"], mix.inputs[6]); nt.links.new(ao.outputs["Color"], mix.inputs[7])
            col = mix.outputs[2]
        nt.links.new(col, bs.inputs["Base Color"])
        if info.get("alpha_mode") != "OPAQUE":
            nt.links.new(bc.outputs["Alpha"], bs.inputs["Alpha"])
            if hasattr(pm, "surface_render_method"):
                pm.surface_render_method = "DITHERED" if info["alpha_mode"] == "CLIP" else "BLENDED"
            elif hasattr(pm, "blend_method"):
                pm.blend_method = "CLIP" if info["alpha_mode"] == "CLIP" else "BLEND"
    r = tex("roughness", "Non-Color", y, "Roughness"); y -= 280
    if r:
        nt.links.new(r.outputs["Color"], bs.inputs["Roughness"])
    m = tex("metallic", "Non-Color", y, "Metallic"); y -= 280
    if m:
        nt.links.new(m.outputs["Color"], bs.inputs["Metallic"])
    sc = tex("specular_color", "sRGB", y, "Specular Color (F0)"); y -= 280
    if sc:
        bw = nt.nodes.new("ShaderNodeRGBToBW"); bw.location = (-150, y + 280)
        dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.location = (60, y + 280)
        dv.inputs[1].default_value = 0.08
        dv.label = "F0 → IOR Level (0.5 = 4%)"
        nt.links.new(sc.outputs["Color"], bw.inputs["Color"])
        nt.links.new(bw.outputs["Val"], dv.inputs[0])
        nt.links.new(dv.outputs["Value"], bs.inputs["Specular IOR Level"])
    elif "specular" in images:
        sl = tex("specular", "Non-Color", y, "Specular Level"); y -= 280
        nt.links.new(sl.outputs["Color"], bs.inputs["Specular IOR Level"])
    stt = tex("specular_tint", "sRGB", y, "Specular Tint"); y -= 280
    if stt:
        nt.links.new(stt.outputs["Color"], bs.inputs["Specular Tint"])
    nm = tex("normal_gl", "Non-Color", y, "Normal (OpenGL)"); y -= 280
    if nm:
        nmn = nt.nodes.new("ShaderNodeNormalMap"); nmn.location = (-150, y + 280)
        nt.links.new(nm.outputs["Color"], nmn.inputs["Color"])
        nt.links.new(nmn.outputs["Normal"], bs.inputs["Normal"])
    sh = tex("sheen", "Non-Color", y, "Sheen (from $rimlight)"); y -= 280
    if sh:
        nt.links.new(sh.outputs["Color"], bs.inputs["Sheen Weight"])
        bs.inputs["Sheen Roughness"].default_value = info.get("sheen_roughness", 0.5)
        bs.inputs["Sheen Tint"].default_value = (1, 1, 1, 1)
    em = tex("emission", "sRGB", y, "Emission"); y -= 280
    if em:
        nt.links.new(em.outputs["Color"], bs.inputs["Emission Color"])
        bs.inputs["Emission Strength"].default_value = 1.0
    if info.get("alpha_mode") == "OPAQUE" and hasattr(pm, "surface_render_method"):
        pm.surface_render_method = "DITHERED"
    try:
        pm.diffuse_color = tuple(info.get("viewport_color", (0.8, 0.8, 0.8))) + (1.0,)
    except Exception:  # noqa
        pass
    pm.use_fake_user = True  # keep it when not assigned to anything yet
    mat.use_fake_user = True  # and keep the VLG original for Swap / re-baking
    pm["vlgdl_source"] = mat.name
    mat["vlgdl_pbr"] = pm.name
    return pm


def write_files(st, mat, maps, info, tex, p, out_dir, prefix):
    """Optional: the PNG texture set (+ packed maps and look .json) on disk. Returns key -> path."""
    h, w = maps["roughness"].shape
    ones = np.ones((h, w), np.float32)
    ao_or_one = maps.get("ao", ones)
    dbits = int(st.data_bits)
    files = {}

    def put(key, arr, bits=8):
        path = os.path.join(out_dir, f"{prefix}_{key}.png")
        pngio.write_png(path, arr, bits=bits)
        files[key] = path

    sel = set(st.maps)
    need_for_mat = {"basecolor", "normal_gl", "roughness", "metallic", "specular_color", "specular_tint",
                    "sheen", "emission"}
    emission_used = float(maps["emission"].max()) > 1e-4
    sheen_used = float(maps["sheen"].max()) > 1e-4
    table = [("BASECOLOR", "basecolor", 8), ("NORMAL_GL", "normal_gl", dbits), ("NORMAL_DX", "normal_dx", dbits),
             ("ROUGHNESS", "roughness", dbits), ("METALLIC", "metallic", dbits), ("SPECULAR", "specular", dbits),
             ("SPECULAR_COLOR", "specular_color", 8), ("SPECULAR_TINT", "specular_tint", 8),
             ("EMISSION", "emission", 8), ("SHEEN", "sheen", dbits), ("OPACITY", "opacity", dbits)]
    for flag, key, bits in table:
        if key == "emission" and not emission_used:
            continue
        if key == "sheen" and not sheen_used:
            continue
        if flag in sel or key in need_for_mat:
            arr = maps[key]
            if key == "basecolor" and info.get("alpha_mode") == "OPAQUE":
                arr = arr[..., :3]
            put(key, arr, bits)
    if "ao" in maps:
        put("ao", maps["ao"], dbits)
    smooth = 1.0 - maps["roughness"]
    if "ORM" in st.packs:
        put("orm", np.stack([ao_or_one, maps["roughness"], maps["metallic"]], -1), dbits)
    if "UNITY_METALLIC" in st.packs:
        m3 = np.repeat(maps["metallic"][..., None], 3, 2)
        put("unity_metallic_smoothness", np.concatenate([m3, smooth[..., None]], 2), 8)
    if "UNITY_SPECULAR" in st.packs:
        put("unity_specular_smoothness", np.concatenate([maps["specular_color"], smooth[..., None]], 2), 8)
    if "HDRP_MASK" in st.packs:
        put("hdrp_mask", np.stack([maps["metallic"], ao_or_one, np.zeros_like(smooth), smooth], -1), 8)

    if st.write_json:
        lw = info["lightwarp"]
        if tex.lightwarp is not None:
            ramp = tex.lightwarp[None, :, :]
            pngio.write_png(os.path.join(out_dir, f"{prefix}_lightwarp_ramp.png"), ramp, bits=8, flip_vertical=False)
        look = {
            "material": mat.name,
            "generator": "VLG → PBR Baker 1.0",
            "maps": {k: os.path.basename(v) for k, v in files.items()},
            "vlg_parameters": core.params_to_dict(p),
            "conversion": {
                "albedo_calibration_rgb": info["albedo_calibration"],
                "normal_input_detected": info.get("normal_detected"),
                "roughness_from_exponent": "r = (2/(4n+2))^0.25",
                "max_f0": info["max_f0"],
                "specular_level_max_unclamped": info["specular_level_max"],
                "specular_level_clipped_fraction": info["specular_level_clipped_fraction"],
                "sheen_roughness": info["sheen_roughness"],
                "alpha_mode": info["alpha_mode"],
                "notes": "Specular Level is clamped to 1 (8% F0). Where it clips, use specular_color (true F0) "
                         "in spec/gloss engines, or the Blender material, which drives IOR Level from F0 unclamped.",
            },
            "stylized_lighting": {
                "description": "Source diffuse = albedo * 2 * lightwarp(0.5*N.L+0.5). Use with a ramp/toon "
                               "shader to reproduce the VLG terminator and shadow colour exactly.",
                "has_lightwarp": lw["has_warp"],
                "lightwarp_ramp": f"{prefix}_lightwarp_ramp.png" if tex.lightwarp is not None else None,
                "lightwarp_lookup": "u = 0.5*dot(N,L)+0.5 ; diffuse = 2 * ramp(u)  (ramp read without sRGB decode)",
                "half_lambert": bool(p.halflambert or lw["has_warp"]),
                "shadow_side_color": lw["shadow_color"],
                "terminator_color": lw["terminator_color"],
                "lit_color": lw["peak_color"],
                "rim": {"enabled": p.rimlight, "boost": p.rimlightboost, "exponent": p.rimlightexponent,
                        "formula": "rimmask * boost * (1-N.V)^4 * ambient(V) * saturate(N.z)"},
                "phong": {"boost": p.phongboost, "exponent": p.phongexponent,
                          "fresnel_ranges": list(p.phongfresnelranges)},
            },
            "warnings": info["warnings"],
        }
        with open(os.path.join(out_dir, f"{prefix}_vlg_look.json"), "w") as f:
            json.dump(look, f, indent=2)

    return files


def run_bake(context, mat, report):
    st = mat.vlgdl
    t0 = time.time()
    if st.base_image is None:
        raise RuntimeError(f"{mat.name}: no $basetexture image – click 'Read VLG Parameters' first")
    prefix = safe_name(st.file_prefix or mat.name)
    out_dir = None
    if st.save_to_disk:
        out_dir = abspath_dir(st.output_dir)
        if st.output_dir.startswith("//") and not bpy.data.filepath:
            raise RuntimeError("Save the .blend first, or choose an absolute output folder "
                               "(or turn off Save Textures To Disk)")
        os.makedirs(out_dir, exist_ok=True)

    def with_alpha(arr, alpha_img, from_alpha):
        if arr is None or alpha_img is None:
            return arr
        a = image_to_array(alpha_img, "data")
        if a is None:
            return arr
        a = a[..., 3] if from_alpha else a[..., 0]
        h, w = arr.shape[:2]
        arr = arr.copy()
        arr[..., 3] = core.resample(a[..., None], h, w)[..., 0]
        return arr

    tex = core.Textures(
        base=with_alpha(image_to_array(st.base_image, "color"), st.base_alpha_image, st.base_alpha_from_alpha),
        bump=with_alpha(image_to_array(st.bump_image, "data"), st.bump_alpha_image, st.bump_alpha_from_alpha)
        if st.bump_image else None,
        exponent=image_to_array(st.exponent_image, "data") if st.exponent_image else None,
        envmask=image_to_array(st.envmask_image, "data") if st.envmask_image else None,
        selfillummask=image_to_array(st.selfillummask_image, "data") if st.selfillummask_image else None,
        detail=image_to_array(st.detail_image, "color") if st.detail_image else None,
        lightwarp=ramp_from_image(st.lightwarp_image) if st.lightwarp_image else None,
    )
    if st.resolution == "BASE":
        size = None
        h, w = tex.base.shape[:2]
    else:
        n = st.custom_res if st.resolution == "CUSTOM" else int(st.resolution)
        bh, bw = tex.base.shape[:2]
        w, h = (n, max(1, round(n * bh / bw))) if bw >= bh else (max(1, round(n * bw / bh)), n)
        size = (w, h)
    p, s = settings_to_core(st)
    maps, info = core.convert(tex, p, s, size=size)
    if st.vmt_lightwarp_path and tex.lightwarp is None:
        info["warnings"].append(f"Lightwarp '{st.vmt_lightwarp_path}' not loaded – albedo is not "
                                f"calibrated to it. Use 'Find in Game'.")

    ao = None
    if st.bake_ao:
        ao, err = bake_ao(context, mat, w, h, st)
        if ao is None:
            info["warnings"].append(f"AO bake skipped: {err}")
    if ao is not None:
        maps["ao"] = ao.astype(np.float32)
    emission_used = float(maps["emission"].max()) > 1e-4
    sheen_used = float(maps["sheen"].max()) > 1e-4
    for_mat = {}
    for key in ("basecolor", "normal_gl", "roughness", "metallic", "specular_color", "specular_tint", "sheen",
                "emission", "ao"):
        if key not in maps or (key == "emission" and not emission_used) or (key == "sheen" and not sheen_used):
            continue
        arr = maps[key]
        if key == "basecolor" and info.get("alpha_mode") == "OPAQUE":
            arr = arr[..., :3]
        for_mat[key] = arr
    try:
        info["viewport_color"] = [float(x) for x in np.asarray(maps["basecolor"][..., :3]).reshape(-1, 3).mean(0)]
    except Exception:  # noqa
        pass
    files = {}
    if st.save_to_disk:
        files = write_files(st, mat, maps, info, tex, p, out_dir, prefix)
        images = images_from_files({k: files[k] for k in for_mat if k in files}, prefix)
    else:
        images = images_from_arrays(for_mat, prefix)

    pm = build_pbr_material(mat, images, info, st)
    if st.assign_material:
        for o in objects_using(mat):
            for sl in o.material_slots:
                if sl.material == mat:
                    sl.material = pm
    where = f"{len(files)} files → {out_dir}" if out_dir else "textures packed in the .blend"
    report.append(f"{mat.name} → {pm.name}: {', '.join(sorted(images))}; {where}  ({time.time() - t0:.1f}s)")
    for wmsg in info["warnings"]:
        report.append(f"  ! {wmsg}")
    if info["specular_level_clipped_fraction"] > 0.001:
        report.append(f"  · F0 up to {info['max_f0']:.2f}: Specular Level clips on "
                      f"{100 * info['specular_level_clipped_fraction']:.1f}% of texels "
                      f"(specular_color keeps the true value)")
    return files, info, pm


def is_vlg(m):
    return m is not None and "vlgdl_source" not in m and (m.vlgdl.imported or find_vlg_group_node(m) is not None
                                                          or "vmt_parameters" in m)


def vlg_source_of(m):
    """The VLG material to bake from: itself, or the original of a baked PBR material."""
    if m is None:
        return None
    if "vlgdl_source" in m:
        return bpy.data.materials.get(m["vlgdl_source"])
    return m


def bake_materials(context, mats):
    """Bake each VLG material to '<name>_PBR'. Returns (ok count, report lines)."""
    report = []
    ok = 0
    for m in mats:
        if not m.vlgdl.imported:
            rep = []
            d = material_vmt_params(m)
            if d is not None:
                apply_vmt_dict(m.vlgdl, d)
            elif find_vlg_group_node(m):
                apply_node_values(m.vlgdl, find_vlg_group_node(m))
            assign_textures(context, m, m.vlgdl, rep)
            m.vlgdl.imported = True
            report += rep
        else:
            detect_alpha_inputs(m, m.vlgdl)
        # re-bake: put the VLG material back on the mesh for a moment (AO is baked from the mesh)
        old = bpy.data.materials.get(m.get("vlgdl_pbr", ""))
        swapped = []
        if old is not None:
            for o in objects_using(old):
                for sl in o.material_slots:
                    if sl.material == old:
                        sl.material = m
                        swapped.append(sl)
        try:
            run_bake(context, m, report)
            ok += 1
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            report.append(f"{m.name}: FAILED – {e}")
            for sl in swapped:
                sl.material = old
        else:
            if not m.vlgdl.assign_material:
                for sl in swapped:
                    sl.material = bpy.data.materials.get(m.get("vlgdl_pbr", "")) or m
    for r in report:
        log(r)
    for m in mats:
        m.vlgdl.last_report = "\n".join(report)
    return ok, report


class VLGDL_OT_bake(bpy.types.Operator):
    bl_idname = "vlgdl.bake"
    bl_label = "Bake to PBR Material"
    bl_description = "Convert the VLG material(s) into PBR material(s) in this .blend, with the textures baked in"
    bl_options = {"REGISTER", "UNDO"}

    all_selected: BoolProperty(name="All VLG Materials On Selected", default=False)

    @classmethod
    def poll(cls, context):
        return getattr(context, "material", None) is not None or context.selected_objects

    def execute(self, context):
        mats = []
        if self.all_selected:
            for o in context.selected_objects:
                for sl in getattr(o, "material_slots", []):
                    m = vlg_source_of(sl.material)
                    if m is not None and m not in mats and is_vlg(m):
                        mats.append(m)
        elif getattr(context, "material", None):
            m = vlg_source_of(getattr(context, "material", None))
            if m is not None:
                mats = [m]
        if not mats:
            self.report({"ERROR"}, "No VLG materials found on the selection")
            return {"CANCELLED"}
        ok, _ = bake_materials(context, mats)
        context.scene["vlgdl_last_report"] = "\n".join(mats[0].vlgdl.last_report.splitlines()[:40])
        self.report({"INFO"} if ok == len(mats) else {"WARNING"},
                    f"Baked {ok}/{len(mats)} material(s) to PBR" + ("" if ok == len(mats) else " – see the report"))
        return {"FINISHED"}


class VLGDL_OT_swap(bpy.types.Operator):
    bl_idname = "vlgdl.swap"
    bl_label = "Swap VLG ⇄ PBR"
    bl_description = "Toggle selected objects between the original VLG materials and their baked PBR versions"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objs = [o for o in context.selected_objects if o.type == "MESH"] or \
               ([context.object] if context.object else [])
        slots = [sl for o in objs for sl in o.material_slots if sl.material]
        # all one way: to PBR unless everything is PBR already
        to_pbr = not slots or any("vlgdl_source" not in sl.material for sl in slots
                                  if sl.material.get("vlgdl_pbr", "") in bpy.data.materials)
        n = 0
        for sl in slots:
            m = sl.material
            if to_pbr and "vlgdl_pbr" in m and m["vlgdl_pbr"] in bpy.data.materials:
                sl.material = bpy.data.materials[m["vlgdl_pbr"]]; n += 1
            elif not to_pbr and "vlgdl_source" in m and m["vlgdl_source"] in bpy.data.materials:
                sl.material = bpy.data.materials[m["vlgdl_source"]]; n += 1
        self.report({"INFO"}, f"{n} slot(s) → {'PBR' if to_pbr else 'VLG'}")
        return {"FINISHED"}


class VLGDL_OT_open_folder(bpy.types.Operator):
    bl_idname = "vlgdl.open_folder"
    bl_label = "Open Output Folder"

    def execute(self, context):
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        bpy.ops.wm.path_open(filepath=abspath_dir(st.output_dir))
        return {"FINISHED"}


# =============================================================================
# UI
# =============================================================================


class VLGDL_PT_main(bpy.types.Panel):
    bl_label = "VLG → PBR"
    bl_idname = "VLGDL_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return vlg_source_of(getattr(context, "material", None)) is not None

    def draw(self, context):
        lay = self.layout
        mat = vlg_source_of(getattr(context, "material", None))
        st = mat.vlgdl
        if mat != getattr(context, "material", None):
            lay.label(text=f"Baked from {mat.name} – settings below are its", icon="INFO")
        col = lay.column(align=True)
        row = col.row(align=True)
        op = row.operator("vlgdl.import_params", text="Read VMT", icon="IMPORT"); op.source = "VMT"
        op = row.operator("vlgdl.import_params", text="Read Node Values", icon="NODETREE"); op.source = "NODES"
        col.operator("vlgdl.load_vmt", text="Load .vmt File…", icon="FILE_TEXT")
        if not st.imported:
            lay.label(text="Read the parameters first.", icon="INFO")
        col = lay.column(align=True)
        col.scale_y = 1.5
        col.operator("vlgdl.bake", text="Bake to PBR Material", icon="RENDER_STILL").all_selected = False
        row = lay.row(align=True)
        row.operator("vlgdl.bake", text="Bake All Selected", icon="OBJECT_DATA").all_selected = True
        row.operator("vlgdl.swap", text="Swap VLG ⇄ PBR", icon="ARROW_LEFTRIGHT")
        if st.last_report:
            box = lay.box()
            for line in st.last_report.splitlines()[:14]:
                box.label(text=line, icon="ERROR" if "!" in line or "FAILED" in line else "DOT")


class VLGDL_PT_textures(bpy.types.Panel):
    bl_label = "Textures"
    bl_parent_id = "VLGDL_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    def draw(self, context):
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        col = self.layout.column()
        for attr in ("base_image", "bump_image", "exponent_image", "envmask_image", "selfillummask_image",
                     "detail_image"):
            if attr == "envmask_image":
                for a2 in ("base_alpha_image", "bump_alpha_image"):
                    r2 = col.row(align=True)
                    r2.prop(st, a2)
                    r2.operator("vlgdl.load_texture_file", text="", icon="FILEBROWSER").slot = a2
            row = col.row(align=True)
            row.prop(st, attr)
            row.operator("vlgdl.load_texture_file", text="", icon="FILEBROWSER").slot = attr
        box = col.box()
        row = box.row(align=True)
        row.prop(st, "lightwarp_image")
        row.operator("vlgdl.load_texture_file", text="", icon="FILEBROWSER").slot = "lightwarp_image"
        if st.vmt_lightwarp_path:
            r = box.row(align=True)
            r.label(text=st.vmt_lightwarp_path, icon="TEXTURE")
            r.operator("vlgdl.find_lightwarp", icon="VIEWZOOM")
            if st.lightwarp_image is None:
                box.label(text="Not loaded – the VLG shading ramp will be missing", icon="ERROR")


class VLGDL_PT_params(bpy.types.Panel):
    bl_label = "VLG Parameters"
    bl_parent_id = "VLGDL_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        lay = self.layout
        lay.use_property_split = True
        b = lay.box(); b.label(text="Albedo")
        b.prop(st, "color2"); b.prop(st, "blendtintbybasealpha"); b.prop(st, "blendtintcoloroverbase")
        r = b.row(); r.prop(st, "translucent"); r.prop(st, "alphatest")
        b = lay.box(); b.prop(st, "phong")
        c = b.column(); c.active = st.phong
        c.prop(st, "phongboost")
        r = c.row(heading="$phongexponent"); r.prop(st, "use_phongexponent", text=""); s = r.row(); s.active = st.use_phongexponent; s.prop(st, "phongexponent", text="")
        r = c.row(heading="Exponent Factor"); r.prop(st, "use_phongexponentfactor", text=""); s = r.row(); s.active = st.use_phongexponentfactor; s.prop(st, "phongexponentfactor", text="")
        c.prop(st, "phongfresnelranges")
        r = c.row(heading="$phongtint"); r.prop(st, "use_phongtint", text=""); s = r.row(); s.active = st.use_phongtint; s.prop(st, "phongtint", text="")
        c.prop(st, "phongalbedotint"); c.prop(st, "phongalbedoboost")
        c.prop(st, "basemapalphaphongmask"); c.prop(st, "invertphongmask"); c.prop(st, "halflambert")
        b = lay.box(); b.prop(st, "rimlight")
        c = b.column(); c.active = st.rimlight
        c.prop(st, "rimlightboost"); c.prop(st, "rimlightexponent"); c.prop(st, "rimmask")
        b = lay.box(); b.prop(st, "envmap")
        c = b.column(); c.active = st.envmap
        c.prop(st, "envmaptint"); c.prop(st, "envmapfresnel"); c.prop(st, "basealphaenvmapmask"); c.prop(st, "normalmapalphaenvmapmask")
        b = lay.box(); b.prop(st, "selfillum")
        c = b.column(); c.active = st.selfillum
        c.prop(st, "selfillumtint"); c.prop(st, "selfillummaskscale")
        b = lay.box(); b.label(text="Detail")
        b.prop(st, "detailblendmode"); b.prop(st, "detailblendfactor"); b.prop(st, "detailscale"); b.prop(st, "detailtint")


class VLGDL_PT_conversion(bpy.types.Panel):
    bl_label = "Conversion"
    bl_parent_id = "VLGDL_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = vlg_source_of(getattr(context, "material", None)).vlgdl
        lay = self.layout
        lay.use_property_split = True
        lay.prop(st, "normal_input")
        c = lay.column(heading="Specular"); c.prop(st, "spec_strength"); c.prop(st, "max_f0"); c.prop(st, "matte_unmasked"); c.prop(st, "roughness_bias")
        c = lay.column(heading="Lightwarp"); c.prop(st, "lightwarp_brightness"); c.prop(st, "lightwarp_hue"); c.prop(st, "lightwarp_srgb")
        c = lay.column(heading="Rim"); c.prop(st, "rim_strength"); c.prop(st, "rim_use_rimexponent")
        c = lay.column(heading="Envmap"); c.prop(st, "env_strength"); c.prop(st, "env_roughness"); c.prop(st, "env_to_metallic")
        lay.prop(st, "selfillum_darken_base")


class VLGDL_PT_output(bpy.types.Panel):
    bl_label = "Output"
    bl_parent_id = "VLGDL_PT_main"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    def draw(self, context):
        draw_output(self.layout, vlg_source_of(getattr(context, "material", None)).vlgdl)


def draw_output(lay, st):
    lay.use_property_split = True
    lay.prop(st, "assign_material")
    row = lay.row(); row.prop(st, "resolution")
    if st.resolution == "CUSTOM":
        row.prop(st, "custom_res")
    c = lay.column(heading="AO")
    c.prop(st, "bake_ao")
    s = c.column(); s.active = st.bake_ao
    s.prop(st, "ao_to_base"); s.prop(st, "ao_samples"); s.prop(st, "ao_distance"); s.prop(st, "ao_margin")
    lay.prop(st, "save_to_disk")
    if st.save_to_disk:
        box = lay.box()
        row = box.row(align=True)
        row.prop(st, "output_dir")
        row.operator("vlgdl.open_folder", text="", icon="FILE_FOLDER")
        box.prop(st, "file_prefix")
        box.prop(st, "data_bits")
        box.label(text="Maps")
        grid = box.grid_flow(columns=2, align=True)
        grid.prop(st, "maps")
        box.label(text="Packed Maps")
        grid = box.grid_flow(columns=2, align=True)
        grid.prop(st, "packs")
        box.prop(st, "write_json")


class VLGDL_PT_sidebar(bpy.types.Panel):
    bl_label = "VLG → PBR"
    bl_idname = "VLGDL_PT_sidebar"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VLG → PBR"

    def draw(self, context):
        lay = self.layout
        objs = [o for o in context.selected_objects if o.type == "MESH"]
        mats = []
        for o in objs:
            for sl in o.material_slots:
                m = vlg_source_of(sl.material)
                if m is not None and is_vlg(m) and m not in mats:
                    mats.append(m)
        col = lay.column(align=True)
        col.scale_y = 1.6
        col.operator("vlgdl.bake", text="Bake Selected to PBR", icon="RENDER_STILL").all_selected = True
        col.enabled = bool(mats)
        lay.operator("vlgdl.swap", icon="ARROW_LEFTRIGHT")
        if not objs:
            lay.label(text="Select the model", icon="INFO")
            return
        if not mats:
            lay.label(text="No VLG materials on the selection", icon="INFO")
            return
        box = lay.box()
        for m in mats[:12]:
            pbr = m.get("vlgdl_pbr", "")
            done = pbr in bpy.data.materials
            box.label(text=f"{m.name}" + ("  → baked" if done else ""), icon="CHECKMARK" if done else "MATERIAL")
        rep = context.scene.get("vlgdl_last_report", "")
        if rep:
            b = lay.box()
            for line in rep.splitlines()[:10]:
                b.label(text=line, icon="ERROR" if "!" in line or "FAILED" in line else "DOT")
        lay.label(text="Settings: Properties ▸ Material ▸ VLG → PBR", icon="PROPERTIES")


classes = (
    VLGDL_MaterialSettings, VLGDL_Preferences,
    VLGDL_OT_detect_tf2, VLGDL_OT_import, VLGDL_OT_load_vmt, VLGDL_OT_find_lightwarp,
    VLGDL_OT_load_texture_file, VLGDL_OT_bake, VLGDL_OT_swap, VLGDL_OT_open_folder,
    VLGDL_PT_main, VLGDL_PT_textures, VLGDL_PT_params, VLGDL_PT_conversion, VLGDL_PT_output,
    VLGDL_PT_sidebar,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Material.vlgdl = PointerProperty(type=VLGDL_MaterialSettings)


def unregister():
    del bpy.types.Material.vlgdl
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
