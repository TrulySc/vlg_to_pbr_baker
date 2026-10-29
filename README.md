# VLG → PBR Baker 2.0 (Blender 4.2+, tested on 5.2)

Turns Source 1 **VertexLitGeneric** materials (as imported by SourceIO, or read from a `.vmt`) into ordinary
Blender **PBR materials**, in one click. The textures are baked and packed into the .blend.

## Install
Edit ▸ Preferences ▸ Get Extensions ▸ ▾ ▸ **Install from Disk…** → `vlg_to_pbr_baker.zip`. Remove the VLG → Deadlock Baker
first; this replaces it.

Optionally set the **Team Fortress 2 folder** in the add-on's preferences. Textures and lightwarps the import
didn't bring in are then pulled from the game's VPKs.

## From a .vmt (recommended)
Select the mesh. In 3D View ▸ Sidebar (N) ▸ **VLG → PBR**, press **Import .vmt…** and pick one or more `.vmt`
files.
- **Values**: everything is read from the file, including `patch` VMTs and their `include`.
- **Textures**: found under the .vmt's own `materials/` folder (`.vtf`, `.png` or `.tga`), then next to the .vmt,
  then in TF2's VPKs and `tf/custom`. Nothing cached in the .blend is used.
- **Which slot**: a slot whose material is named like the .vmt (as SourceIO names them), otherwise the active
  slot.
- **The link**: the material remembers its .vmt. Every **Bake** re-reads the file and its textures, so after you
  edit the .vmt or repaint a texture, press **Bake Selected to PBR** again. **Link .vmt File…** / **Unlink** are
  in Properties ▸ Material ▸ VLG → PBR.

## Bake From Current Material (secondary)
**Bake From Current Material** ignores the linked .vmt and bakes the material as it is now. That includes textures
you swapped in its VertexLitGeneric nodes or in the Textures panel, and image files you edited, which are re-read
from disk. Use it to try replacement textures before a bake. The material's current values are kept.

## Without a .vmt
**Bake Selected to PBR** also works on SourceIO materials with no .vmt linked. It uses the VMT values SourceIO
stored, or the VertexLitGeneric node's values. If the group takes the base or normal-map alpha from separate mask
images (its "[basemap texture alpha]" / "[bumpmap texture alpha]" inputs), those masks are used.

## What you get
For each VLG material `X`, a new material `X_PBR` is put on the mesh in its place.
- **Nodes**: a Principled BSDF with Base Color (× AO), Roughness, Metallic and a Normal Map (OpenGL).
  `$selfillum` becomes Emission, and `$translucent`/`$alphatest` become Alpha.
- **Images**: baked and packed into the .blend.
- **The original**: `X` is kept. **Swap VLG ⇄ PBR** switches back, and baking again re-bakes from `X`.

**$detail** (default **Mix Node**): the detail texture stays its own image. A UV Mapping node applies
`$detailscale` and a mix node follows `$detailblendmode`, as on the VDC $detail page:

| Mode | Blend |
|---|---|
| 0 DecalModulate / 7 Two-Pattern | Multiply × 2 (raw texture, 0.5 = no change) |
| 1 Additive | Add |
| 2 Translucent Detail | Mix by detail alpha × factor |
| 3 Blend Factor Fade | Mix by factor |
| 4 Translucent Base | Mix by base alpha |
| 5 / 6 Unlit Additive | Added to Emission |
| 8 Multiply | Multiply |
| 9 Base Mask via Detail Alpha | Alpha × detail alpha |

`$detailblendfactor` drives the mix and `$detailtint` tints the detail. **Bake Into Color** (Conversion panel)
bakes the detail into the base colour instead. If your AO is already in the `$detail`, turn **Bake AO** off so it
isn't applied twice.

**TF2 Look** (default **Auto**: added when the VMT uses `$phong`, `$lightwarptexture`, `$halflambert` or
`$rimlight`) is a small shading setup on its own EEVEE output, so it shows in the Material Preview and EEVEE
viewport:

    [TF2 Light] ──► [$lightwarptexture] ──► [TF2 Shading] ──► Material Output (EEVEE)

- **Textures**: it only uses the material's own PBR textures (Base Color, Specular, Roughness, Normal) plus the
  lightwarp. No extra textures are added.
- **TF2 Light**: an HLMV-style key light, fixed to the view so highlights move as you orbit. **Light Direction** is
  (right, up, toward you). It gives the half-lambert term.
- **TF2 Shading**: Source's VLG maths in gamma space.
  - Diffuse is albedo × (2 × lightwarp, or half-lambert² without one), plus ambient.
  - Phong is pow(R·L, n) × N·L × fresnel ranges × specular. The phong mask comes from the Specular map and the
    exponent from the Roughness map, so scratches in the phong mask show only where the highlight is.
  - **Light Strength**, **Ambient** and **Phong Boost** are inputs on the group node.
- **Phong**: uses `$phongalbedotint` (the highlight is tinted by the albedo where the exponent texture's green
  channel says so), `$phongboost`, `$phongfresnelranges` and the exponent texture. `$rimlight` uses
  `$rimlightboost` and `$rimlightexponent`.
- **Viewport: TF2 Shading / PBR (Phong Shine)** (sidebar) switches between the two. The PBR material stays on the
  Cycles output. Switch to PBR before exporting with glTF or FBX.

**Phong Shine** (Clean PBR, default 0.5): scales the specular texture when it's baked. 0.5 caps it at the
standard 4% reflectance of a non-metal, so there's no white sheen. 1 allows up to 8%.

**Occlusion in the specular texture**: how much the `$detail` darkens each texel is baked into the specular texture,
as Source draws `$detail` over the phong. **Bake TF2 AO** with **AO Into Base Color** on does the same. Nothing is
done with nodes, so it exports as is.

**Bake TF2 AO** (its own button, sidebar and Output panel): bakes ambient occlusion for the selected model and
adds it to its baked PBR material. Nothing else bakes AO, and later re-bakes keep the AO image. Bake the material to
PBR first. The TF2 style is short-range occlusion through TF2's contrast curve, so crevices go nearly black and open
surfaces stay white, like TF2's own AO textures.
- **AO Distance** (Source units): how far it looks for occluders.
- **Import Scale**: the scale the model was imported at in SourceIO. **Auto** guesses it from the model's size.
- Only the model itself casts AO. Helper objects such as SourceIO's bone shapes, and anything else in the scene,
  are left out.

**Plain** is Blender's ordinary AO bake.

**Recipe** (Conversion panel):
- **Clean PBR** (default): strong phong becomes metal, everything else keeps the standard specular, and AO goes
  into the colour. It looks like the VLG original under PBR lighting.
- **Detailed Specular**: phong drives Specular Level (up to 4× normal) and Specular Tint, and `$rimlight` becomes
  Sheen. Shinier and brighter in Blender's lighting.

## Settings
They are per material, in Properties ▸ Material ▸ **VLG → PBR**:
- **Link .vmt File… / Read VMT / Read Node Values**, plus the texture slots and the VLG parameters
- **Conversion**: recipe, metal thresholds, specular intensity, max F0, matte where unmasked, roughness bias, lightwarp hue, rim → sheen,
  envmap → metallic
- **Output**:
  - **Use On Mesh**
  - **Resolution**
  - **AO**
  - **Save Textures To Disk**, which also writes PNGs (base colour, OpenGL/DirectX normals, roughness, metallic,
    specular, ORM and Unity/HDRP packs, plus an optional look .json) and makes the material use those files
