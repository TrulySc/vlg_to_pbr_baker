# VLG → PBR Baker 2.0 (Blender 4.2+, tested on 5.2)

Turns Source 1 **VertexLitGeneric** materials (as imported by SourceIO, or read from a `.vmt`) into ordinary
Blender **PBR materials**, in one click. The textures are baked and packed into the .blend.

## Install
Edit ▸ Preferences ▸ Get Extensions ▸ ▾ ▸ **Install from Disk…** → `vlg_to_pbr_baker.zip`. Remove the VLG → Deadlock Baker
first; this replaces it.

Optionally set the **Team Fortress 2 folder** in the add-on's preferences. Textures and lightwarps the import
didn't bring in are then pulled from the game's VPKs.

## One click
Select the model. In 3D View ▸ Sidebar (N) ▸ **VLG → PBR**, press **Bake Selected to PBR**.

For each VLG material `X` you get `X_PBR`, put on the mesh in its place:
- **Nodes**: a Principled BSDF with Base Color, Roughness, Metallic, Specular (from the phong) and a Normal Map
  (OpenGL). `$rimlight` becomes Sheen, `$selfillum` becomes Emission, and `$translucent`/`$alphatest` become Alpha.
- **Images**: baked and packed into the .blend, so there are no files to keep next to it.
- **Ambient occlusion**: baked with Cycles. It's kept as its own image, the way PBR engines expect, or multiplied
  into the base colour with **AO Into Base Color**.
- **The original**: `X` is kept. **Swap VLG ⇄ PBR** switches the selected objects between the two, and baking again
  re-bakes from `X`.

The VMT is read automatically the first time each material is baked. If the material's VertexLitGeneric group
takes the base or normal-map alpha from separate mask images (its "[basemap texture alpha]" / "[bumpmap texture
alpha]" inputs), those masks are used. They show as **Base Alpha** and **Normal Alpha** under Textures.

## Settings
They are per material, in Properties ▸ Material ▸ **VLG → PBR**:
- **Read VMT / Read Node Values / Load .vmt**, plus the texture slots and the VLG parameters
- **Conversion**: specular intensity, max F0, matte where unmasked, roughness bias, lightwarp hue, rim → sheen,
  envmap → metallic
- **Output**:
  - **Use On Mesh**
  - **Resolution**
  - **AO**
  - **Save Textures To Disk**, which also writes PNGs (base colour, OpenGL/DirectX normals, roughness, metallic,
    specular, ORM and Unity/HDRP packs, plus an optional look .json) and makes the material use those files
