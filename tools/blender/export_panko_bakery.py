"""Exports PankoBakery.blend as a static .glb the engine can load.

Run from the repo root:
    blender -b application/Assets/Scenes/PankoBakery/PankoBakery.blend \
        --python tools/blender/export_panko_bakery.py

The engine has no skinning, so Panko's current pose is baked into the meshes
and the rig is dropped. Point lights are exported (KHR_lights_punctual); the
sun is not, since the engine's sun is fixed in its shader. The .blend itself
is not saved. The scene camera is exported too; the engine starts the view
there (with its own fixed field of view). Load the result with
    HyperbolicTimeChamber --scene=Assets/Scenes/PankoBakery/PankoBakery.glb
"""

from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[2]
OUT_GLB = REPO / "application/Assets/Scenes/PankoBakery/PankoBakery.glb"


def bake_pose():
    for obj in list(bpy.data.objects):
        # Bone-parented props keep their posed place once the rig is gone
        if obj.parent is not None and obj.parent.type == "ARMATURE":
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
        if obj.type != "MESH":
            continue
        for mod in list(obj.modifiers):
            if mod.type == "ARMATURE":
                with bpy.context.temp_override(object=obj, active_object=obj):
                    bpy.ops.object.modifier_apply(modifier=mod.name)
    for obj in list(bpy.data.objects):
        if obj.type == "ARMATURE":
            bpy.data.objects.remove(obj)


def main():
    bake_pose()
    # The engine's sun and sky are fixed in raytrace.slang; only point lights are read
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and obj.data.type != "POINT":
            bpy.data.objects.remove(obj)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT_GLB),
        export_format="GLB",
        export_apply=True,
        export_skins=False,
        export_animations=False,
        export_cameras=True,
        export_lights=True,
        # Watts / (4 pi) as the glTF intensity, the same radiometric units the
        # engine's sun uses, rather than physical candela
        export_import_convert_lighting_mode="COMPAT",
        export_yup=True,
    )
    print(f"Exported {OUT_GLB}")


main()
