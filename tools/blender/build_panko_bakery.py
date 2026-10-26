"""Builds the Panko bakery storefront scene as a .blend file.

Run from the repo root:
    blender -b --factory-startup --python tools/blender/build_panko_bakery.py

The scene only uses what the engine's ray tracer understands: triangle meshes
with base colour factors (plus Panko's base colour texture) and one sun. The
sun and sky copy the constants in engine/shaders/common/raytrace.slang, so the
Cycles preview shows roughly what the engine will draw. Panko stays rigged so
the pose can be changed in the .blend; export_panko_bakery.py bakes it.
"""

import math
import random

import numpy as np
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

REPO = Path(__file__).resolve().parents[2]
PANKO_GLB = REPO / "application/Assets/Models/Panko/PANKO_Rigged.glb"
OUT_BLEND = REPO / "application/Assets/Scenes/PankoBakery/PankoBakery.blend"
# Low Poly Bakery Pack by reodesuxz. Its license forbids redistributing the
# models themselves, so they are read from the local asset library, not the repo.
BAKERY_PACK = Path.home() / "3d-game-assets/LowPolyBakeryPack_v1.0/GLB"
# The pack is in metres; this scene is about 2.5 units per metre (Panko is 3.6 tall)
PACK_SCALE = 2.5
# Stylized Forest Kit by Asset Quest (CC0)
FOREST_KIT = Path.home() / "3d-game-assets/Forest_Kit_AssetQuest"
STREET_TREE = "Oak_Tree_2"
# Smaller than real scale so the canopy fits under the frame top: about 12.5 units tall
TREE_SCALE = 1.45

# Blender is Z-up, glTF is Y-up: glTF (x, y, z) -> Blender (x, -z, y).
# raytrace.slang: lightDir = normalize(float3(0.7, 0.35, 0.6)) points toward the sun.
SUN_TOWARD = Vector((0.7, -0.6, 0.35)).normalized()
SUN_COLOR = (1.0, 0.6, 0.3)
SUN_IRRADIANCE = 6.0
SKY_GROUND = (0.3, 0.2, 0.18)
SKY_HORIZON = (1.0, 0.55, 0.35)
SKY_ZENITH = (0.3, 0.36, 0.65)

# Street lamp: warm yellow. LAMP_POWER is in watts; Cycles and the engine both
# treat a point light as intensity = power / (4 pi) per steradian.
LAMP_COLOR = (1.0, 0.75, 0.35)
LAMP_POWER = 2000.0
LAMP_GLOW = 2.5  # Emission strength of the lamp head

# Storefront layout, in Blender units. Panko is about 3.6 tall.
WALL_FRONT_Y = 3.0
WALL_BACK_Y = 3.6
FACADE_X = (-8.0, 8.0)
FACADE_HEIGHT = 14.0
DOOR_X = (-5.5, -3.0)
DOOR_TOP = 5.2
WINDOW_X = (1.0, 7.0)
WINDOW_Z = (1.5, 5.5)
SIDEWALK_FRONT_Y = -4.0
AWNING_X = (-6.0, 7.5)
AWNING_TOP_Z, AWNING_LOW_Z, AWNING_FRONT_Y = 7.0, 5.9, 0.3
TREE_X, TREE_Y = -9.8, -2.9
STREET_Z = -0.35


# --- Helpers ---------------------------------------------------------------

_materials = {}


def material(name, color, roughness=0.9):
    if name not in _materials:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        mat.diffuse_color = (*color, 1.0)
        _materials[name] = mat
    return _materials[name]


def collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def link(obj, coll):
    for old in obj.users_collection:
        old.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def mesh_object(name, bm, mat, coll):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    return obj


def add_box(bm, lo, hi, matrix=None):
    """Adds an axis-aligned box from corner lo to corner hi, then transforms it."""
    lo, hi = Vector(lo), Vector(hi)
    geom = bmesh.ops.create_cube(bm, size=1.0)
    verts = geom["verts"]
    bmesh.ops.scale(bm, vec=hi - lo, verts=verts)
    bmesh.ops.translate(bm, vec=(lo + hi) / 2, verts=verts)
    if matrix is not None:
        bmesh.ops.transform(bm, matrix=matrix, verts=verts)


def box(name, lo, hi, mat, coll, bevel=0.0):
    bm = bmesh.new()
    add_box(bm, lo, hi)
    obj = mesh_object(name, bm, mat, coll)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
    return obj


def boxes(name, specs, mat, coll):
    """Many boxes in one object: specs are (lo, hi) or (lo, hi, matrix)."""
    bm = bmesh.new()
    for spec in specs:
        add_box(bm, *spec)
    return mesh_object(name, bm, mat, coll)


def ellipsoid(name, center, radii, mat, coll, rotation=(0, 0, 0), segments=24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=1.0)
    obj = mesh_object(name, bm, mat, coll)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.location = center
    obj.scale = radii
    obj.rotation_euler = rotation
    return obj


def cylinder(name, center, radius, depth, mat, coll, rotation=(0, 0, 0), segments=24):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments,
                          radius1=radius, radius2=radius, depth=depth)
    obj = mesh_object(name, bm, mat, coll)
    obj.location = center
    obj.rotation_euler = rotation
    return obj


def cut(target, cutter_lo, cutter_hi):
    """Boolean-subtracts a box from target and applies it."""
    bm = bmesh.new()
    add_box(bm, cutter_lo, cutter_hi)
    mesh = bpy.data.meshes.new("Cutter")
    bm.to_mesh(mesh)
    bm.free()
    cutter = bpy.data.objects.new("Cutter", mesh)
    bpy.context.scene.collection.objects.link(cutter)
    mod = target.modifiers.new("Cut", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    with bpy.context.temp_override(object=target, active_object=target):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter)


def text(name, body, location, size, mat, coll, rotation=(math.radians(90), 0, 0), extrude=0.04):
    """Text converted to a mesh, since glTF has no text."""
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.size = size
    curve.extrude = extrude
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    temp = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(temp)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(temp.evaluated_get(depsgraph))
    bpy.data.objects.remove(temp)
    bpy.data.curves.remove(curve)
    mesh.materials.clear()
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    return obj


# --- Scene parts -----------------------------------------------------------

def build_world():
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    # Same gradient as Sky() in raytrace.slang, which lerps over direction.z:
    # ground at z <= -0.25, horizon at 0, zenith at z >= 0.5. With factor
    # z * 0.5 + 0.5 those are ramp stops 0.375, 0.5 and 0.75.
    coords = nodes.new("ShaderNodeTexCoord")
    split = nodes.new("ShaderNodeSeparateXYZ")
    remap = nodes.new("ShaderNodeMapRange")
    remap.inputs["From Min"].default_value = -1.0
    ramp = nodes.new("ShaderNodeValToRGB")
    stops = ramp.color_ramp.elements
    stops[0].position, stops[0].color = 0.375, (*SKY_GROUND, 1.0)
    stops[1].position, stops[1].color = 0.75, (*SKY_ZENITH, 1.0)
    horizon = stops.new(0.5)
    horizon.color = (*SKY_HORIZON, 1.0)
    links.new(coords.outputs["Generated"], split.inputs[0])
    links.new(split.outputs["Z"], remap.inputs["Value"])
    links.new(remap.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Background"].inputs["Color"])

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    scene.collection.objects.link(sun)
    sun.data.energy = SUN_IRRADIANCE
    sun.data.color = SUN_COLOR
    sun.data.angle = math.radians(0.5)
    # A sun light shines along its local -Z, so point +Z at the sun
    sun.rotation_euler = SUN_TOWARD.to_track_quat("Z", "Y").to_euler()
    sun.location = (0, 0, 20)


def build_ground(coll):
    sidewalk = material("Sidewalk", (0.11, 0.11, 0.12))
    curb = material("Curb", (0.2, 0.2, 0.21))
    street = material("StreetBase", (0.18, 0.17, 0.17))
    box("Sidewalk", (-20, SIDEWALK_FRONT_Y, -0.2), (20, 12, 0), sidewalk, coll)
    box("Curb", (-20, SIDEWALK_FRONT_Y - 0.3, STREET_Z), (20, SIDEWALK_FRONT_Y + 0.05, 0.02), curb, coll)
    box("Street", (-20, -24, STREET_Z - 0.2), (20, SIDEWALK_FRONT_Y - 0.3, STREET_Z), street, coll)

    # Sidewalk paving joints
    joints = material("SidewalkJoint", (0.05, 0.05, 0.055))
    specs = [((x - 0.03, SIDEWALK_FRONT_Y, -0.19), (x + 0.03, WALL_FRONT_Y, 0.005))
             for x in range(-20, 21, 2)]
    specs.append(((-20, -0.53, -0.19), (20, -0.47, 0.005)))
    boxes("SidewalkJoints", specs, joints, coll)

    # Cobblestones, offset every other row, in three shades
    shades = [material(f"Cobble{i}", c) for i, c in
              enumerate([(0.38, 0.36, 0.34), (0.3, 0.29, 0.28), (0.45, 0.42, 0.38)])]
    stones = [[] for _ in shades]
    size, gap = 0.7, 0.08
    for row in range(26):
        y = SIDEWALK_FRONT_Y - 0.35 - size / 2 - row * size
        offset = (row % 2) * size / 2
        for col in range(-30, 30):
            x = col * size + offset
            # Cheap deterministic variation
            h = (col * 7 + row * 13) % 5
            top = STREET_Z + 0.06 + 0.015 * (h % 3)
            stones[(col * 5 + row * 3) % 3].append(
                ((x - size / 2 + gap, y - size / 2 + gap, STREET_Z - 0.05),
                 (x + size / 2 - gap, y + size / 2 - gap, top)))
    for i, (mat, specs) in enumerate(zip(shades, stones)):
        obj = boxes(f"Cobblestones{i}", specs, mat, coll)
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = 0.05
        mod.segments = 2


def build_storefront(coll):
    plaster = material("Plaster", (0.88, 0.76, 0.58))
    wall = box("Facade", (FACADE_X[0], WALL_FRONT_Y, 0), (FACADE_X[1], WALL_BACK_Y, FACADE_HEIGHT), plaster, coll)
    cut(wall, (DOOR_X[0], WALL_FRONT_Y - 1, -1), (DOOR_X[1], WALL_BACK_Y + 1, DOOR_TOP))
    cut(wall, (WINDOW_X[0], WALL_FRONT_Y - 1, WINDOW_Z[0]), (WINDOW_X[1], WALL_BACK_Y + 1, WINDOW_Z[1]))
    for x0, x1 in [(-5.0, -2.0), (2.0, 5.0)]:
        cut(wall, (x0, WALL_FRONT_Y - 1, 9.8), (x1, WALL_BACK_Y + 1, 12.3))

    # Brick base band and cornice
    brick = material("Brick", (0.55, 0.25, 0.17))
    base = box("BrickBase", (FACADE_X[0] - 0.1, WALL_FRONT_Y - 0.12, 0), (FACADE_X[1] + 0.1, WALL_FRONT_Y, 1.3), brick, coll)
    cut(base, (DOOR_X[0], WALL_FRONT_Y - 1, -1), (DOOR_X[1], WALL_BACK_Y, DOOR_TOP))
    trim = material("Trim", (0.95, 0.93, 0.88))
    box("Cornice", (FACADE_X[0] - 0.3, WALL_FRONT_Y - 0.4, FACADE_HEIGHT - 0.6), (FACADE_X[1] + 0.3, WALL_BACK_Y, FACADE_HEIGHT), trim, coll, bevel=0.05)
    box("Ledge", (FACADE_X[0] - 0.1, WALL_FRONT_Y - 0.2, 9.2), (FACADE_X[1] + 0.1, WALL_FRONT_Y, 9.4), trim, coll)

    # Shop interior behind the openings, so they read as depth rather than holes
    interior = material("Interior", (0.8, 0.66, 0.5))
    box("InteriorBack", (FACADE_X[0], 6.5, 0), (FACADE_X[1], 6.8, FACADE_HEIGHT), interior, coll)
    box("InteriorFloor", (FACADE_X[0], WALL_BACK_Y, 0), (FACADE_X[1], 6.5, 0.05), material("InteriorFloor", (0.5, 0.35, 0.22)), coll)
    box("InteriorCeiling", (FACADE_X[0], WALL_BACK_Y, 6.0), (FACADE_X[1], 6.5, 6.3), interior, coll)
    box("UpstairsFloor", (FACADE_X[0], WALL_BACK_Y, 9.5), (FACADE_X[1], 6.5, 9.7), interior, coll)

    build_door(coll)
    build_display_window(coll)
    build_upstairs_windows(coll)
    build_awning(coll)
    build_sign(coll)


def build_door(coll):
    green = material("DoorPaint", (0.38, 0.1, 0.09))
    frame = material("Trim", None)
    glass = material("DoorGlass", (0.15, 0.2, 0.24), roughness=0.3)
    gold = material("Brass", (0.8, 0.6, 0.25), roughness=0.4)
    y = WALL_FRONT_Y + 0.35
    x0, x1 = DOOR_X
    door = box("Door", (x0, y, 0), (x1, y + 0.12, DOOR_TOP), green, coll)
    cut(door, (x0 + 0.4, y - 1, 2.6), (x1 - 0.4, y + 1, DOOR_TOP - 0.4))
    box("DoorGlass", (x0 + 0.4, y + 0.05, 2.6), (x1 - 0.4, y + 0.08, DOOR_TOP - 0.4), glass, coll)
    boxes("DoorPanels", [((x0 + 0.4, y - 0.04, 0.5), (x1 - 0.4, y, 2.2))], green, coll)
    boxes("DoorFrame", [
        ((x0 - 0.25, WALL_FRONT_Y - 0.12, 0), (x0, WALL_FRONT_Y + 0.1, DOOR_TOP + 0.25)),
        ((x1, WALL_FRONT_Y - 0.12, 0), (x1 + 0.25, WALL_FRONT_Y + 0.1, DOOR_TOP + 0.25)),
        ((x0 - 0.25, WALL_FRONT_Y - 0.12, DOOR_TOP), (x1 + 0.25, WALL_FRONT_Y + 0.1, DOOR_TOP + 0.25)),
    ], frame, coll)
    ellipsoid("DoorKnob", (x1 - 0.35, y - 0.08, 2.5), (0.1, 0.1, 0.1), gold, coll, segments=12)
    box("DoorStep", (x0 - 0.3, WALL_FRONT_Y - 0.6, 0), (x1 + 0.3, WALL_FRONT_Y + 0.35, 0.15), material("Curb", None), coll)


def build_display_window(coll):
    frame = material("Trim", None)
    wood = material("ShelfWood", (0.62, 0.42, 0.24))
    x0, x1 = WINDOW_X
    z0, z1 = WINDOW_Z
    yf = WALL_FRONT_Y - 0.12
    # No glass or mullions: the ray tracer has no transmission, so the display stays open
    boxes("WindowFrame", [
        ((x0 - 0.25, yf, z0 - 0.3), (x1 + 0.25, WALL_FRONT_Y + 0.2, z0)),  # sill
        ((x0 - 0.25, yf, z1), (x1 + 0.25, WALL_FRONT_Y + 0.2, z1 + 0.25)),
        ((x0 - 0.25, yf, z0), (x0, WALL_FRONT_Y + 0.2, z1)),
        ((x1, yf, z0), (x1 + 0.25, WALL_FRONT_Y + 0.2, z1)),
    ], frame, coll)

    # Three shelves just inside the opening, close enough to the street for daylight
    front, back = WALL_FRONT_Y + 0.25, WALL_BACK_Y + 1.0
    shelf_tops = [z0 + 0.1, z0 + 1.25, z0 + 2.4]
    boxes("DisplayShelves", [((x0, front, z - 0.08), (x1, back, z)) for z in shelf_tops], wood, coll)
    box("DisplayBackdrop", (x0, back, z0), (x1, back + 0.1, z1), material("Backdrop", (0.85, 0.72, 0.52)), coll)
    mid = (front + back) / 2

    bottom, middle, top = shelf_tops
    turn = math.radians
    # Pastries a bit larger than life so they read from the street; front and back rows
    big = 1.3
    fy, by = mid - 0.33, mid + 0.3

    # Bottom shelf: bread basket, tray, pie, layer cake, with donuts along the front
    pack("bread_basket", "DisplayBasket", (1.75, mid, bottom), coll)
    pack("baking_tray", "DisplayTray", (3.25, by, bottom), coll)
    pack("pie", "DisplayPie", (4.7, by, bottom), coll, scale=big)
    pack("layer_cake", "DisplayCake", (6.1, mid, bottom), coll, yaw=turn(20), scale=1.2)
    for i in range(5):
        pack("donut", f"FrontDonut{i}", (2.75 + 0.5 * i, fy, bottom), coll, yaw=turn(35 * i), scale=big)

    # Middle shelf: cupcake stand, loaves behind croissants, more cupcakes
    pack("cake_stand", "CupcakeStand", (1.7, mid, middle), coll, scale=1.2)
    for i, (dx, dy) in enumerate([(-0.22, -0.18), (0.22, -0.18), (0.0, 0.22)]):
        pack("cupcake", f"StandCupcake{i}", (1.7 + dx, mid + dy, middle + 0.72), coll, scale=big)
    for i, x in enumerate((3.0, 4.05, 5.1)):
        pack("bread_loaf", f"ShelfLoaf{i}", (x, by, middle), coll, yaw=turn(6 - 6 * i), scale=big)
        pack("croissant", f"ShelfCroissant{i}", (x - 0.1, fy, middle), coll, yaw=turn(-12 + 12 * i), scale=big)
    for i, (dx, dy) in enumerate([(-0.18, -0.25), (0.18, -0.25), (-0.18, 0.15), (0.18, 0.15)]):
        pack("cupcake", f"ShelfCupcake{i}", (6.3 + dx, mid + dy, middle), coll, scale=big)

    # Top shelf: a pile of baguettes, pies, croissants, and a second cake
    for i, (dx, dy, dz) in enumerate([(0, -0.2, 0), (0, 0.2, 0), (0.1, 0.0, 0.35)]):
        pack("baguette", f"ShelfBaguette{i}", (2.3 + dx, by - 0.1 + dy, top + dz), coll, yaw=turn(3 - 3 * i), scale=big)
    for i, x in enumerate((1.75, 2.85)):
        pack("croissant", f"TopCroissantFront{i}", (x, fy - 0.05, top), coll, yaw=turn(15 - 30 * i), scale=big)
    pack("pie", "TopPie", (4.15, by, top), coll, yaw=turn(45), scale=big)
    for i, x in enumerate((3.85, 4.55)):
        pack("donut", f"TopDonut{i}", (x, fy, top), coll, yaw=turn(50 * i), scale=big)
    pack("layer_cake", "TopCake", (5.85, mid, top), coll, yaw=turn(-30), scale=1.1)

    # String of bulbs sagging across the top of the window
    bulbs = material("Bulb", (1.0, 0.88, 0.55), roughness=0.3)
    wire = material("Iron", (0.08, 0.08, 0.08), roughness=0.5)
    points = []
    spans = 3
    for i in range(spans * 8 + 1):
        t = i / (spans * 8)
        x = x0 + 0.1 + t * (x1 - x0 - 0.2)
        sag = 0.25 * math.sin(math.pi * ((t * spans) % 1.0))
        points.append(Vector((x, WALL_FRONT_Y + 0.4, z1 - 0.1 - sag)))
    for i, (a, b) in enumerate(zip(points, points[1:])):
        direction = b - a
        cylinder(f"LightWire{i}", (a + b) / 2, 0.012, direction.length, wire, coll,
                 rotation=direction.to_track_quat("Z", "Y").to_euler(), segments=6)
    for i, point in enumerate(points[1:-1:2]):
        ellipsoid(f"Bulb{i}", point - Vector((0, 0, 0.07)), (0.05, 0.05, 0.07), bulbs, coll, segments=10)


def build_upstairs_windows(coll):
    frame = material("Trim", None)
    shutter = material("DoorPaint", None)
    flowers = material("Mums", (0.92, 0.55, 0.1))
    leaves = material("Leaves", (0.25, 0.35, 0.15))
    for i, (x0, x1) in enumerate([(-5.0, -2.0), (2.0, 5.0)]):
        boxes(f"UpstairsFrame{i}", [
            ((x0 - 0.2, WALL_FRONT_Y - 0.1, 9.6), (x1 + 0.2, WALL_FRONT_Y + 0.1, 9.8)),
            ((x0 - 0.2, WALL_FRONT_Y - 0.1, 12.3), (x1 + 0.2, WALL_FRONT_Y + 0.1, 12.5)),
            (((x0 + x1) / 2 - 0.05, WALL_FRONT_Y + 0.1, 9.8), ((x0 + x1) / 2 + 0.05, WALL_FRONT_Y + 0.2, 12.3)),
        ], frame, coll)
        boxes(f"Shutters{i}", [
            ((x0 - 1.4, WALL_FRONT_Y - 0.08, 9.8), (x0 - 0.1, WALL_FRONT_Y, 12.3)),
            ((x1 + 0.1, WALL_FRONT_Y - 0.08, 9.8), (x1 + 1.4, WALL_FRONT_Y, 12.3)),
        ], shutter, coll)
        box(f"FlowerBox{i}", (x0, WALL_FRONT_Y - 0.6, 9.4), (x1, WALL_FRONT_Y, 9.9), material("ShelfWood", None), coll)
        for j in range(5):
            x = x0 + 0.3 + j * (x1 - x0 - 0.6) / 4
            ellipsoid(f"Leaves{i}_{j}", (x, WALL_FRONT_Y - 0.3, 10.0), (0.38, 0.3, 0.3), leaves, coll, segments=12)
            ellipsoid(f"Flower{i}_{j}", (x + 0.1, WALL_FRONT_Y - 0.45, 10.25), (0.14, 0.14, 0.14), flowers, coll, segments=10)


def build_awning(coll):
    red = material("AwningOrange", (0.8, 0.36, 0.08))
    white = material("AwningWhite", (0.95, 0.93, 0.88))
    x0, x1 = AWNING_X
    top_z, low_z, front_y = AWNING_TOP_Z, AWNING_LOW_Z, AWNING_FRONT_Y
    valance = 0.45
    stripe = 0.75
    slope = Vector((0, front_y - WALL_FRONT_Y, low_z - top_z))
    # Shear a unit-depth box onto the awning slope: y in [0,1] maps along the slope
    shear = Matrix(((1, 0, 0, 0),
                    (0, slope.y, 0, WALL_FRONT_Y),
                    (0, slope.z, 1, top_z),
                    (0, 0, 0, 1)))
    stripes = [[], []]
    count = round((x1 - x0) / stripe)
    for i in range(count):
        sx0, sx1 = x0 + i * stripe, x0 + (i + 1) * stripe
        stripes[i % 2].append(((sx0, 0, -0.05), (sx1, 1, 0), shear))
        # Hanging valance with a rounded tip under every stripe
        stripes[i % 2].append(((sx0, front_y - 0.05, low_z - valance), (sx1, front_y, low_z)))
    boxes("AwningOrange", stripes[0], red, coll)
    boxes("AwningWhite", stripes[1], white, coll)
    for i in range(count):
        x = x0 + (i + 0.5) * stripe
        ellipsoid(f"ValanceTip{i}", (x, front_y - 0.025, low_z - valance), (stripe / 2, 0.025, 0.2),
                        red if i % 2 == 0 else white, coll, segments=16)
    # Side arms holding the awning out
    metal = material("Iron", (0.08, 0.08, 0.08), roughness=0.5)
    for x in (x0 + 0.1, x1 - 0.1):
        length = slope.length
        mid = Vector((x, WALL_FRONT_Y, top_z - 0.15)) + slope / 2
        arm = cylinder("AwningArm", mid, 0.04, length, metal, coll)
        arm.rotation_euler = slope.to_track_quat("Z", "Y").to_euler()


def build_sign(coll):
    wood = material("SignWood", (0.3, 0.17, 0.09))
    cream = material("SignLetters", (0.98, 0.9, 0.7))
    box("SignBoard", (-4.6, WALL_FRONT_Y - 0.2, 7.5), (4.6, WALL_FRONT_Y, 9.0), wood, coll, bevel=0.04)
    text("SignText", "Panko's Bakery", (0, WALL_FRONT_Y - 0.2, 8.25), 1.0, cream, coll)


def build_neighbors(coll):
    window = material("NeighborWindow", (0.12, 0.14, 0.18), roughness=0.3)
    frame = material("Trim", None)
    for name, x0, x1, height, color in [
        ("NeighborLeft", -20.0, FACADE_X[0], 12.0, (0.78, 0.6, 0.25)),
        ("NeighborRight", FACADE_X[1], 20.0, 16.0, (0.47, 0.5, 0.3)),
    ]:
        box(name, (x0, WALL_FRONT_Y + 0.5, 0), (x1, 12, height), material(name, color), coll)
        y = WALL_FRONT_Y + 0.5
        specs, frames = [], []
        for wz in range(2, int(height) - 2, 4):
            for wx in (x0 + 2.5, x1 - 3.5):
                specs.append(((wx, y - 0.05, wz), (wx + 1.6, y, wz + 2.4)))
                frames.append(((wx - 0.15, y - 0.12, wz - 0.2), (wx + 1.75, y, wz)))
        boxes(name + "Windows", specs, window, coll)
        boxes(name + "Sills", frames, frame, coll)


# --- Props -----------------------------------------------------------------

_pack_meshes = {}


def pack_mesh(asset):
    """Imports one Low Poly Bakery Pack model once and returns its mesh.
    The pack's materials are shared between models, so they're merged by name."""
    if asset not in _pack_meshes:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(BAKERY_PACK / f"{asset}.glb"))
        (obj,) = set(bpy.data.objects) - before
        mesh = obj.data
        bpy.data.objects.remove(obj)
        for i, mat in enumerate(mesh.materials):
            name = "Pack_" + mat.name.split(".")[0]
            existing = bpy.data.materials.get(name)
            if existing is None:
                mat.name = name
            elif existing != mat:
                mesh.materials[i] = existing
                bpy.data.materials.remove(mat)
        mesh.name = asset
        _pack_meshes[asset] = mesh
    return _pack_meshes[asset]


def pack(asset, name, location, coll, yaw=0.0, scale=1.0, rotation=None):
    """Places a pack model with its base centre at location."""
    obj = bpy.data.objects.new(name, pack_mesh(asset))
    coll.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation if rotation is not None else (0, 0, yaw)
    obj.scale = (PACK_SCALE * scale,) * 3
    return obj


def build_props(coll):
    wood = material("SignFrameWood", (0.6, 0.42, 0.25))
    chalk = material("Chalkboard", (0.08, 0.1, 0.09))
    chalk_text = material("Chalk", (0.92, 0.92, 0.9))

    # A-frame chalkboard sign by the door: two boards leaning together at the top
    ax, ay = -6.8, -0.9
    lean = math.radians(14)
    height = 2.4
    spread = height * math.sin(lean)
    for side, name in ((-1, "Front"), (1, "Back")):
        # side -1 stands toward the street and leans back (top toward +Y)
        # Rotating +lean about X tips the top toward -Y
        tilt = Matrix.Rotation(side * lean, 4, "X")
        outer = Matrix.Translation((ax, ay + side * spread, 0)) @ tilt
        boxes(f"ChalkFrame{name}", [((-0.7, -0.05, 0), (0.7, 0.05, height), outer)], wood, coll)
        boxes(f"ChalkBoard{name}", [((-0.6, side * 0.06 - 0.01, 0.5), (0.6, side * 0.06 + 0.01, 2.3), outer)], chalk, coll)
    front = Matrix.Rotation(-lean, 3, "X")
    text("ChalkText", "FRESH\nBREAD", Vector((ax, ay - spread, 0)) + front @ Vector((0, -0.08, 1.45)),
         0.32, chalk_text, coll, rotation=(math.radians(90) - lean, 0, 0), extrude=0.005)

    # Potted shrubs either side of the door
    terracotta = material("Terracotta", (0.7, 0.35, 0.2))
    foliage = material("ShrubFoliage", (0.6, 0.2, 0.08))
    for i, x in enumerate((DOOR_X[0] - 0.9, DOOR_X[1] + 0.9)):
        cylinder(f"Pot{i}", (x, WALL_FRONT_Y - 0.7, 0.4), 0.45, 0.8, terracotta, coll)
        ellipsoid(f"Shrub{i}", (x, WALL_FRONT_Y - 0.7, 1.35), (0.6, 0.6, 0.75), foliage, coll, segments=16)

    # Street lamp at the curb: a glowing head, and a point light just under it
    # (inside the closed head its own walls would block every shadow ray)
    iron = material("Iron", None)
    lamp = material("LampGlass", LAMP_COLOR, roughness=0.3)
    glow = lamp.node_tree.nodes["Principled BSDF"]
    glow.inputs["Emission Color"].default_value = (*LAMP_COLOR, 1.0)
    glow.inputs["Emission Strength"].default_value = LAMP_GLOW
    lx, ly = 9.5, SIDEWALK_FRONT_Y + 0.8
    cylinder("LampBase", (lx, ly, 0.3), 0.3, 0.6, iron, coll)
    cylinder("LampPole", (lx, ly, 3.8), 0.1, 7.0, iron, coll, segments=16)
    box("LampHead", (lx - 0.35, ly - 0.35, 7.3), (lx + 0.35, ly + 0.35, 8.2), lamp, coll)
    box("LampCap", (lx - 0.5, ly - 0.5, 8.2), (lx + 0.5, ly + 0.5, 8.4), iron, coll)
    light = bpy.data.objects.new("LampLight", bpy.data.lights.new("LampLight", "POINT"))
    coll.objects.link(light)
    light.location = (lx, ly, 7.15)
    light.data.color = LAMP_COLOR
    light.data.energy = LAMP_POWER
    light.data.shadow_soft_size = 0.0  # The engine's point lights have no size


# --- Autumn ----------------------------------------------------------------

def leaf_materials():
    return [material(name, color) for name, color in [
        ("LeafOrange", (0.85, 0.38, 0.06)),
        ("LeafRed", (0.62, 0.12, 0.05)),
        ("LeafYellow", (0.9, 0.65, 0.12)),
        ("LeafBrown", (0.45, 0.25, 0.1)),
    ]]


def add_leaf(bm, location, normal, rng):
    """A thin oval lying on a surface with the given normal."""
    size = rng.uniform(0.2, 0.32)
    shape = Matrix.Diagonal((size, size * 0.55, 0.012, 1.0))
    # Leaves curl, so none lie perfectly flat
    tilt = Matrix.Rotation(rng.uniform(-0.25, 0.25), 4, "X")
    spin = Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, "Z")
    align = normal.to_track_quat("Z", "Y").to_matrix().to_4x4()
    lift = Matrix.Translation(location + normal * 0.015)
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, radius=1.0,
                              matrix=lift @ align @ spin @ tilt @ shape)


def scatter_leaves(name, count, sample, coll, seed):
    """sample(rng) returns (location, normal) for one leaf."""
    rng = random.Random(seed)
    mats = leaf_materials()
    meshes = [bmesh.new() for _ in mats]
    for _ in range(count):
        location, normal = sample(rng)
        add_leaf(meshes[rng.randrange(len(mats))], location, normal, rng)
    for mat, bm in zip(mats, meshes):
        mesh_object(f"{name}{mat.name}", bm, mat, coll)


def build_fallen_leaves(coll):
    up = Vector((0, 0, 1))

    def sidewalk(rng):
        # Denser drifts along the wall and under the tree
        roll = rng.random()
        if roll < 0.3:
            x, y = rng.uniform(-16, 16), rng.uniform(WALL_FRONT_Y - 1.2, WALL_FRONT_Y - 0.2)
        elif roll < 0.55:
            x, y = rng.gauss(TREE_X, 2.5), rng.gauss(TREE_Y, 1.8)
            y = min(max(y, SIDEWALK_FRONT_Y + 0.1), WALL_FRONT_Y - 0.2)
        else:
            x, y = rng.uniform(-16, 16), rng.uniform(SIDEWALK_FRONT_Y + 0.1, WALL_FRONT_Y - 0.2)
        return Vector((x, y, 0.0)), up

    def street(rng):
        # Most collect in the gutter against the curb
        y = SIDEWALK_FRONT_Y - 0.4 - abs(rng.gauss(0, 1.5)) if rng.random() < 0.6 \
            else rng.uniform(-20, SIDEWALK_FRONT_Y - 0.4)
        return Vector((rng.uniform(-18, 18), y, STREET_Z + 0.09)), up

    scatter_leaves("SidewalkLeaves", 420, sidewalk, coll, seed=1)
    scatter_leaves("StreetLeaves", 260, street, coll, seed=2)


def build_awning_leaves(coll):
    top_z, low_z, front_y = AWNING_TOP_Z, AWNING_LOW_Z, AWNING_FRONT_Y
    slope = Vector((0, front_y - WALL_FRONT_Y, low_z - top_z))
    normal = Vector((0, -slope.z, slope.y)).normalized()
    if normal.z < 0:
        normal = -normal

    def sample(rng):
        # Leaves slide down and gather toward the front edge
        t = 1.0 - abs(rng.gauss(0, 0.35)) % 1.0
        x = rng.uniform(AWNING_X[0] + 0.2, AWNING_X[1] - 0.2)
        return Vector((x, WALL_FRONT_Y, top_z)) + slope * t, normal

    scatter_leaves("AwningLeaves", 90, sample, coll, seed=3)


def autumn_atlas(image):
    """Copy of the Forest Kit leaf atlas with its greens turned to fall colours:
    light leaves go yellow, dark ones red. Alpha is kept for the leaf cut-outs."""
    width, height = image.size
    px = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    high, low = px[:, :3].max(1), px[:, :3].min(1)
    span = high - low + 1e-6
    hue = np.where(high == r, ((g - b) / span) % 6,
                   np.where(high == g, (b - r) / span + 2, (r - g) / span + 4)) / 6
    saturation = span / (high + 1e-6)
    leaf = (hue > 0.12) & (hue < 0.5) & (px[:, 3] > 0.01)

    lightness = np.clip((high - 0.15) / 0.55, 0, 1)
    new_hue = 0.01 + 0.11 * lightness
    new_sat = np.clip(0.7 + 0.3 * saturation, 0, 1)
    new_val = np.clip(high * 1.35 + 0.1, 0, 1)
    # HSV -> RGB
    sector = np.floor(new_hue * 6).astype(int) % 6
    f = new_hue * 6 - np.floor(new_hue * 6)
    p = new_val * (1 - new_sat)
    q = new_val * (1 - f * new_sat)
    t = new_val * (1 - (1 - f) * new_sat)
    v = new_val
    rgb = np.choose(sector[:, None], [np.stack(c, 1) for c in
                                      [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]])
    px[leaf, :3] = rgb[leaf]

    autumn = bpy.data.images.new("Forest_Pack_Atlas_Autumn", width, height, alpha=True)
    autumn.pixels.foreach_set(px.ravel())
    autumn.file_format = "PNG"
    autumn.pack()
    return autumn


def build_tree(coll):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(FOREST_KIT / "Meshes" / f"{STREET_TREE}.fbx"))
    (tree,) = set(bpy.data.objects) - before
    link(tree, coll)
    tree.name = "StreetTree"
    tree.location = (TREE_X, TREE_Y, 0)
    tree.scale = tree.scale * TREE_SCALE
    for mat in tree.data.materials:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Roughness"].default_value = 0.9
        if not bsdf.inputs["Alpha"].is_linked:
            continue
        # Leaves: swap in the autumn atlas, and round the alpha so the glTF
        # exporter writes alphaMode MASK (the engine discards masked texels;
        # blended materials would render as opaque cards)
        alpha_link = bsdf.inputs["Alpha"].links[0]
        source = alpha_link.from_socket
        links.remove(alpha_link)
        clip = nodes.new("ShaderNodeMath")
        clip.operation = "ROUND"
        links.new(source, clip.inputs[0])
        links.new(clip.outputs[0], bsdf.inputs["Alpha"])
        textures = [n for n in nodes if n.type == "TEX_IMAGE"]
        autumn = autumn_atlas(textures[0].image)
        for texture in textures:
            texture.image = autumn
    box("TreeGrate", (TREE_X - 0.9, TREE_Y - 0.9, -0.01), (TREE_X + 0.9, TREE_Y + 0.9, 0.02),
        material("Iron", None), coll)


def pumpkin(name, location, size, coll):
    skin = material("Pumpkin", (0.88, 0.42, 0.08))
    stem = material("PumpkinStem", (0.3, 0.3, 0.12))
    x, y, z = location
    # Ribs: tall ellipsoids around the middle
    for i in range(6):
        a = i * math.pi / 3
        ellipsoid(f"{name}_{i}", (x + math.cos(a) * size * 0.35, y + math.sin(a) * size * 0.35, z + size * 0.6),
                  (size * 0.6, size * 0.6, size * 0.6), skin, coll, rotation=(0, 0, a), segments=14)
    cylinder(f"{name}Stem", (x, y, z + size * 1.25), size * 0.08, size * 0.35, stem, coll,
             rotation=(0.2, 0, 0), segments=8)


def build_pumpkins(coll):
    y = WALL_FRONT_Y - 0.9
    pumpkin("PumpkinA", (DOOR_X[0] - 1.9, y + 0.2, 0), 0.55, coll)
    pumpkin("PumpkinB", (DOOR_X[0] - 0.4, y - 0.7, 0), 0.38, coll)
    pumpkin("PumpkinC", (DOOR_X[1] + 1.7, y + 0.1, 0), 0.42, coll)


# --- Panko -----------------------------------------------------------------

def rotate_bone(rig, bone_name, axis, degrees):
    """Rotates a pose bone about a world axis through its head."""
    pose_bone = rig.pose.bones[bone_name]
    head = pose_bone.head.copy()
    rotation = Matrix.Rotation(math.radians(degrees), 4, axis)
    pose_bone.matrix = Matrix.Translation(head) @ rotation @ Matrix.Translation(-head) @ pose_bone.matrix
    bpy.context.view_layer.update()


def build_panko(coll):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(PANKO_GLB))
    imported = set(bpy.data.objects) - before
    # Strays from the source file: unused spheres and Rigify's metarig
    for obj in list(imported):
        if obj.name.startswith("Icosphere") or obj.name.startswith("metarig"):
            imported.discard(obj)
            bpy.data.objects.remove(obj)
    stray = bpy.data.collections.get("glTF_not_exported")
    if stray is not None:
        bpy.data.collections.remove(stray)
    for obj in imported:
        link(obj, coll)

    rig = bpy.data.objects["rig"]
    rig.name = "PankoRig"
    rig.location = (0.1, 0.0, 0.0)
    rig.rotation_euler = (0, 0, math.radians(18))
    bpy.context.view_layer.update()

    # Pose in the rig's own space (arms rest straight out along X).
    # Panko's left arm waves; the right one holds a baguette up in front.
    rotate_bone(rig, "DEF-upper_arm.L", "Y", -40)
    rotate_bone(rig, "DEF-forearm.L", "Y", -35)
    rotate_bone(rig, "DEF-upper_arm.R", "Y", -20)
    rotate_bone(rig, "DEF-upper_arm.R", "Z", 55)
    rotate_bone(rig, "DEF-forearm.R", "Z", 40)

    # The baguette follows the hand bone if the pose is changed
    hand = rig.pose.bones["DEF-hand.R"]
    grip = rig.matrix_world @ hand.tail
    hold = Euler((math.radians(15), math.radians(-20), 0)).to_matrix() @ Matrix.Rotation(math.radians(90), 3, "Y")
    bread = pack("baguette", "PankoBaguette", grip, coll, rotation=hold.to_euler())
    bpy.context.view_layer.update()
    world = bread.matrix_world.copy()
    bread.parent = rig
    bread.parent_type = "BONE"
    bread.parent_bone = "DEF-hand.R"
    bread.matrix_world = world
    return rig


def build_camera():
    scene = bpy.context.scene
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = 28
    cam.location = (4.5, -19.0, 4.0)
    target = Vector((-0.2, 1.0, 4.6))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    # The engine traces two diffuse bounces
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 2
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    # Closest match to the engine's ACES fit (Narkowicz) in raytrace.slang
    scene.view_settings.view_transform = "ACES 1.3"


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_world()
    build_ground(collection("Ground"))
    store = collection("Storefront")
    build_storefront(store)
    build_neighbors(collection("Neighbors"))
    build_props(collection("Props"))
    autumn = collection("Autumn")
    build_tree(autumn)
    build_pumpkins(autumn)
    build_fallen_leaves(autumn)
    build_awning_leaves(autumn)
    build_panko(collection("Panko"))
    build_camera()
    configure_render()
    OUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))
    print(f"Saved {OUT_BLEND}")


main()
