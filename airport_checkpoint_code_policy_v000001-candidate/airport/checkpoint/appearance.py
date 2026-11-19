"""Declared visual calibration profiles, separate from perception and control."""

import numpy as np

PAPER_PROFILES = {"legacy_warm", "neutral_white"}
LIGHTING_PROFILES = {"legacy_spot", "indoor_soft_neutral"}


def configure_paper(stage, body_paths: list[str], profile: str) -> dict:
    from pxr import Gf, UsdGeom, UsdShade

    if profile not in PAPER_PROFILES:
        raise ValueError("Unknown paper appearance profile")
    materials = {}
    for path in body_paths:
        for prim in stage.GetPrimAtPath(path).GetChildren():
            if not prim.IsA(UsdGeom.Cube):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            shader = UsdShade.Shader(stage.GetPrimAtPath(material.GetPath().AppendChild("Surface")))
            if not shader:
                raise RuntimeError("Paper visual material has no surface shader")
            if profile == "neutral_white":
                shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.90, 0.90, 0.90))
                shader.GetInput("roughness").Set(0.8)
            materials[str(material.GetPath())] = {
                "diffuse_color": list(shader.GetInput("diffuseColor").Get()),
                "roughness": float(shader.GetInput("roughness").Get()),
            }
    if not materials:
        raise RuntimeError("No paper visual materials found")
    return {"profile": profile, "materials": materials, "emission_added": False}


def configure_lighting(stage, offset: np.ndarray, profile: str) -> dict:
    from pxr import Gf, UsdGeom, UsdLux

    if profile not in LIGHTING_PROFILES:
        raise ValueError("Unknown checkpoint lighting profile")
    paths = [
        "/World/CheckpointLight",
        "/World/CheckpointLocalLight",
        "/World/CheckpointCeilingFront",
        "/World/CheckpointCeilingBack",
    ]
    for path in paths:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    dome = UsdLux.DomeLight.Define(stage, paths[0])
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr(Gf.Vec3f(1, 1, 1))
    lights = [{"path": paths[0], "kind": "dome", "intensity": 800.0}]
    if profile == "legacy_spot":
        light = UsdLux.SphereLight.Define(stage, paths[1])
        light.CreateIntensityAttr(2200.0)
        light.CreateRadiusAttr(0.4)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*(offset + [0, -1, 3])))
        lights.append({"path": paths[1], "kind": "sphere", "intensity": 2200.0})
    else:
        # Broad ceiling panels illuminate both sides of the vertical strip.
        # Fixed engineering approximation, not measured Hongqiao photometry.
        for path, position in zip(paths[2:], [(0.35, 0.8, 2.7), (0.35, -1.0, 2.7)], strict=True):
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateIntensityAttr(1000.0)
            light.CreateWidthAttr(1.2)
            light.CreateHeightAttr(0.6)
            light.CreateColorAttr(Gf.Vec3f(1, 1, 1))
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*(offset + position)))
            lights.append(
                {
                    "path": path,
                    "kind": "rect",
                    "position_work_m": position,
                    "size_m": [1.2, 0.6],
                    "intensity": 1000.0,
                    "color": [1, 1, 1],
                }
            )
    return {
        "profile": profile,
        "lights": lights,
        "camera_exposure_modified": False,
        "real_airport_photometry_calibrated": False,
    }
