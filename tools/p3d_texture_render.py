"""Small CPU rasterizer for applying P3D face UVs to decoded textures."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from PIL import Image

import measure_p3d_models as measure
from p3d_texture_io import TextureResolver


@dataclass(frozen=True, slots=True)
class RenderFace:
    indices: tuple[int, ...]
    uvs: tuple[tuple[float, float], ...]
    texture_path: str


def _project(
    points: np.ndarray,
    width: int,
    height: int,
    azim_deg: float,
    elev_deg: float,
    zoom: float = 1.0,
) -> np.ndarray:
    center = (np.min(points, axis=0) + np.max(points, axis=0)) * 0.5
    radius = max(float(np.max(np.linalg.norm(points - center, axis=1))), 1.0e-3)
    azim, elev = math.radians(azim_deg), math.radians(elev_deg)
    dist = radius * 2.8
    camera = np.array([
        center[0] + math.sin(azim) * math.cos(elev) * dist,
        center[1] + math.sin(elev) * dist,
        center[2] - math.cos(azim) * math.cos(elev) * dist,
    ])
    forward = center.astype(float) - camera
    forward /= np.linalg.norm(forward)
    right = np.array([-forward[2], 0.0, forward[0]])
    right /= max(np.linalg.norm(right), 1.0e-8)
    up = np.cross(right, forward)
    rel = points.astype(float) - camera
    cx, cy, cz = rel @ right, rel @ up, rel @ forward
    focal = (
        min(width, height)
        * 0.5
        / math.tan(math.radians(22.5))
        * max(float(zoom), 0.05)
    )
    return np.column_stack(
        (
            width * 0.5 + focal * cx / np.maximum(cz, 1e-6),
            height * 0.5 - focal * cy / np.maximum(cz, 1e-6),
            cz,
        )
    )


def _triangles(faces: Sequence[RenderFace]):
    for face in faces:
        if len(face.indices) == 3:
            yield face.indices, face.uvs, face.texture_path
        elif len(face.indices) == 4:
            for tri in ((0, 1, 2), (0, 2, 3)):
                yield (
                    tuple(face.indices[i] for i in tri),
                    tuple(face.uvs[i] for i in tri),
                    face.texture_path,
                )


def _sample_bilinear(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Repeat-wrap and bilinearly sample an RGBA texture for array UVs."""
    th, tw = texture.shape[:2]
    # Half-texel convention keeps integer texel centres stable while still smoothing
    # between adjacent texels. P3D UVs may extend outside 0..1, so repeat-wrap them.
    x = (np.mod(u, 1.0) * tw) - 0.5
    y = (np.mod(1.0 - v, 1.0) * th) - 0.5

    x0f = np.floor(x)
    y0f = np.floor(y)
    fx = x - x0f
    fy = y - y0f

    x0 = np.mod(x0f.astype(np.int64), tw)
    y0 = np.mod(y0f.astype(np.int64), th)
    x1 = (x0 + 1) % tw
    y1 = (y0 + 1) % th

    c00 = texture[y0, x0].astype(np.float32)
    c10 = texture[y0, x1].astype(np.float32)
    c01 = texture[y1, x0].astype(np.float32)
    c11 = texture[y1, x1].astype(np.float32)

    fx = fx[..., None].astype(np.float32)
    fy = fy[..., None].astype(np.float32)
    top = c00 * (1.0 - fx) + c10 * fx
    bottom = c01 * (1.0 - fx) + c11 * fx
    return np.clip(top * (1.0 - fy) + bottom * fy, 0, 255).astype(np.uint8)



def _mipmap_chain(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build progressively filtered mipmaps from a decoded RGBA texture.

    Sampling the 1024px base image on a handful of screen pixels causes the
    speckled moiré visible in detailed modern Arma buildings. BOX downsampling
    approximates the area covered by distant fragments before bilinear lookup.
    """
    levels = [image]
    while min(levels[-1].shape[:2]) > 1 or max(levels[-1].shape[:2]) > 1:
        current = levels[-1]
        height, width = current.shape[:2]
        next_size = (max(1, (width + 1) // 2), max(1, (height + 1) // 2))
        reduced = Image.fromarray(current).resize(next_size, resample=Image.Resampling.BOX)
        levels.append(np.asarray(reduced))
    return tuple(levels)


def _triangle_texture_lod(
    uv: np.ndarray,
    xy: np.ndarray,
    texture_width: int,
    texture_height: int,
    max_level: int,
) -> float:
    """Estimate the number of texture texels covered by one screen pixel.

    UVs stay unwrapped for derivatives. Wrapping them first would turn a
    correctly repeating brick texture into an arbitrarily scaled atlas.
    """
    dx1, dy1 = xy[1] - xy[0]
    dx2, dy2 = xy[2] - xy[0]
    det = float(dx1 * dy2 - dx2 * dy1)
    if abs(det) < 1.0e-12:
        return 0.0
    du1, dv1 = uv[1] - uv[0]
    du2, dv2 = uv[2] - uv[0]
    dudx = (du1 * dy2 - du2 * dy1) / det
    dudy = (dx1 * du2 - dx2 * du1) / det
    dvdx = (dv1 * dy2 - dv2 * dy1) / det
    dvdy = (dx1 * dv2 - dx2 * dv1) / det
    footprint = max(
        math.hypot(dudx * texture_width, dvdx * texture_height),
        math.hypot(dudy * texture_width, dvdy * texture_height),
    )
    if not math.isfinite(footprint) or footprint <= 1.0:
        return 0.0
    return min(float(max_level), math.log2(footprint))


def _sample_mipmapped(
    levels: Sequence[np.ndarray],
    u: np.ndarray,
    v: np.ndarray,
    lod: float,
) -> np.ndarray:
    """Bilinear texture sampling with trilinear transitions between mipmaps."""
    base = max(0, min(int(math.floor(lod)), len(levels) - 1))
    fraction = min(1.0, max(0.0, lod - base))
    first = _sample_bilinear(levels[base], u, v)
    if fraction <= 1.0e-6 or base == len(levels) - 1:
        return first
    second = _sample_bilinear(levels[base + 1], u, v)
    return np.clip(
        first.astype(np.float32) * (1.0 - fraction)
        + second.astype(np.float32) * fraction + 0.5, 0, 255
    ).astype(np.uint8)


def render_textured_model(
    points: np.ndarray,
    faces: Sequence[RenderFace],
    resolver: TextureResolver,
    source: str,
    *,
    width: int = 700,
    height: int = 520,
    azim_deg: float = 35.0,
    elev_deg: float = 25.0,
    zoom: float = 1.0,
    max_triangles: int = 12_000,
    load_textures: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Perspective-correct rasterization, optionally skipping all texture I/O."""
    image = np.full((height, width, 3), 236, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=float)
    projected = _project(points, width, height, azim_deg, elev_deg, zoom)
    tris = list(_triangles(faces))
    if len(tris) > max_triangles:
        picks = np.linspace(0, len(tris) - 1, max_triangles, dtype=np.int64)
        tris = [tris[int(i)] for i in picks]
    cache: dict[str, tuple[np.ndarray, ...] | None] = {}
    hits = misses = 0
    light = np.array([0.35, 0.75, -0.55])
    light /= np.linalg.norm(light)

    for indices, uvs, texture_path in tris:
        idx = np.asarray(indices, dtype=np.int32)
        p = projected[idx]
        if np.any(p[:, 2] <= 0.01):
            continue
        xy = p[:, :2]
        min_x = max(0, int(np.floor(xy[:, 0].min())))
        max_x = min(width - 1, int(np.ceil(xy[:, 0].max())))
        min_y = max(0, int(np.floor(xy[:, 1].min())))
        max_y = min(height - 1, int(np.ceil(xy[:, 1].max())))
        if max_x < min_x or max_y < min_y:
            continue
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        x2, y2 = xy[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1) + 0.5,
            np.arange(min_y, max_y + 1) + 0.5,
        )
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        if not inside.any():
            continue
        iz = 1.0 / p[:, 2]
        inv_z = w0 * iz[0] + w1 * iz[1] + w2 * iz[2]
        depth = 1.0 / np.maximum(inv_z, 1e-12)
        zslice = zbuf[min_y : max_y + 1, min_x : max_x + 1]
        mask = inside & (depth < zslice)
        if not mask.any():
            continue

        mipmaps = None
        if load_textures and texture_path:
            canonical = measure._canonical_model_path(texture_path)
            if canonical not in cache:
                decoded = resolver.load_rgba(canonical, source)
                cache[canonical] = _mipmap_chain(decoded) if decoded is not None else None
            mipmaps = cache[canonical]
        if mipmaps is None:
            if load_textures:
                misses += 1
            world = points[idx].astype(float)
            normal = np.cross(world[1] - world[0], world[2] - world[0])
            nlen = np.linalg.norm(normal)
            shade = (
                0.8
                if nlen < 1e-8
                else 0.55 + 0.45 * abs(float(np.dot(normal / nlen, light)))
            )
            target = image[min_y : max_y + 1, min_x : max_x + 1]
            target[mask] = np.clip(
                np.array([178, 184, 190]) * shade, 0, 255
            ).astype(np.uint8)
            zslice[mask] = depth[mask]
            continue

        hits += 1
        uv = np.asarray(uvs, dtype=float)
        u = (
            w0 * uv[0, 0] * iz[0]
            + w1 * uv[1, 0] * iz[1]
            + w2 * uv[2, 0] * iz[2]
        ) / np.maximum(inv_z, 1e-12)
        v = (
            w0 * uv[0, 1] * iz[0]
            + w1 * uv[1, 1] * iz[1]
            + w2 * uv[2, 1] * iz[2]
        ) / np.maximum(inv_z, 1e-12)
        lod = _triangle_texture_lod(
            uv, xy, mipmaps[0].shape[1], mipmaps[0].shape[0], len(mipmaps) - 1
        )
        sampled = _sample_mipmapped(mipmaps, u, v, lod)
        visible = mask & (sampled[:, :, 3] >= 16)
        if visible.any():
            target = image[min_y : max_y + 1, min_x : max_x + 1]
            target[visible] = sampled[:, :, :3][visible]
            zslice[visible] = depth[visible]
    return image, hits, misses
