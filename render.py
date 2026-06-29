"""
render.py
=========
Turns a shim.Scene into shaded 3D faces in a matplotlib Axes3D.
Boxes and cylinders become Poly3DCollections; snap points become coloured dots
with a small direction arrow. The view auto-scales to an equal-aspect cube.
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plant3d_shim as shim

_SOLID_FACE = "#5b8db8"
_SOLID_EDGE = "#274b6d"
_CHILD_FACE = "#7aa6c9"
_POINT_COLOR = "#e0563f"

# corner index = 4*ix + 2*iy + iz   (ix/iy/iz in {0,1})
_BOX_FACES = [
    (0, 1, 3, 2),   # x-lo
    (4, 5, 7, 6),   # x-hi
    (0, 1, 5, 4),   # y-lo
    (2, 3, 7, 6),   # y-hi
    (0, 2, 6, 4),   # z-lo
    (1, 3, 7, 5),   # z-hi
]


def _transform(M, pts):
    """Apply a 4x4 matrix to an (N,3) array of points -> (N,3)."""
    pts = np.asarray(pts, dtype=float)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ h.T).T[:, :3]


def _box_faces(M, dims):
    L, W, H = dims['L'], dims['W'], dims['H']
    if shim.BOX_CENTERED:
        xs, ys, zs = (-L / 2, L / 2), (-W / 2, W / 2), (-H / 2, H / 2)
    else:
        xs, ys, zs = (0, L), (0, W), (0, H)
    corners = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    world = _transform(M, corners)
    return [world[list(f)] for f in _BOX_FACES]


def _cylinder_faces(M, dims, n=24):
    R, H = dims['R'], dims['H']
    axis = shim.CYLINDER_AXIS
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    half = H / 2.0
    ring = np.cos(ang) * R, np.sin(ang) * R
    if axis == 'z':
        bot = np.column_stack([ring[0], ring[1], np.full(n, -half)])
        top = np.column_stack([ring[0], ring[1], np.full(n, half)])
    elif axis == 'y':
        bot = np.column_stack([ring[0], np.full(n, -half), ring[1]])
        top = np.column_stack([ring[0], np.full(n, half), ring[1]])
    else:  # x
        bot = np.column_stack([np.full(n, -half), ring[0], ring[1]])
        top = np.column_stack([np.full(n, half), ring[0], ring[1]])
    bot, top = _transform(M, bot), _transform(M, top)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([bot[i], bot[j], top[j], top[i]]))   # side
    faces.append(bot)                                              # caps
    faces.append(top)
    return faces


def draw(ax, scene):
    """Clear ax and draw the scene. Returns the (min, max) bounds tuple."""
    ax.clear()

    all_faces, child_faces = [], []
    for solid in scene.render_solids():
        if solid.kind == 'box':
            all_faces += _box_faces(solid.M, solid.dims)
        elif solid.kind == 'cylinder':
            all_faces += _cylinder_faces(solid.M, solid.dims)
        for ch in solid.children:                  # absorbed (united) geometry
            M = solid.M @ ch['rel']
            if ch['kind'] == 'box':
                child_faces += _box_faces(M, ch['dims'])
            elif ch['kind'] == 'cylinder':
                child_faces += _cylinder_faces(M, ch['dims'])

    if all_faces:
        ax.add_collection3d(Poly3DCollection(
            all_faces, facecolor=_SOLID_FACE, edgecolor=_SOLID_EDGE,
            linewidths=0.4, alpha=0.92))
    if child_faces:
        ax.add_collection3d(Poly3DCollection(
            child_faces, facecolor=_CHILD_FACE, edgecolor=_SOLID_EDGE,
            linewidths=0.4, alpha=0.92))

    # snap / connection points
    for pt in scene.points:
        x, y, z = pt['pt']
        ax.scatter([x], [y], [z], color=_POINT_COLOR, s=45, depthshade=False)
        if pt['dir']:
            dx, dy, dz = pt['dir']
            ax.quiver(x, y, z, dx, dy, dz, length=25, color=_POINT_COLOR,
                      normalize=True, linewidth=1.5)

    bounds = _autoscale(ax, all_faces + child_faces, scene.points)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    return bounds


def _autoscale(ax, faces, points):
    pts = []
    for f in faces:
        pts.extend(np.asarray(f).tolist())
    pts.extend(p['pt'] for p in points)
    if not pts:
        ax.text(0, 0, 0, "No geometry produced", ha="center")
        return None
    arr = np.array(pts)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    centre = (lo + hi) / 2
    span = max((hi - lo).max(), 1.0) * 0.6
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(centre[2] - span, centre[2] + span)
    return lo, hi
