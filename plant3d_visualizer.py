# plant3d_visualizer.py  --  Plant 3D Live Part Previewer
# pip install matplotlib numpy watchdog
#
# Drop anywhere, run with:  python plant3d_visualizer.py
# Open any varmain/aqa part script and see live 3D geometry.
# Save the script -> viewport refreshes automatically.

import os, sys, math, time, types, inspect, traceback, threading
import tkinter as tk
from tkinter import filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# SECTION 1 — GEOMETRY SHIM
# Fake implementations of every Plant 3D primitive.  Each class stores its
# local geometry as polygon face lists and an accumulated 4×4 world transform.
# The part script runs completely unmodified; it just hits these fakes instead
# of the real varmain/aqa classes.

#  Primitive kind labels (drive colour coding in the renderer) 
KIND_BOX        = 'box'
KIND_CYLINDER   = 'cylinder'
KIND_CONE       = 'cone'
KIND_ARC        = 'arc'
KIND_PYRAMID    = 'pyramid'
KIND_HALFSPHERE = 'halfsphere'
KIND_TORUS      = 'torus'

# ── Calibration knobs ─────────────────────────────────────────────────────────
# Flip once, compare to a known-good Plant 3D part, then leave alone.
BOX_CENTERED  = True    # BOX origin at geometric centre (not lower corner)
PREMULTIPLY   = True    # transforms in world space: M_new = Op @ M_old
POLY_N        = 24      # polygon sides for curved primitives


# ── 4×4 homogeneous helpers ───────────────────────────────────────────────────
def _T(x, y, z):
    M = np.eye(4); M[:3, 3] = (x, y, z); return M

def _R(axis, degrees):
    a = math.radians(degrees); c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    if axis == 'x': M[1,1],M[1,2],M[2,1],M[2,2] = c,-s,s,c
    elif axis == 'y': M[0,0],M[0,2],M[2,0],M[2,2] = c,s,-s,c
    elif axis == 'z': M[0,0],M[0,1],M[1,0],M[1,1] = c,-s,s,c
    return M

def _xform(M, pts):
    """Apply 4×4 matrix to (N,3) array → (N,3)."""
    pts = np.asarray(pts, dtype=float)
    h = np.hstack([pts, np.ones((len(pts),1))])
    return (M @ h.T).T[:, :3]


# ── Base solid class ──────────────────────────────────────────────────────────
class _Solid:
    """
    Every fake primitive inherits this.  It handles the 4×4 transform stack,
    boolean-ish ops (uniteWith / subtractFrom / intersectWith), and erase().
    Subclasses only need to implement _local_faces() → list of vertex arrays.
    """
    kind = KIND_BOX

    def __init__(self, scene):
        self.M        = np.eye(4)
        self.children = []       # geometry absorbed from uniteWith
        self.rendered = True     # set False by erase()
        self.absorbed = False    # set True when united into another solid
        self._scene   = scene
        scene._add(self)

    # ── transforms ──────────────────────────────────────────────────────────
    def _apply(self, op):
        self.M = (op @ self.M) if PREMULTIPLY else (self.M @ op)
        return self

    def translate(self, vec):    return self._apply(_T(*vec))
    def rotateX(self, deg):      return self._apply(_R('x', deg))
    def rotateY(self, deg):      return self._apply(_R('y', deg))
    def rotateZ(self, deg):      return self._apply(_R('z', deg))

    # ── boolean ops ─────────────────────────────────────────────────────────
    def uniteWith(self, other):
        """
        Fold other's geometry into self as a frozen child, then flag other as
        absorbed so it is not drawn independently.  Future transforms on self
        move the child with it.  We don't compute a real CSG union — the
        overlap region remains visible, which is fine for a live preview.
        """
        inv = np.linalg.inv(self.M)
        self.children.append({
            'kind':  other.kind,
            'faces': other._local_faces(),   # stored in other's local space
            'M_rel': inv @ other.M,          # other's frame relative to self's frame at union time
        })
        for ch in other.children:            # fold nested unions
            self.children.append({
                'kind':  ch['kind'],
                'faces': ch['faces'],
                'M_rel': inv @ other.M @ ch['M_rel'],
            })
        other.absorbed = True
        return self

    def subtractFrom(self, other):
        # No real CSG — mark self as erased (cutter disappears as in real Plant 3D)
        self.rendered = False
        return self

    def intersectWith(self, other):
        return self   # preview: treat as no-op

    def erase(self):
        self.rendered = False
        return self

    # ── geometry accessors ───────────────────────────────────────────────────
    def _local_faces(self):
        """Faces in local space (before self.M applied). Subclass implements."""
        return []

    def world_faces(self):
        return [_xform(self.M, f) for f in self._local_faces()]

    def child_faces(self):
        """World-space faces of all absorbed children."""
        out = []
        for ch in self.children:
            Mw = self.M @ ch['M_rel']
            for f in ch['faces']:
                out.append((_xform(Mw, f), ch['kind']))
        return out

    # scene proxy so scripts can write  solid.setPoint(...)
    def setPoint(self, *a, **k):
        return self._scene.setPoint(*a, **k)


# ── BOX ──────────────────────────────────────────────────────────────────────
_BOX_QUAD = [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]

class FakeBox(_Solid):
    kind = KIND_BOX
    def __init__(self, scene, L=1, W=1, H=1):
        super().__init__(scene)
        self.L, self.W, self.H = float(L), float(W), float(H)

    def _local_faces(self):
        L, W, H = self.L, self.W, self.H
        xs = (-L/2, L/2) if BOX_CENTERED else (0, L)
        ys = (-W/2, W/2) if BOX_CENTERED else (0, W)
        zs = (-H/2, H/2) if BOX_CENTERED else (0, H)
        c = np.array([(x,y,z) for x in xs for y in ys for z in zs])
        return [c[list(q)] for q in _BOX_QUAD]


# ── CYLINDER ─────────────────────────────────────────────────────────────────
class FakeCylinder(_Solid):
    kind = KIND_CYLINDER
    def __init__(self, scene, R=1, H=1, O=0, Ry=None):
        super().__init__(scene)
        self.R  = float(R)
        self.Ry = float(Ry) if Ry is not None else float(R)  # elliptical minor radius
        self.H  = float(H)
        self.O  = float(O)   # hole (annular) radius; 0 = solid

    def _local_faces(self):
        n = POLY_N
        ang = np.linspace(0, 2*math.pi, n, endpoint=False)
        ox, oy = np.cos(ang)*self.R, np.sin(ang)*self.Ry
        # base at z=0, top at z=H  (Plant 3D: base point = centre of bottom face)
        bot = np.column_stack([ox, oy, np.zeros(n)])
        top = np.column_stack([ox, oy, np.full(n, self.H)])
        faces = []
        for i in range(n):
            j = (i+1) % n
            faces.append(np.array([bot[i], bot[j], top[j], top[i]]))  # side
        if self.O > 0:
            ix, iy = np.cos(ang)*self.O, np.sin(ang)*self.O
            ibot = np.column_stack([ix, iy, np.zeros(n)])
            itop = np.column_stack([ix, iy, np.full(n, self.H)])
            for i in range(n):
                j = (i+1) % n
                faces.append(np.array([ibot[i], ibot[j], itop[j], itop[i]]))  # inner wall
                faces.append(np.array([bot[i],  ibot[i], ibot[j], bot[j]]))   # bottom annulus
                faces.append(np.array([top[i],  itop[i], itop[j], top[j]]))   # top annulus
        else:
            faces.append(bot)   # solid bottom cap
            faces.append(top)   # solid top cap
        return faces


# ── CONE ─────────────────────────────────────────────────────────────────────
class FakeCone(_Solid):
    kind = KIND_CONE
    def __init__(self, scene, R1=1, R2=0, H=1, E=0):
        super().__init__(scene)
        self.R1 = float(R1)   # bottom radius
        self.R2 = float(R2)   # top radius (0 = full cone)
        self.H  = float(H)
        self.E  = float(E)    # eccentricity: top-centre X offset

    def _local_faces(self):
        n = POLY_N
        ang = np.linspace(0, 2*math.pi, n, endpoint=False)
        bot = np.column_stack([np.cos(ang)*self.R1, np.sin(ang)*self.R1, np.zeros(n)])
        faces = [bot]  # bottom cap
        if self.R2 > 0:
            top = np.column_stack([np.cos(ang)*self.R2 + self.E,
                                   np.sin(ang)*self.R2,
                                   np.full(n, self.H)])
            faces.append(top)
            for i in range(n):
                j = (i+1) % n
                faces.append(np.array([bot[i], bot[j], top[j], top[i]]))
        else:
            apex = np.array([self.E, 0.0, self.H])
            for i in range(n):
                j = (i+1) % n
                faces.append(np.array([bot[i], bot[j], apex]))
        return faces


# ── ARC3D (pipe elbow) ────────────────────────────────────────────────────────
class FakeArc(_Solid):
    """
    Swept tube elbow.  Sweeps in the XZ plane.  Base point is placed at the
    intersection of the two extended centreline tangents (matching Plant 3D's
    convention), so translate/rotate calls land in the right place.
    """
    kind = KIND_ARC
    def __init__(self, scene, D=1, R=5, A=90, D2=None, S=None):
        super().__init__(scene)
        self.D  = float(D)
        self.D2 = float(D2) if D2 is not None else float(D)  # reduced elbow exit radius
        self.R  = float(R)
        self.A  = float(A)
        self.S  = int(S) if S is not None else None

    def _local_faces(self):
        A_rad = math.radians(self.A)
        R = self.R
        n_tube  = POLY_N
        n_sweep = self.S if self.S else max(8, int(abs(self.A) / 4))

        # Base point: intersection of extended centreline tangents.
        # Centreline: P(θ) = (R·sinθ, 0, R·(1−cosθ))  in the XZ plane.
        # Tangent at θ=0 is +X; tangent at θ=A is (cosA, 0, sinA).
        # Intersection of the two tangent lines lands at X = R·(1−cosA)/sinA.
        s_a = math.sin(A_rad)
        base_x = R * (1 - math.cos(A_rad)) / s_a if abs(s_a) > 1e-6 else 0.0

        thetas = np.linspace(0, A_rad, n_sweep + 1)
        tube_ang = np.linspace(0, 2*math.pi, n_tube, endpoint=False)

        rings = []
        for θ in thetas:
            # Centreline position shifted so base point sits at origin
            cx = R * math.sin(θ) - base_x
            cz = R * (1 - math.cos(θ))
            center = np.array([cx, 0.0, cz])
            # Cross-section basis perpendicular to sweep tangent (cosθ, 0, sinθ)
            e1 = np.array([0.0, 1.0, 0.0])
            e2 = np.array([-math.sin(θ), 0.0, math.cos(θ)])
            # Interpolate radius linearly for reduced elbow (ARC3D2)
            t = θ / A_rad if A_rad else 0
            r = self.D * (1 - t) + self.D2 * t
            ring = np.array([center + r*(math.cos(φ)*e1 + math.sin(φ)*e2)
                             for φ in tube_ang])
            rings.append(ring)

        faces = []
        for i in range(len(rings)-1):
            r0, r1 = rings[i], rings[i+1]
            for k in range(n_tube):
                j = (k+1) % n_tube
                faces.append(np.array([r0[k], r0[j], r1[j], r1[k]]))
        faces.append(rings[0])    # end cap 1
        faces.append(rings[-1])   # end cap 2
        return faces


# ── PYRAMID ───────────────────────────────────────────────────────────────────
class FakePyramid(_Solid):
    kind = KIND_PYRAMID
    def __init__(self, scene, L=1, W=1, H=1, HT=2):
        super().__init__(scene)
        self.L, self.W, self.H, self.HT = float(L), float(W), float(H), float(HT)

    def _local_faces(self):
        L, W, H, HT = self.L, self.W, self.H, self.HT
        scale = max(0.0, 1 - H/HT) if HT else 0.0
        hL, hW = L/2, W/2
        bot = np.array([(-hL,-hW,0),(hL,-hW,0),(hL,hW,0),(-hL,hW,0)])
        top = np.array([(-hL*scale,-hW*scale,H),(hL*scale,-hW*scale,H),
                        (hL*scale,hW*scale,H),(-hL*scale,hW*scale,H)])
        faces = [bot, top]
        for i in range(4):
            j = (i+1) % 4
            faces.append(np.array([bot[i], bot[j], top[j], top[i]]))
        return faces


# ── HALFSPHERE ────────────────────────────────────────────────────────────────
class FakeHalfsphere(_Solid):
    kind = KIND_HALFSPHERE
    def __init__(self, scene, R=1):
        super().__init__(scene)
        self.R = float(R)

    def _local_faces(self):
        n_lat, n_lon = 10, POLY_N
        # elevation 0 = rim (z=0), elevation π/2 = apex (z=R)
        lats = np.linspace(0, math.pi/2, n_lat+1)
        lons = np.linspace(0, 2*math.pi, n_lon, endpoint=False)
        def spt(el, lo):
            return np.array([self.R*math.cos(el)*math.cos(lo),
                             self.R*math.cos(el)*math.sin(lo),
                             self.R*math.sin(el)])
        faces = []
        for i in range(n_lat):
            la0, la1 = lats[i], lats[i+1]
            for j in range(n_lon):
                lo0 = lons[j]; lo1 = lons[(j+1)%n_lon]
                p00,p01 = spt(la0,lo0), spt(la0,lo1)
                p10,p11 = spt(la1,lo0), spt(la1,lo1)
                if i == n_lat-1:
                    faces.append(np.array([p00,p01,p11]))
                else:
                    faces.append(np.array([p00,p01,p11,p10]))
        # flat bottom disk
        rim = np.array([[self.R*math.cos(a),self.R*math.sin(a),0] for a in lons])
        faces.append(rim)
        return faces


# ── TORUS ─────────────────────────────────────────────────────────────────────
class FakeTorus(_Solid):
    kind = KIND_TORUS
    def __init__(self, scene, R1=3, R2=1):
        super().__init__(scene)
        # Plant 3D: R1=outer radius, R2=inner radius of the torus solid
        self.Rc = (float(R1) + float(R2)) / 2   # tube-centre radius
        self.Rt = (float(R1) - float(R2)) / 2   # tube radius

    def _local_faces(self):
        n_maj, n_min = POLY_N, 12
        maj = np.linspace(0, 2*math.pi, n_maj, endpoint=False)
        minn = np.linspace(0, 2*math.pi, n_min, endpoint=False)
        def pt(u, v):
            return np.array([(self.Rc+self.Rt*math.cos(v))*math.cos(u),
                             (self.Rc+self.Rt*math.cos(v))*math.sin(u),
                              self.Rt*math.sin(v)])
        faces = []
        for i,u0 in enumerate(maj):
            u1 = maj[(i+1)%n_maj]
            for j,v0 in enumerate(minn):
                v1 = minn[(j+1)%n_min]
                faces.append(np.array([pt(u0,v0),pt(u0,v1),pt(u1,v1),pt(u1,v0)]))
        return faces


# ── Scene ("s") ───────────────────────────────────────────────────────────────
class Scene:
    """The session object passed as the first argument to every part function."""
    def __init__(self):
        self.solids = []
        self.points = []

    def _add(self, solid):
        self.solids.append(solid)

    def setPoint(self, pt, direction=None, *a, **k):
        self.points.append({
            'pt':  tuple(float(v) for v in pt),
            'dir': tuple(float(v) for v in direction) if direction else None,
        })

    def render_solids(self):
        return [s for s in self.solids if s.rendered and not s.absorbed]


# ── Primitive factory functions (what part scripts actually call) ──────────────
def BOX(scene, L=1, W=1, H=1, **kw):
    return FakeBox(scene, L, W, H)

def CYLINDER(scene, R=1.0, H=1.0, O=0.0, **kw):
    # Support R1/R2 keyword args for elliptical cylinders
    r1 = float(kw.get('R1', R))
    r2 = float(kw.get('R2', r1))
    return FakeCylinder(scene, r1, float(H), float(O), Ry=r2)

def CONE(scene, R1=1, R2=0, H=1, E=0, **kw):
    return FakeCone(scene, R1, R2, H, E)

def ARC3D(scene, D=1, R=5, A=90, **kw):
    return FakeArc(scene, D, R, A)

def ARC3D2(scene, D=1, D2=None, R=5, A=90, **kw):
    return FakeArc(scene, D, R, A, D2=D2 if D2 is not None else D)

def ARC3DS(scene, D=1, R=5, A=90, S=8, **kw):
    return FakeArc(scene, D, R, A, S=S)

def PYRAMID(scene, L=1, W=1, H=1, HT=2, **kw):
    return FakePyramid(scene, L, W, H, HT)

def HALFSPHERE(scene, R=1, **kw):
    return FakeHalfsphere(scene, R)

def TORUS(scene, R1=3, R2=1, **kw):
    return FakeTorus(scene, R1, R2)


# ── Parameter type sentinels ──────────────────────────────────────────────────
LENGTH = LENGTH0 = 'LENGTH'
ANGLE  = 'ANGLE'
INT    = 'INT'
DOUBLE = 'DOUBLE'
STRING = 'STRING'
BOOL   = 'BOOL'
_TYPES = {LENGTH, ANGLE, INT, DOUBLE, STRING, BOOL}

_REGISTRY = []   # functions decorated with @activate, in definition order


# ── Decorators (metadata only; geometry is irrelevant to these) ───────────────
def param(**kw):
    def deco(fn):
        params = getattr(fn, '_p3d_params', [])
        name = ptype = None
        meta = {}
        for k, v in kw.items():
            if v in _TYPES and name is None:
                name, ptype = k, v
            else:
                meta[k] = v
        if name is not None:
            params.append({'name': name, 'type': ptype, 'meta': meta})
        fn._p3d_params = params
        return fn
    return deco

def activate(**kw):
    def deco(fn):
        fn._p3d_activate = kw
        _REGISTRY.append(fn)
        return fn
    return deco

def group(*a, **k):   return lambda fn: fn   # metadata-only
def enum(*a, **k):    return lambda fn: fn   # metadata-only


# ── Fake module injection ─────────────────────────────────────────────────────
_PUBLIC = {
    'BOX': BOX, 'CYLINDER': CYLINDER, 'CONE': CONE,
    'ARC3D': ARC3D, 'ARC3D2': ARC3D2, 'ARC3DS': ARC3DS,
    'PYRAMID': PYRAMID, 'HALFSPHERE': HALFSPHERE, 'TORUS': TORUS,
    'param': param, 'activate': activate, 'group': group, 'enum': enum,
    'LENGTH': LENGTH, 'LENGTH0': LENGTH0, 'ANGLE': ANGLE,
    'INT': INT, 'DOUBLE': DOUBLE, 'STRING': STRING, 'BOOL': BOOL,
    'Scene': Scene,
}
_MATH_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}


def _shim_reset():
    _REGISTRY.clear()


def _make_fake_modules():
    """Build fake aqa/varmain sub-modules so any import resolves to this shim."""
    def make(name):
        m = types.ModuleType(name)
        for k, v in _PUBLIC.items():      setattr(m, k, v)
        for k, v in _MATH_NAMES.items():  setattr(m, k, v)
        return m
    aqa = make('aqa');           aqa_math  = make('aqa.math')
    varmain = make('varmain');   primitiv  = make('varmain.primitiv')
    var_basic = make('varmain.var_basic'); custom = make('varmain.custom')
    aqa.math = aqa_math
    varmain.primitiv = primitiv; varmain.var_basic = var_basic; varmain.custom = custom
    return {
        'aqa': aqa, 'aqa.math': aqa_math,
        'varmain': varmain, 'varmain.primitiv': primitiv,
        'varmain.var_basic': var_basic, 'varmain.custom': custom,
    }


# 
# SECTION 2 — SCRIPT RUNNER
# Loads a part script into a clean namespace, finds the part function,
# extracts parameter specs, and calls it with the Scene object.
# 

def _find_entry(namespace):
    """Prefer @activate-decorated function; else the last defined function."""
    if _REGISTRY:
        return _REGISTRY[-1]
    funcs = [v for v in namespace.values() if isinstance(v, types.FunctionType)]
    return funcs[-1] if funcs else None


def _param_specs(fn):
    """
    Build an ordered list of parameter dicts in function-signature order,
    skipping the first arg (scene) and **kwargs.  Merges tooltip metadata
    from @param decorators.
    """
    meta = {p['name']: p for p in getattr(fn, '_p3d_params', [])}
    specs = []
    for p in list(inspect.signature(fn).parameters.values())[1:]:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        default = None if p.default is inspect.Parameter.empty else p.default
        info = meta.get(p.name, {})
        ptype = info.get('type', 'LENGTH')
        specs.append({
            'name':    p.name,
            'default': default,
            'type':    ptype,
            'tooltip': info.get('meta', {}).get('TooltipShort', ''),
            'numeric': isinstance(default, (int, float)) or ptype in (
                       'LENGTH', 'LENGTH0', 'ANGLE', 'INT', 'DOUBLE'),
        })
    return specs


def run_script(path, overrides=None):
    """
    Execute the part script with fake primitives injected.
    Returns (scene, specs, used_values).
    Raises on syntax/runtime errors — caller catches and shows in console.
    """
    _shim_reset()
    for name, mod in _make_fake_modules().items():
        sys.modules[name] = mod

    with open(path, 'r', encoding='utf-8') as fh:
        source = fh.read()

    namespace = {}
    code = compile(source, path, 'exec')
    exec(code, namespace)  # noqa: S102  (trusted, self-authored scripts)

    fn = _find_entry(namespace)
    if fn is None:
        raise RuntimeError("No part function found.  Add @activate or define a function.")

    specs = _param_specs(fn)

    # Start from defaults, apply UI overrides on top
    used = {s['name']: s['default'] for s in specs if s['default'] is not None}
    if overrides:
        for k, v in overrides.items():
            if k in used or any(s['name'] == k for s in specs):
                used[k] = v

    scene = Scene()
    first_arg = list(inspect.signature(fn).parameters)[0]
    fn(scene, **{k: v for k, v in used.items() if k != first_arg})

    return scene, specs, used


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RENDERER
# Converts a Scene into matplotlib 3D artists.  Each primitive kind gets its
# own colour; ports are drawn as red spheres with direction arrows.
# ═══════════════════════════════════════════════════════════════════════════════

_KIND_COLOUR = {
    KIND_BOX:        ('#4a90d9', '#2a5a99'),
    KIND_CYLINDER:   ('#00adb5', '#006b72'),
    KIND_CONE:       ('#e05c00', '#8a3800'),
    KIND_ARC:        ('#7c4dbd', '#4a2e73'),
    KIND_PYRAMID:    ('#e8b400', '#8a6c00'),
    KIND_HALFSPHERE: ('#5cb85c', '#357535'),
    KIND_TORUS:      ('#e91e8c', '#8a1154'),
}
_DEFAULT_COL = ('#888888', '#555555')
FACE_ALPHA   = 0.35


def _collect_geometry(scene):
    """Return [(faces_list, kind), ...] for all visible solids and children."""
    items = []
    for solid in scene.render_solids():
        wf = solid.world_faces()
        if wf:
            items.append((wf, solid.kind))
        for face, kind in solid.child_faces():
            items.append(([face], kind))
    return items


def _all_vertices(items, points):
    arr = []
    for faces, _ in items:
        for f in faces:
            arr.extend(np.asarray(f).tolist())
    for p in points:
        arr.append(list(p['pt']))
    return np.array(arr, dtype=float) if arr else None


def draw(ax, scene):
    """Clear ax and render the scene.  Returns vertex array for autoscale."""
    ax.clear()
    ax.set_facecolor("#1c1c2e")

    items = _collect_geometry(scene)
    verts = _all_vertices(items, scene.points)

    # All faces go into ONE Poly3DCollection with per-face colours.
    # matplotlib's 3D engine has no real depth buffer -- it depth-sorts
    # polygons by centroid distance, but only WITHIN a single collection.
    # Drawing one collection per primitive kind meant faces from different
    # kinds couldn't be sorted against each other, causing flicker/glitching
    # as the camera rotated. A single shared collection fixes that.
    all_faces, face_colors, edge_colors = [], [], []
    seen_kinds = {}
    for faces, kind in items:
        fc, ec = _KIND_COLOUR.get(kind, _DEFAULT_COL)
        all_faces.extend(faces)
        face_colors.extend([fc] * len(faces))
        edge_colors.extend([ec] * len(faces))
        seen_kinds.setdefault(kind, (fc, ec))

    if all_faces:
        ax.add_collection3d(Poly3DCollection(
            all_faces, facecolor=face_colors, edgecolor=edge_colors,
            linewidths=0.3, alpha=FACE_ALPHA))

    legend_handles = [
        mpatches.Patch(facecolor=fc, edgecolor=ec, label=kind.capitalize())
        for kind, (fc, ec) in seen_kinds.items()
    ]

    # Ports: red sphere + arrow + label
    if verts is not None and len(verts):
        diag = max(np.linalg.norm(verts.max(0) - verts.min(0)) * 0.15, 1.0)
    else:
        diag = 10.0

    for i, p in enumerate(scene.points):
        x, y, z = p['pt']
        ax.scatter([x], [y], [z], color='#ff2222', s=55, depthshade=False, zorder=5)
        if p['dir']:
            dx, dy, dz = p['dir']
            ax.quiver(x, y, z, dx, dy, dz, length=diag, normalize=True,
                      color='#ff2222', linewidth=2, arrow_length_ratio=0.25)
        ax.text(x, y, z, f"  Port {i+1}", color='#ff4444', fontsize=7, zorder=6)

    _autoscale(ax, verts)
    ax.set_xlabel("X", color='#8899bb')
    ax.set_ylabel("Y", color='#8899bb')
    ax.set_zlabel("Z", color='#8899bb')
    ax.tick_params(colors='#6677aa', labelsize=7)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=7,
                  framealpha=0.5, facecolor='#22223b', labelcolor='white',
                  edgecolor='#334466')
    return verts


def _autoscale(ax, verts):
    if verts is None or len(verts) == 0:
        ax.text(0, 0, 0, "No geometry produced", ha='center', color='#886688')
        return
    lo, hi = verts.min(0), verts.max(0)
    centre = (lo + hi) / 2
    span   = max((hi - lo).max() * 0.6, 1.0)
    ax.set_xlim(centre[0]-span, centre[0]+span)
    ax.set_ylim(centre[1]-span, centre[1]+span)
    ax.set_zlim(centre[2]-span, centre[2]+span)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — USER INTERFACE
# Dark-themed tkinter window: toolbar / left panel (params + console) / viewport.
# File watching via watchdog if installed, or mtime polling as fallback.
# ═══════════════════════════════════════════════════════════════════════════════

POLL_MS  = 350    # mtime poll interval (ms) when watchdog not available
DEBOUNCE = 0.30   # minimum seconds between successive auto-reloads

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False

# ── App state ─────────────────────────────────────────────────────────────────
_st = {
    'path':        None,
    'mtime':       None,
    'specs':       [],
    'entries':     {},      # param name → tk.Entry widget
    'observer':    None,    # watchdog Observer instance
    'last_reload': 0.0,
    'xlim':        None,    # remembered zoom box (None = use autoscale)
    'ylim':        None,
    'zlim':        None,
}

# ── Root window ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Plant 3D Live Previewer")
root.geometry("1300x800")
root.configure(bg="#1c1c2e")

# ── Toolbar ───────────────────────────────────────────────────────────────────
bar = tk.Frame(root, bg="#16213e", pady=8, padx=14)
bar.pack(fill='x')

tk.Label(bar, text="Plant 3D Live Previewer", bg="#16213e", fg="#dde8ff",
         font=("Segoe UI", 13, "bold")).pack(side='left')

_watch_var  = tk.StringVar(value="")
_status_var = tk.StringVar(value="Open a part script to begin.")

tk.Label(bar, textvariable=_watch_var, bg="#16213e", fg="#00e5cc",
         font=("Segoe UI", 9)).pack(side='right', padx=(0, 18))
tk.Label(bar, textvariable=_status_var, bg="#16213e", fg="#8aaccc",
         font=("Segoe UI", 9)).pack(side='right')

# ── Body ──────────────────────────────────────────────────────────────────────
body = tk.Frame(root, bg="#1c1c2e")
body.pack(fill='both', expand=True)

# Left panel
left = tk.Frame(body, bg="#1a1a2e", width=230)
left.pack(side='left', fill='y')
left.pack_propagate(False)

# 3D viewport
fig = Figure(figsize=(9, 7), dpi=96)
fig.patch.set_facecolor("#1c1c2e")
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor("#1c1c2e")
ax.set_axis_off()
ax.text(0, 0, 0, "Open a script\nto see your part here",
        ha='center', va='center', fontsize=13, color='#445566')

canvas = FigureCanvasTkAgg(fig, master=body)
canvas.get_tk_widget().pack(side='left', fill='both', expand=True)
canvas.draw()


# ── Scroll-wheel zoom ─────────────────────────────────────────────────────────
def _on_scroll(event):
    """Mouse wheel over the viewport zooms in/out around the current view centre."""
    if event.inaxes != ax:
        return
    factor = 0.88 if event.button == 'up' else 1.0 / 0.88
    new_lims = []
    for get_lim in (ax.get_xlim3d, ax.get_ylim3d, ax.get_zlim3d):
        lo, hi = get_lim()
        centre = (lo + hi) / 2
        half = (hi - lo) / 2 * factor
        new_lims.append((centre - half, centre + half))
    ax.set_xlim3d(*new_lims[0])
    ax.set_ylim3d(*new_lims[1])
    ax.set_zlim3d(*new_lims[2])
    # remember it so the next auto-reload keeps this zoom level
    _st['xlim'], _st['ylim'], _st['zlim'] = new_lims
    canvas.draw_idle()

canvas.mpl_connect('scroll_event', _on_scroll)

# Hint bar
hint = tk.Frame(root, bg="#12121f", pady=4)
hint.pack(fill='x')
tk.Label(hint,
         text="Drag to rotate  ·  Scroll to zoom  ·  Save your script to auto-refresh  ·  "
              "Edit a parameter and press Enter to re-render",
         bg="#12121f", fg="#445566", font=("Segoe UI", 8)).pack()

# ── Left panel: Parameters section ───────────────────────────────────────────
_SEP = dict(bg="#1a1a2e")
tk.Label(left, text="PARAMETERS", **_SEP, fg="#5577aa",
         font=("Segoe UI", 8, "bold")).pack(anchor='w', padx=10, pady=(14, 2))

_param_frame = tk.Frame(left, **_SEP)
_param_frame.pack(fill='x', padx=6)

_apply_btn = tk.Button(left, text="Apply ↵", bg="#00adb5", fg="white",
                       font=("Segoe UI", 8), relief='flat', pady=3,
                       activebackground="#009aa0")
_apply_btn.pack(fill='x', padx=10, pady=(5, 0))

# ── Left panel: Console section ───────────────────────────────────────────────
tk.Label(left, text="CONSOLE", **_SEP, fg="#5577aa",
         font=("Segoe UI", 8, "bold")).pack(anchor='w', padx=10, pady=(14, 2))

_console = tk.Text(left, bg="#111120", fg="#9aaabb",
                   font=("Consolas", 8), relief='flat',
                   state='disabled', wrap='word',
                   insertbackground='white')
_console.pack(fill='both', expand=True, padx=4, pady=(0, 4))
_console.tag_config('err',  foreground='#ff6060')
_console.tag_config('ok',   foreground='#55dd88')
_console.tag_config('info', foreground='#7799bb')
_console.tag_config('port', foreground='#ff9988')
_console.tag_config('warn', foreground='#ddaa44')


def _log(msg, tag='info'):
    _console.configure(state='normal')
    _console.insert('end', msg + "\n", tag)
    _console.see('end')
    _console.configure(state='disabled')


def _log_clear():
    _console.configure(state='normal')
    _console.delete('1.0', 'end')
    _console.configure(state='disabled')


# ── Parameter panel builder ───────────────────────────────────────────────────
def _overrides():
    out = {}
    for name, e in _st['entries'].items():
        txt = e.get().strip()
        if not txt:
            continue
        try:
            out[name] = float(txt)
        except ValueError:
            pass
    return out


def _build_param_panel(specs, values):
    for w in _param_frame.winfo_children():
        w.destroy()
    _st['entries'] = {}
    _st['specs'] = specs

    for spec in specs:
        if not spec['numeric']:
            continue
        row = tk.Frame(_param_frame, bg="#1a1a2e")
        row.pack(fill='x', pady=2)
        tk.Label(row, text=spec['name'], width=7, anchor='w',
                 bg="#1a1a2e", fg="#c8d8f0",
                 font=("Consolas", 9, "bold")).pack(side='left')
        e = tk.Entry(row, width=9, font=("Consolas", 9),
                     bg="#0e0e22", fg="#e8f0ff",
                     insertbackground='white', relief='flat',
                     highlightthickness=1, highlightcolor="#334488",
                     highlightbackground="#222244")
        val = values.get(spec['name'], spec['default'])
        e.insert(0, str(round(val, 6) if isinstance(val, float) else val))
        e.pack(side='left', padx=(2, 0))
        e.bind("<Return>", lambda _: render_now())
        _st['entries'][spec['name']] = e
        if spec['tooltip']:
            tk.Label(_param_frame, text=spec['tooltip'],
                     bg="#1a1a2e", fg="#445566",
                     font=("Segoe UI", 7), anchor='w',
                     wraplength=204, justify='left').pack(fill='x', pady=(0, 1))

_apply_btn.configure(command=lambda: render_now())


# ── Core render function ──────────────────────────────────────────────────────
def render_now():
    path = _st['path']
    if not path:
        return

    # Debounce: ignore calls that arrive within DEBOUNCE seconds of the last one
    now = time.monotonic()
    if now - _st['last_reload'] < DEBOUNCE:
        return
    _st['last_reload'] = now

    # Preserve camera angle across re-renders
    try:
        elev, azim = ax.elev, ax.azim
    except Exception:
        elev, azim = 25, -60

    _log_clear()
    _log(f"[{time.strftime('%H:%M:%S')}]  {os.path.basename(path)}", 'info')

    try:
        scene, specs, used = run_script(path, _overrides())
    except SyntaxError as exc:
        msg = f"Syntax error  line {exc.lineno}: {exc.msg}"
        _status_var.set(msg)
        _log(msg, 'err')
        if exc.text:
            _log(f"  {exc.text.rstrip()}", 'err')
        return
    except Exception:
        tb = traceback.format_exc()
        short = tb.strip().splitlines()[-1][:90]
        _status_var.set("ERROR: " + short)
        _log(tb, 'err')
        return

    # Rebuild param panel only when the parameter set changes
    if [s['name'] for s in specs] != [s['name'] for s in _st['specs']]:
        _build_param_panel(specs, used)

    ax.set_axis_on()
    draw(ax, scene)
    ax.view_init(elev=elev, azim=azim)
    # Reapply a remembered zoom level (set via the scroll wheel) so tweaking
    # a parameter or auto-reloading on save doesn't snap back to full zoom-out.
    if _st['xlim'] is not None:
        ax.set_xlim3d(*_st['xlim'])
        ax.set_ylim3d(*_st['ylim'])
        ax.set_zlim3d(*_st['zlim'])
    canvas.draw()

    n = len(scene.render_solids())
    _status_var.set(f"OK  ·  {os.path.basename(path)}  ·  {n} solid(s)")
    _log(f"OK — {n} solid(s),  {len(scene.points)} port(s)", 'ok')

    for i, p in enumerate(scene.points):
        x, y, z = p['pt']
        d = p['dir']
        dir_str = f"  →  ({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f})" if d else ""
        _log(f"  Port {i+1}:  ({x:.2f}, {y:.2f}, {z:.2f}){dir_str}", 'port')


# ── File watching ─────────────────────────────────────────────────────────────
def _stop_observer():
    obs = _st.get('observer')
    if obs and obs.is_alive():
        obs.stop()
        obs.join(timeout=1)
    _st['observer'] = None


def _start_watchdog(path):
    _stop_observer()
    if not _HAS_WATCHDOG:
        return

    target = os.path.abspath(path)

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if os.path.abspath(event.src_path) == target:
                root.after(0, render_now)

    obs = Observer()
    obs.schedule(_Handler(), path=os.path.dirname(target), recursive=False)
    obs.daemon = True
    obs.start()
    _st['observer'] = obs
    _watch_var.set("● watching")


def _poll_mtime():
    """Fallback when watchdog is absent — poll modification time."""
    path = _st['path']
    if path and os.path.exists(path):
        mt = os.path.getmtime(path)
        if mt != _st['mtime']:
            _st['mtime'] = mt
            render_now()
    root.after(POLL_MS, _poll_mtime)


# ── Open / Reload buttons ─────────────────────────────────────────────────────
def open_script():
    path = filedialog.askopenfilename(
        title="Open Plant 3D part script",
        filetypes=[("Python part scripts", "*.py"), ("All files", "*.*")])
    if not path:
        return
    _st.update(path=path, mtime=None, specs=[], last_reload=0.0,
               xlim=None, ylim=None, zlim=None)   # fresh script -> fresh autoscale
    _watch_var.set("")
    _log_clear()
    _log(f"Opened: {path}", 'info')
    if not _HAS_WATCHDOG:
        _log("Tip: pip install watchdog  for instant file-save detection.", 'warn')
    _start_watchdog(path)
    render_now()


def _force_reload():
    _st['last_reload'] = 0.0
    render_now()


def _recenter():
    """Drop the remembered zoom and let draw()'s autoscale fit everything again."""
    _st['xlim'] = _st['ylim'] = _st['zlim'] = None
    _force_reload()


tk.Button(bar, text="Open Script", command=open_script,
          bg="#00adb5", fg="white", font=("Segoe UI", 9),
          relief='flat', padx=12, pady=4,
          activebackground="#009aa0").pack(side='left', padx=(18, 0))

tk.Button(bar, text="Reload", command=_force_reload,
          bg="#2a3a5a", fg="#aabbdd", font=("Segoe UI", 9),
          relief='flat', padx=10, pady=4,
          activebackground="#3a4a6a").pack(side='left', padx=(6, 0))

tk.Button(bar, text="Recenter", command=_recenter,
          bg="#2a3a5a", fg="#aabbdd", font=("Segoe UI", 9),
          relief='flat', padx=10, pady=4,
          activebackground="#3a4a6a").pack(side='left', padx=(6, 0))

# ── Start ─────────────────────────────────────────────────────────────────────
root.after(POLL_MS, _poll_mtime)
root.protocol("WM_DELETE_WINDOW", lambda: (_stop_observer(), root.destroy()))
root.mainloop()
