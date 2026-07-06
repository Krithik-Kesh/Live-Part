"""
plant3d_shim.py
===============
A stand-in ("shim") for Plant 3D's private  aqa / varmain  scripting API, so that
a custom-part script can RUN OUTSIDE Plant 3D for live preview.

It does NOT build real CAD solids. Every primitive just records its size plus a
4x4 transform matrix. render.py reads those records and draws them in matplotlib.

Because the SAME unmodified part script runs here and in Plant 3D, the preview
matches what Plant 3D will build -- provided the convention knobs below match the
ones Plant 3D's geometry kernel uses. Calibrate them ONCE against a known-good
part (see example_part.py), then they're locked for every future part.

------------------------------------------------------------------------------
 CALIBRATION KNOBS  -- the only things that can make the preview disagree with
                       Plant 3D. Flip one, reload, compare. Leave them alone
                       once the sample part looks identical to Plant 3D.
------------------------------------------------------------------------------
"""

import math
import types
import numpy as np

# True  -> BOX is centred on its local origin (extends +/- size/2 on each axis)
# False -> BOX's lower corner sits at the local origin (extends 0 .. size)
BOX_CENTERED = True

# True  -> a transform is applied in WORLD space   (M_new = Op @ M_old)
#          i.e. ".translate((S,0,0))" after a rotate still moves along world X.
# False -> a transform is applied in the part's LOCAL space (M_new = M_old @ Op)
PREMULTIPLY = True

# Axis the CYLINDER's height runs along ('x', 'y', or 'z')
CYLINDER_AXIS = 'z'


# ---------------------------------------------------------------------------
# 4x4 homogeneous matrix helpers
# ---------------------------------------------------------------------------
def _translation(x, y, z):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def _rotation(axis, degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    if axis == 'x':
        M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
    elif axis == 'y':
        M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    elif axis == 'z':
        M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
class _Solid:
    """A box or cylinder that records its dimensions and accumulated transform."""

    def __init__(self, scene, kind, dims):
        self.kind = kind          # 'box' | 'cylinder'
        self.dims = dims          # dict of sizes
        self.M = np.eye(4)        # accumulated transform
        self.children = []        # absorbed geometry from uniteWith (snapshots)
        self.rendered = True      # set False by erase()
        self.absorbed = False     # set True when united into another solid
        self._scene = scene
        scene._add(self)

    # -- transforms (return self so calls can be chained) -------------------
    def _apply(self, T):
        self.M = (T @ self.M) if PREMULTIPLY else (self.M @ T)
        return self

    def translate(self, vec):
        x, y, z = vec
        return self._apply(_translation(x, y, z))

    def rotateX(self, deg):
        return self._apply(_rotation('x', deg))

    def rotateY(self, deg):
        return self._apply(_rotation('y', deg))

    def rotateZ(self, deg):
        return self._apply(_rotation('z', deg))

    # -- boolean-ish ops ----------------------------------------------------
    def uniteWith(self, other):
        """
        Plant 3D merges 'other' INTO self and returns self (one solid).
        For preview we don't compute a real boolean -- we snapshot other's
        geometry as a child of self, frozen relative to self's current frame,
        so any later transform on self moves the child too. 'other' is then
        flagged absorbed so it is not drawn as a standalone shape.
        """
        inv_self = np.linalg.inv(self.M)
        self.children.append({
            'kind': other.kind,
            'dims': other.dims,
            'rel': inv_self @ other.M,
        })
        for ch in other.children:                       # fold nested unions
            self.children.append({
                'kind': ch['kind'],
                'dims': ch['dims'],
                'rel': inv_self @ other.M @ ch['rel'],
            })
        other.absorbed = True
        return self

    def erase(self):
        self.rendered = False
        return self

    # -- convenience used by some scripts -----------------------------------
    def setPoint(self, *a, **k):
        return self._scene.setPoint(*a, **k)


def BOX(scene, L=1.0, W=1.0, H=1.0, **kw):
    return _Solid(scene, 'box', {'L': float(L), 'W': float(W), 'H': float(H)})


def CYLINDER(scene, R=1.0, H=1.0, **kw):
    return _Solid(scene, 'cylinder', {'R': float(R), 'H': float(H)})


# ---------------------------------------------------------------------------
# The scene / builder object ("s", the first argument to a part function)
# ---------------------------------------------------------------------------
class Scene:
    def __init__(self):
        self.solids = []
        self.points = []          # snap / connection points from setPoint
        self.unit = 'mm'

    def _add(self, solid):
        self.solids.append(solid)

    def setPoint(self, pt, direction=None, *a, **k):
        self.points.append({
            'pt': tuple(float(v) for v in pt),
            'dir': tuple(float(v) for v in direction) if direction else None,
        })

    # solids that should actually be drawn
    def render_solids(self):
        return [s for s in self.solids if s.rendered and not s.absorbed]


# ---------------------------------------------------------------------------
# Parameter-type sentinels and decorators
# ---------------------------------------------------------------------------
LENGTH = 'LENGTH'
ANGLE = 'ANGLE'
BOOL = 'BOOL'
STRING = 'STRING'
_TYPES = {LENGTH, ANGLE, BOOL, STRING}

_REGISTRY = []        # functions decorated with @activate, in definition order


def param(**kw):
    """
    @param(D = LENGTH, TooltipShort="...")  -- declares one parameter (the kwarg
    whose value is a type sentinel) plus optional tooltip metadata.
    Decorators stack bottom-up; we record each and reorder later to match the
    function's signature.
    """
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
    """@activate(LengthUnit='mm', ...) -- marks a function as the part entry point."""
    def deco(fn):
        fn._p3d_activate = kw
        _REGISTRY.append(fn)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Module plumbing: build fake aqa / varmain modules and reset the registry
# ---------------------------------------------------------------------------
_PUBLIC = {
    'BOX': BOX, 'CYLINDER': CYLINDER,
    'param': param, 'activate': activate,
    'LENGTH': LENGTH, 'ANGLE': ANGLE, 'BOOL': BOOL, 'STRING': STRING,
    'Scene': Scene,
}
_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}


def reset():
    """Clear the entry-point registry before re-running a script."""
    _REGISTRY.clear()


def get_registry():
    return list(_REGISTRY)


def build_fake_modules():
    """
    Return {module_name: module} to inject into sys.modules so that
        from aqa.math import *
        from varmain.primitiv import *   (etc.)
    resolve to this shim. Every name is exported from every module so a script
    resolves it no matter which sub-module it truly lives in.
    """
    def make(name, extra=None):
        m = types.ModuleType(name)
        for k, v in _PUBLIC.items():
            setattr(m, k, v)
        for k, v in _MATH.items():
            setattr(m, k, v)
        if extra:
            for k, v in extra.items():
                setattr(m, k, v)
        return m

    aqa = make('aqa')
    aqa_math = make('aqa.math')
    varmain = make('varmain')
    primitiv = make('varmain.primitiv')
    var_basic = make('varmain.var_basic')
    custom = make('varmain.custom')

    aqa.math = aqa_math
    varmain.primitiv = primitiv
    varmain.var_basic = var_basic
    varmain.custom = custom

    return {
        'aqa': aqa, 'aqa.math': aqa_math,
        'varmain': varmain, 'varmain.primitiv': primitiv,
        'varmain.var_basic': var_basic, 'varmain.custom': custom,
    }
