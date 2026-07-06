"""
plant3d_shim.py
===============
A stand-in ("shim") for Plant 3D's private  aqa / varmain  scripting API, so that
a custom-part script can RUN OUTSIDE Plant 3D for live preview.

It does NOT build real CAD solids. Every primitive just records its size plus a
4x4 transform matrix. render.py reads those records and draws them in matplotlib.

DESIGN PRINCIPLE (added after the uniteWith incident):
    Where Plant 3D is silent, the shim must SCREAM.
Plant 3D swallows script exceptions and returns nil from TESTACPSCRIPT; this
shim instead raises loud, specific errors for every known misuse pattern:

  1. Using the return value of uniteWith / subtractFrom / intersectWith.
     In the real API these MUTATE the calling object; their return value is
     not a usable solid. The shim returns a poison object that raises on any
     attribute access, with a message explaining the correct pattern.
  2. Using a solid after .erase() -- dead objects raise on any method call.
  3. Boolean-op operands that don't overlap (uniteWith requires physical
     intersection in Plant 3D) -- the shim warns with both AABBs printed.

Transform methods (rotateX/Y/Z, translate) DO chain in the real API and
therefore still return self here. Only the boolean ops are poisoned.

------------------------------------------------------------------------------
 CALIBRATION KNOBS  -- the only things that can make the preview disagree with
                       Plant 3D. Flip one, reload, compare. Leave them alone
                       once the sample part looks identical to Plant 3D.
------------------------------------------------------------------------------
"""

import math
import types
import warnings
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

# If True, non-overlapping uniteWith operands raise instead of warn.
STRICT_OVERLAP = False


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
# Strictness helpers
# ---------------------------------------------------------------------------
class ShimAPIError(RuntimeError):
    """Raised for API misuse that Plant 3D would swallow silently."""


class _Consumed:
    """
    Poison object returned by uniteWith / subtractFrom / intersectWith.
    In Plant 3D those methods mutate the calling object; using their return
    value crashes the script with a swallowed exception (TESTACPSCRIPT -> nil).
    Here, ANY use of it raises immediately with an explanation.
    """
    __slots__ = ('_op',)

    def __init__(self, op):
        object.__setattr__(self, '_op', op)

    def _blow_up(self, how):
        op = object.__getattribute__(self, '_op')
        raise ShimAPIError(
            f"You used the return value of .{op}() ({how}). In Plant 3D, "
            f".{op}() mutates the object it is called ON and its return value "
            f"is NOT a usable solid -- doing this in Plant 3D crashes the "
            f"script silently (TESTACPSCRIPT returns nil).\n"
            f"  WRONG:  merged = a.{op}(b); merged.translate(...)\n"
            f"  RIGHT:  a.{op}(b); b.erase(); a.translate(...)"
        )

    def __getattr__(self, name):
        self._blow_up(f"accessed .{name}")

    def __setattr__(self, name, value):
        self._blow_up(f"set .{name}")

    def __call__(self, *a, **k):
        self._blow_up("called it")

    def __bool__(self):
        # Allow "if result:" style checks without exploding; the poison is
        # falsy, like None.
        return False

    def __repr__(self):
        op = object.__getattribute__(self, '_op')
        return f"<consumed return value of .{op}() -- do not use>"


def _guard_alive(method):
    """Decorator: raise if the solid has been erased."""
    def wrapper(self, *a, **k):
        if not getattr(self, '_alive', True):
            raise ShimAPIError(
                f"Called .{method.__name__}() on an erased solid "
                f"({self.kind} {self.dims}). In Plant 3D, using an object "
                f"after .erase() is undefined behaviour and can crash "
                f"silently. Erase an object only when you are finished "
                f"with it."
            )
        return method(self, *a, **k)
    wrapper.__name__ = method.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
class _Solid:
    """A recorded primitive with dimensions and accumulated transform."""

    def __init__(self, scene, kind, dims):
        self.kind = kind          # 'box' | 'cylinder' | ...
        self.dims = dims          # dict of sizes
        self.M = np.eye(4)        # accumulated transform
        self.children = []        # absorbed geometry from uniteWith (snapshots)
        self.subtracted = []      # geometry removed via subtractFrom (for render)
        self.rendered = True      # set False by erase()
        self.absorbed = False     # set True when united into another solid
        self._alive = True        # set False by erase()
        self._scene = scene
        scene._add(self)

    # -- local-space AABB (for overlap checks) ------------------------------
    def _local_corners(self):
        if self.kind == 'box':
            L, W, H = self.dims['L'], self.dims['W'], self.dims['H']
            if BOX_CENTERED:
                lo = np.array([-L / 2, -W / 2, -H / 2])
                hi = np.array([L / 2, W / 2, H / 2])
            else:
                lo = np.zeros(3)
                hi = np.array([L, W, H])
        elif self.kind == 'cylinder':
            R, H = self.dims['R'], self.dims['H']
            ext = {'x': (H, R, R), 'y': (R, H, R), 'z': (R, R, H)}[CYLINDER_AXIS]
            # cylinder base at origin along its axis, radial extents +/- R
            lo = np.array([-ext[0] if CYLINDER_AXIS != 'x' else 0.0,
                           -ext[1] if CYLINDER_AXIS != 'y' else 0.0,
                           -ext[2] if CYLINDER_AXIS != 'z' else 0.0])
            lo = np.where([CYLINDER_AXIS == a for a in 'xyz'], 0.0,
                          [-ext[0], -ext[1], -ext[2]])
            hi = np.array(ext, dtype=float)
        else:
            return None
        corners = np.array([[x, y, z, 1.0]
                            for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        return corners

    def world_aabb(self):
        """World-space AABB of this solid (transformed local corners)."""
        corners = self._local_corners()
        if corners is None:
            return None
        wc = (self.M @ corners.T).T[:, :3]
        return wc.min(axis=0), wc.max(axis=0)

    # -- transforms (return self so calls can be chained) -------------------
    def _apply(self, T):
        self.M = (T @ self.M) if PREMULTIPLY else (self.M @ T)
        return self

    @_guard_alive
    def translate(self, vec):
        x, y, z = vec
        return self._apply(_translation(x, y, z))

    @_guard_alive
    def rotateX(self, deg):
        return self._apply(_rotation('x', deg))

    @_guard_alive
    def rotateY(self, deg):
        return self._apply(_rotation('y', deg))

    @_guard_alive
    def rotateZ(self, deg):
        return self._apply(_rotation('z', deg))

    # -- boolean ops (STRICT: poison return value) ---------------------------
    def _snapshot_other(self, other, into):
        """Fold other's geometry (and its children) into `into`, relative to self."""
        inv_self = np.linalg.inv(self.M)
        into.append({
            'kind': other.kind,
            'dims': other.dims,
            'rel': inv_self @ other.M,
        })
        for ch in other.children:
            into.append({
                'kind': ch['kind'],
                'dims': ch['dims'],
                'rel': inv_self @ other.M @ ch['rel'],
            })

    def _check_overlap(self, other, op):
        a, b = self.world_aabb(), other.world_aabb()
        if a is None or b is None:
            return
        eps = 1e-9
        overlaps = np.all(a[0] <= b[1] + eps) and np.all(b[0] <= a[1] + eps)
        if not overlaps:
            msg = (
                f".{op}(): operands do NOT physically overlap (AABB check).\n"
                f"  self  ({self.kind}):  min={np.round(a[0], 4)}  max={np.round(a[1], 4)}\n"
                f"  other ({other.kind}): min={np.round(b[0], 4)}  max={np.round(b[1], 4)}\n"
                f"In Plant 3D, uniteWith requires physical intersection -- "
                f"non-overlapping unions fail or produce broken solids."
            )
            if STRICT_OVERLAP:
                raise ShimAPIError(msg)
            warnings.warn(msg, stacklevel=3)

    @_guard_alive
    def uniteWith(self, other):
        if isinstance(other, _Consumed):
            other._blow_up("passed it to uniteWith")
        if not getattr(other, '_alive', True):
            raise ShimAPIError(
                "uniteWith(): the other object has already been erased."
            )
        self._check_overlap(other, 'uniteWith')
        self._snapshot_other(other, self.children)
        other.absorbed = True
        return _Consumed('uniteWith')

    @_guard_alive
    def subtractFrom(self, other):
        if isinstance(other, _Consumed):
            other._blow_up("passed it to subtractFrom")
        if not getattr(other, '_alive', True):
            raise ShimAPIError(
                "subtractFrom(): the other object has already been erased."
            )
        # No real boolean; record for renderers that want to show cutouts
        self._snapshot_other(other, self.subtracted)
        other.absorbed = True
        return _Consumed('subtractFrom')

    @_guard_alive
    def intersectWith(self, other):
        if isinstance(other, _Consumed):
            other._blow_up("passed it to intersectWith")
        if not getattr(other, '_alive', True):
            raise ShimAPIError(
                "intersectWith(): the other object has already been erased."
            )
        self._check_overlap(other, 'intersectWith')
        # No real boolean; keep self's shape, note the operand
        self._snapshot_other(other, self.subtracted)
        other.absorbed = True
        return _Consumed('intersectWith')

    def erase(self):
        """Mark dead. Any later method call on this object raises."""
        self.rendered = False
        self._alive = False
        return None    # real API gives you nothing back; don't chain off erase

    # -- convenience used by some scripts -----------------------------------
    @_guard_alive
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
LENGTH0 = 'LENGTH0'   # length that MAY be zero (real scripts use this)
ANGLE = 'ANGLE'
BOOL = 'BOOL'
STRING = 'STRING'
INT = 'INT'
DOUBLE = 'DOUBLE'
ENUM = 'ENUM'
_TYPES = {LENGTH, LENGTH0, ANGLE, BOOL, STRING, INT, DOUBLE, ENUM}

_REGISTRY = []        # functions decorated with @activate, in definition order

# If True, run_part() passes every parameter as a STRING, mimicking the Plant 3D
# properties palette. Scripts that don't float()-convert their params will fail
# here the same way they misbehave in Plant 3D.
STRING_PARAMS_MODE = False


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


def group(*a, **kw):
    """@group("MainDimensions") / @group(Name=...) -- parameter group. No-op here."""
    def deco(fn):
        groups = getattr(fn, '_p3d_groups', [])
        groups.append({'args': a, 'kwargs': kw})
        fn._p3d_groups = groups
        return fn
    return deco


def enum(*a, **kw):
    """@enum(1, "align X") -- enumeration entry for a preceding ENUM param. No-op."""
    def deco(fn):
        enums = getattr(fn, '_p3d_enums', [])
        enums.append({'args': a, 'kwargs': kw})
        fn._p3d_enums = enums
        return fn
    return deco


def activate(**kw):
    """@activate(LengthUnit='mm', ...) -- marks a function as the part entry point."""
    def deco(fn):
        fn._p3d_activate = kw
        _REGISTRY.append(fn)
        return fn
    return deco


def run_part(fn, scene=None, **overrides):
    """
    Execute a registered part function the way Plant 3D would.
    With STRING_PARAMS_MODE on, every default and override is stringified
    first -- exactly what the properties palette does -- so missing float()
    conversions blow up here instead of only in Plant 3D.
    """
    import inspect
    scene = scene or Scene()
    sig = inspect.signature(fn)
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == 's' or p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        val = overrides.get(name, p.default)
        if STRING_PARAMS_MODE and not isinstance(val, str):
            val = str(val)
        kwargs[name] = val
    fn(scene, **kwargs)
    return scene


# ---------------------------------------------------------------------------
# Module plumbing: build fake aqa / varmain modules and reset the registry
# ---------------------------------------------------------------------------
_PUBLIC = {
    'BOX': BOX, 'CYLINDER': CYLINDER,
    'param': param, 'activate': activate, 'group': group, 'enum': enum,
    'LENGTH': LENGTH, 'LENGTH0': LENGTH0, 'ANGLE': ANGLE, 'BOOL': BOOL,
    'STRING': STRING, 'INT': INT, 'DOUBLE': DOUBLE, 'ENUM': ENUM,
    'Scene': Scene,
}
_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}


def asRadiants(deg):
    """Plant 3D helper (note Autodesk's spelling): degrees -> radians."""
    return math.radians(deg)


def asDegrees(rad):
    return math.degrees(rad)


_MATH['asRadiants'] = asRadiants
_MATH['asDegrees'] = asDegrees


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