"""
runner.py
=========
Loads a Plant 3D part script, runs it against the shim, and returns the scene
(geometry) plus the part's parameters. Knows nothing about the GUI.

Usage:
    scene, specs, used = run_script("example_part.py")
    scene, specs, used = run_script("example_part.py", overrides={"S": 60.0})
"""

import sys
import types
import inspect

import plant3d_shim as shim


def _find_entry(namespace):
    """Pick the part function: prefer @activate, else the last function defined."""
    reg = shim.get_registry()
    if reg:
        return reg[-1]
    funcs = [v for v in namespace.values() if isinstance(v, types.FunctionType)]
    return funcs[-1] if funcs else None


def _param_specs(fn):
    """
    Build an ordered list of parameter specs for the UI, in signature order,
    skipping the first arg (the scene 's') and **kwargs. Merges any tooltip
    metadata declared via @param.
    """
    meta_by_name = {p['name']: p for p in getattr(fn, '_p3d_params', [])}
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    specs = []
    for p in params[1:]:                       # skip the scene argument
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        default = None if p.default is inspect._empty else p.default
        info = meta_by_name.get(p.name, {})
        specs.append({
            'name': p.name,
            'default': default,
            'type': info.get('type', 'LENGTH'),
            'tooltip': info.get('meta', {}).get('TooltipShort', ''),
            'numeric': isinstance(default, (int, float)),
        })
    return specs


def run_script(path, overrides=None):
    """
    Execute the part script fresh and return (scene, specs, used_values).
      scene  : shim.Scene with the built geometry
      specs  : list of {name, default, type, tooltip, numeric}
      used   : dict of the actual argument values passed in
    Raises on any error in the script (caller should catch and display it).
    """
    shim.reset()
    for name, module in shim.build_fake_modules().items():
        sys.modules[name] = module

    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()

    namespace = {}
    code = compile(source, path, 'exec')
    exec(code, namespace)                       # noqa: S102 - trusted, self-authored

    fn = _find_entry(namespace)
    if fn is None:
        raise RuntimeError("No part function found (expected a function, ideally "
                           "decorated with @activate).")

    specs = _param_specs(fn)

    # build call arguments: declared defaults, then any UI overrides on top
    used = {}
    for spec in specs:
        if spec['default'] is not None:
            used[spec['name']] = spec['default']
    if overrides:
        for k, v in overrides.items():
            if k in used or any(s['name'] == k for s in specs):
                used[k] = v

    scene = shim.Scene()
    sig = inspect.signature(fn)
    first_arg = list(sig.parameters)[0]         # the scene parameter name
    call_kwargs = {k: v for k, v in used.items() if k != first_arg}
    fn(scene, **call_kwargs)

    return scene, specs, used


if __name__ == "__main__":
    # quick self-test from the command line
    target = sys.argv[1] if len(sys.argv) > 1 else "example_part.py"
    sc, sp, uv = run_script(target)
    pstr = ", ".join("{0}={1}".format(s["name"], uv.get(s["name"])) for s in sp)
    print("Script   : {0}".format(target))
    print("Solids   : {0} drawn ({1} total)".format(len(sc.render_solids()), len(sc.solids)))
    print("Points   : {0} snap point(s)".format(len(sc.points)))
    print("Params   : {0}".format(pstr))
