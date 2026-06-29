# =============================================================================
#  example_part.py  --  MCA T-Shoe pipe support with two symmetric L-guides
#
#  This is a WORKING sample written against the same API your real Plant 3D
#  scripts use. Two jobs:
#    1. Calibration reference -- open it in the previewer, compare to Plant 3D.
#       If they look identical, the shim knobs are correct and locked.
#    2. A fill-in template -- copy it, change the numbers, keep the structure.
#
#  All dimensions are in millimetres. Change a value, save, watch it update.
# =============================================================================
from aqa.math import *
from varmain.primitiv import *
from varmain.var_basic import *
from varmain.custom import *


@activate(LengthUnit="mm")
@param(D = LENGTH, TooltipShort="Pipe outside diameter")
@param(L = LENGTH, TooltipShort="Shoe length (along pipe)")
@param(W = LENGTH, TooltipShort="Flange width")
@param(H = LENGTH, TooltipShort="Overall height")
@param(T = LENGTH, TooltipShort="Steel thickness")
@param(S = LENGTH, TooltipShort="L-guide offset from flange centreline")
def mca(s, D=80.0, L=150.0, W=100.0, H=100.0, T=12.0, S=75.0, ID="MCA", **kw):

    # -- T-shoe -------------------------------------------------------------
    # Geometry derived from parameters so changing any value scales cleanly
    # (no raw literals mixed in -> stays correct in any unit setting).
    FL, FW, FT = L, W, T                 # flange length / width / thickness
    SL, SW, SH = L, T, H - T             # stem  length / width / height

    stem   = BOX(s, L=SL, W=SW, H=SH).translate((0.0, 0.0, FT + SH / 2.0))
    flange = BOX(s, L=FL, W=FW, H=FT).translate((0.0, 0.0, FT / 2.0))
    shoe   = flange.uniteWith(stem)
    stem.erase()

    # -- two L-guides, one each side, distance S from the centreline --------
    GLEN, GFB, GWT, GWH = 40.0, 40.0, T, 60.0     # leg / breadth / web / height

    for side, x in (("right", -S), ("left", S)):
        foot = BOX(s, L=GFB, W=GLEN, H=GWT).translate((0.0, 0.0, GWT / 2.0))
        web  = BOX(s, L=GWT, W=GLEN, H=GWH).translate((-GFB / 2.0 + GWT / 2.0,
                                                       0.0, GWT + GWH / 2.0))
        guide = foot.uniteWith(web)
        web.erase()
        guide.translate((x, 0.0, 0.0))            # S drives the offset

    # -- snap point: runs along the TOP of the stem (the pipe contact line) -
    s.setPoint((0.0, 0.0, H), (1.0, 0.0, 0.0))
