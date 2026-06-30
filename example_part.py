# =============================================================================
#  example_part.py  --  MCA T-Shoe pipe support with two symmetric L-brackets
#
#  This is a WORKING sample written against the same API your real Plant 3D
#  scripts use. Three jobs:
#    1. Calibration reference -- open it in the previewer, compare to Plant 3D.
#       If they look identical, the shim knobs are correct and locked.
#    2. A fill-in template -- copy it, change the numbers, keep the structure.
#    3. Demonstrates fully independent sub-assemblies: the L-brackets have
#       their OWN position/height/width/thickness parameters (BRK_*) and
#       share nothing with the T-shoe's own D/L/W/H/T -- move or resize one
#       without touching the other.
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
@param(BRK_X = LENGTH, TooltipShort="L-bracket offset from centreline (position)")
@param(BRK_Z = LENGTH, TooltipShort="L-bracket vertical offset (position)")
@param(BRK_W = LENGTH, TooltipShort="L-bracket width (length along pipe)")
@param(BRK_H = LENGTH, TooltipShort="L-bracket height (web height)")
@param(BRK_T = LENGTH, TooltipShort="L-bracket steel thickness")
def mca(s, D=80.0, L=150.0, W=100.0, H=100.0, T=12.0,
        BRK_X=75.0, BRK_Z=0.0, BRK_W=40.0, BRK_H=60.0, BRK_T=8.0,
        ID="MCA", **kw):

    # -- T-shoe -------------------------------------------------------------
    # Geometry derived only from D/L/W/H/T -- the bracket block below never
    # reads these, so resizing the shoe never moves or resizes the brackets.
    FL, FW, FT = L, W, T                 # flange length / width / thickness
    SL, SW, SH = L, T, H - T             # stem  length / width / height

    stem   = BOX(s, L=SL, W=SW, H=SH).translate((0.0, 0.0, FT + SH / 2.0))
    flange = BOX(s, L=FL, W=FW, H=FT).translate((0.0, 0.0, FT / 2.0))
    shoe   = flange.uniteWith(stem)
    stem.erase()

    # -- two L-brackets, fully independent of the T-shoe ---------------------
    # BRK_X / BRK_Z  -> position (lateral offset / vertical offset)
    # BRK_W          -> width  (length along the pipe)
    # BRK_H          -> height (web height)
    # BRK_T          -> thickness (both the foot and the web)
    LEG = 40.0   # fixed foot reach, independent of every shoe dimension

    for side, x in (("right", -BRK_X), ("left", BRK_X)):
        foot = BOX(s, L=LEG, W=BRK_W, H=BRK_T).translate((0.0, 0.0, BRK_T / 2.0))
        web  = BOX(s, L=BRK_T, W=BRK_W, H=BRK_H).translate((-LEG / 2.0 + BRK_T / 2.0,
                                                            0.0, BRK_T + BRK_H / 2.0))
        bracket = foot.uniteWith(web)
        web.erase()
        bracket.translate((x, 0.0, BRK_Z))         # BRK_X / BRK_Z drive position

    # -- snap point: runs along the TOP of the stem (the pipe contact line) -
    s.setPoint((0.0, 0.0, H), (1.0, 0.0, 0.0))
