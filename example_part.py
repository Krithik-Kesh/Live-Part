# =============================================================================
#  example_part.py  --  MCA T-Shoe pipe support with two symmetric L-brackets
# =============================================================================
from aqa.math import *
from varmain.primitiv import *
from varmain.var_basic import *
from varmain.custom import *


@activate(Group="Support", TooltipShort="T-shoe with L-brackets",
          TooltipLong="T-shoe pipe support with two independent symmetric L-brackets",
          LengthUnit="mm", Ports="1")
@group("Shoe")
@param(D = LENGTH, TooltipShort="Pipe outside diameter")
@param(L = LENGTH, TooltipShort="Shoe length (along pipe)")
@param(W = LENGTH, TooltipShort="Flange width")
@param(H = LENGTH, TooltipShort="Overall height")
@param(T = LENGTH, TooltipShort="Steel thickness")
@group("Brackets")
@param(BRK_X = LENGTH, TooltipShort="L-bracket offset from centreline (position)")
@param(BRK_Z = LENGTH0, TooltipShort="L-bracket vertical offset (position)")
@param(BRK_W = LENGTH, TooltipShort="L-bracket width (length along pipe)")
@param(BRK_H = LENGTH, TooltipShort="L-bracket height (web height)")
@param(BRK_T = LENGTH, TooltipShort="L-bracket steel thickness")
def mca(s, D=80.0, L=150.0, W=100.0, H=100.0, T=12.0,
        BRK_X=75.0, BRK_Z=0.0, BRK_W=40.0, BRK_H=60.0, BRK_T=8.0,
        ID="MCA", **kw):

    # -- palette-proofing: the properties palette can hand these in as strings
    D = float(D); L = float(L); W = float(W); H = float(H); T = float(T)
    BRK_X = float(BRK_X); BRK_Z = float(BRK_Z)
    BRK_W = float(BRK_W); BRK_H = float(BRK_H); BRK_T = float(BRK_T)

    # -- T-shoe -------------------------------------------------------------
    # Geometry derived only from D/L/W/H/T -- the bracket block below never
    # reads these, so resizing the shoe never moves or resizes the brackets.
    FL, FW, FT = L, W, T                 # flange length / width / thickness
    SL, SW, SH = L, T, H - T             # stem  length / width / height

    stem   = BOX(s, L=SL, W=SW, H=SH).translate((0.0, 0.0, FT + SH / 2.0))
    flange = BOX(s, L=FL, W=FW, H=FT).translate((0.0, 0.0, FT / 2.0))
    flange.uniteWith(stem)               # flange now IS the whole shoe
    stem.erase()                         # stem is gone -- never touch it again

    # -- two L-brackets, fully independent of the T-shoe ---------------------
    # BRK_X / BRK_Z  -> position (lateral offset / vertical offset)
    # BRK_W          -> width  (length along the pipe)
    # BRK_H          -> height (web height)
    # BRK_T          -> thickness (both the foot and the web)
    LEG = 40.0   # fixed foot reach, independent of every shoe dimension

    for x, mirror in ((-BRK_X, 0.0), (BRK_X, 180.0)):
        # Build each bracket at the origin: foot flat on Z=0, web rising from
        # the foot's -X end. Web overlaps the foot's full thickness so the L
        # stays welded (uniteWith needs physical overlap) for any BRK_* value.
        foot = BOX(s, L=LEG, W=BRK_W, H=BRK_T).translate((0.0, 0.0, BRK_T / 2.0))
        web  = BOX(s, L=BRK_T, W=BRK_W, H=BRK_H).translate(
                   (-LEG / 2.0 + BRK_T / 2.0, 0.0, BRK_T + BRK_H / 2.0))
        foot.uniteWith(web)              # foot now IS the whole L-bracket
        web.erase()
        foot.rotateZ(mirror)             # 180 on the +X side -> true mirror pair
        foot.translate((x, 0.0, BRK_Z))  # BRK_X / BRK_Z drive position

    # -- snap point: runs along the TOP of the stem (the pipe contact line) -
    s.setPoint((0.0, 0.0, H), (1.0, 0.0, 0.0))