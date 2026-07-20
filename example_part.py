#  example_part.py-  MCA T-Shoe pipe support with two symmetric L-brackets
#
#  Plant 3D test command:  (TESTACPSCRIPT "mca")
from aqa.math import *
from varmain.primitiv import *
from varmain.var_basic import *
from varmain.custom import *


@activate(Group="Support", TooltipShort="T-shoe with L-brackets",
          TooltipLong="T-shoe pipe support with two independent symmetric L-brackets",
          LengthUnit="mm")
@group("Shoe")
@param(D=LENGTH, TooltipShort="Pipe outside diameter")
@param(L=LENGTH, TooltipShort="Shoe length (along pipe)")
@param(W=LENGTH, TooltipShort="Flange width")
@param(H=LENGTH, TooltipShort="Overall height")
@param(T=LENGTH, TooltipShort="Steel thickness")
@group("Brackets")
@param(BRK_X=LENGTH, TooltipShort="L-bracket offset from centreline")
@param(BRK_Z=LENGTH0, TooltipShort="L-bracket vertical offset")
@param(BRK_W=LENGTH, TooltipShort="L-bracket width (length along pipe)")
@param(BRK_H=LENGTH, TooltipShort="L-bracket height (web height)")
@param(BRK_T=LENGTH, TooltipShort="L-bracket steel thickness")
def mca(s, D=80.0, L=150.0, W=100.0, H=100.0, T=12.0,
        BRK_X=75.0, BRK_Z=0.0, BRK_W=40.0, BRK_H=60.0, BRK_T=8.0, **kw):

    FL, FW, FT = L, W, T
    SL, SW, SH = L, T, H - T

    stem   = BOX(s, L=SL, W=SW, H=SH).translate((0.0, 0.0, FT + SH / 2.0))
    flange = BOX(s, L=FL, W=FW, H=FT).translate((0.0, 0.0, FT / 2.0))
    flange.uniteWith(stem)
    stem.erase()

    # Pipe saddle: a cylinder the same OD as the pipe is subtracted from the
    # top of the stem so the shoe cradles any pipe size exactly.
    # Axis is along X (pipe direction); centre at Z=H (the pipe centreline).
    saddle = CYLINDER(s, R=D / 2.0, H=L + 2.0 * T).rotateY(90.0).translate((0.0, 0.0, H))
    flange.subtractFrom(saddle)
    saddle.erase()

    LEG = 40.0
    for x, mirror in ((-BRK_X, 0.0), (BRK_X, 180.0)):
        foot = BOX(s, L=LEG, W=BRK_W, H=BRK_T).translate((0.0, 0.0, BRK_T / 2.0))
        web  = BOX(s, L=BRK_T, W=BRK_W, H=BRK_H).translate(
                   (-LEG / 2.0 + BRK_T / 2.0, 0.0, BRK_T + BRK_H / 2.0))
        foot.uniteWith(web)
        web.erase()
        foot.rotateZ(mirror)
        foot.translate((x, 0.0, BRK_Z))

    # Snap point: pipe centreline at the top of the saddle (Z=H, pipe runs along X)
    s.setPoint((0.0, 0.0, H), (1.0, 0.0, 0.0))