"""Pad copper from a ``.kicad_pcb`` file, shaped as KiCad shapes it.

KiCad turns a pad into a polygon (``PAD::TransformShapeToPolygon``) with its corners and
ends cut into straight segments no further than the board's maximum error from the true
curve, every vertex on the curve. The same rules here give the polygons the KiCad API
returns for the same pad, so a board reads the same from the file as from KiCad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from kipi.core.geometry.copper import shapely_to_polys
from kipi.core.model.board import Poly
from kipi.sources.kicad_pcb.items import NM, Frame, drawn_copper, stroke_width
from kipi.sources.kicad_pcb.sexpr import Node, atoms, child, children, flag, value, xy

DEFAULT_MAX_ERROR_MM = 0.005
MIN_CIRCLE_SEGMENTS = 8
# Rounded corners are cut by Clipper's offset with at least this many segments per circle.
MIN_CORNER_CIRCLE_SEGMENTS = 16

_TYPES = {"thru_hole": "pth", "smd": "smd", "connect": "edge", "np_thru_hole": "npth"}
_CHAMFER_CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


def pad_type(node: Node) -> str:
    return _TYPES.get(node[2] if len(node) > 2 else "", "unknown")


def arc_segments(radius: float, max_error: float, angle_deg: float) -> int:
    """How many segments KiCad cuts an arc into (``GetArcToSegmentCount``)."""
    if radius <= 0:
        return 2
    rel = min(max_error / radius, 1.0)
    step = math.degrees(math.acos(1.0 - rel)) * 2.0
    step = min(360.0 / MIN_CIRCLE_SEGMENTS, step) if step > 0 else 360.0 / MIN_CIRCLE_SEGMENTS
    return max(2, int(round(abs(angle_deg) / step)))


def circle_segments(radius: float, max_error: float) -> int:
    """Segments in a pad's full circle: KiCad rounds the count up to a multiple of 8, so
    the polygon lines up at 45 degrees."""
    n = arc_segments(radius, max_error, 360.0)
    return max(MIN_CIRCLE_SEGMENTS, (n + 7) // 8 * 8)


def circle_points(cx: float, cy: float, r: float, max_error: float) -> List[tuple]:
    n = circle_segments(r, max_error)
    step = 2 * math.pi / n
    return [(cx + r * math.cos(step * (i + 0.5)), cy + r * math.sin(step * (i + 0.5)))
            for i in range(n)]


def corner_segments(radius: float, max_error: float) -> int:
    """Segments in a rounded corner. KiCad rounds a rectangle's corners by growing a
    smaller one with Clipper, which steps ``ceil(n / 4)`` times per quarter circle."""
    n = max(MIN_CORNER_CIRCLE_SEGMENTS, arc_segments(radius, max_error, 360.0))
    return max(1, math.ceil(n / 4.0 - 1e-9))


def _corner_arc(cx, cy, r, start_deg, max_error) -> List[tuple]:
    """A quarter circle from ``start_deg`` (math angle, y down) on, both ends included."""
    n = corner_segments(r, max_error)
    return [(cx + r * math.cos(math.radians(start_deg + 90.0 * i / n)),
             cy + r * math.sin(math.radians(start_deg + 90.0 * i / n))) for i in range(n + 1)]


def rounded_rect(w: float, h: float, radius: float, chamfer: float,
                 chamfered: Sequence[str], max_error: float) -> List[tuple]:
    """A ``w`` x ``h`` rectangle centered on the origin with corners rounded by
    ``radius`` and the ``chamfered`` ones cut by ``chamfer`` instead."""
    hw, hh = w / 2.0, h / 2.0
    radius = max(0.0, min(radius, hw, hh))
    chamfer = max(0.0, min(chamfer, hw, hh) if chamfered else 0.0)
    # Corners clockwise on screen (y down): top-left, top-right, bottom-right, bottom-left;
    # each with its x/y signs and the math angle its rounding starts at.
    corners = (("top_left", -1, -1, 180.0), ("top_right", 1, -1, 270.0),
               ("bottom_right", 1, 1, 0.0), ("bottom_left", -1, 1, 90.0))
    pts: List[tuple] = []
    for name, sx, sy, start in corners:
        x, y = sx * hw, sy * hh
        if chamfer > 0 and name in chamfered:
            # Coming from the previous corner, then leaving toward the next one.
            if sx == sy:       # top-left, bottom-right: arrive along y, leave along x
                pts += [(x, y - sy * chamfer), (x - sx * chamfer, y)]
            else:
                pts += [(x - sx * chamfer, y), (x, y - sy * chamfer)]
        elif radius > 0:
            pts += _corner_arc(x - sx * radius, y - sy * radius, radius, start, max_error)
        else:
            pts.append((x, y))
    return pts


def oval(w: float, h: float, max_error: float) -> List[tuple]:
    """A stadium: a ``min(w, h)`` wide segment along the long side with round ends. Each
    end runs from one side to the other through points half a step off its ends, as
    KiCad cuts it."""
    r = min(w, h) / 2.0
    half = abs(w - h) / 2.0
    if half == 0:
        return circle_points(0.0, 0.0, r, max_error)
    m = circle_segments(r, max_error) // 2
    step = 180.0 / m
    pts: List[tuple] = []
    for end, base in ((half, -90.0), (-half, 90.0)):      # along x: right end, left end
        angles = [base] + [base + step * (i + 0.5) for i in range(m)] + [base + 180.0]
        pts += [(end + r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
                for a in angles]
    if h > w:                                             # along y instead: turn it
        pts = [(-y, x) for x, y in pts]
    return pts


def trapezoid(w: float, h: float, dx: float, dy: float) -> List[tuple]:
    """KiCad's trapezoid: ``rect_delta`` widens one side and narrows the other."""
    hx, hy, ddx, ddy = w / 2.0, h / 2.0, dx / 2.0, dy / 2.0
    return [(-hx + ddy, -hy - ddx), (hx - ddy, -hy + ddx),
            (hx + ddy, hy - ddx), (-hx - ddy, hy + ddx)]


@dataclass
class PadLayerShape:
    """How a pad is shaped on one copper layer."""
    shape: str
    size: Tuple[float, float]
    offset: Tuple[float, float] = (0.0, 0.0)
    rratio: float = 0.0
    chamfer_ratio: float = 0.0
    chamfered: List[str] = field(default_factory=list)
    delta: Tuple[float, float] = (0.0, 0.0)
    anchor: str = "circle"
    primitives: Optional[Node] = None


def _shape_from(node: Node, base: Optional[PadLayerShape], shape_name: Optional[str]):
    """The shape ``node`` gives (the pad itself, or a padstack layer), over ``base``."""
    b = base or PadLayerShape(shape="circle", size=(0.0, 0.0))
    size = child(node, "size")
    off = child(node, "offset")
    rr = value(node, "roundrect_rratio")
    cr = value(node, "chamfer_ratio")
    ch = child(node, "chamfer")
    delta = child(node, "rect_delta")
    opts = child(node, "options")
    anchor = value(opts, "anchor") if opts is not None else None
    return PadLayerShape(
        shape=shape_name or value(node, "shape") or b.shape,
        size=xy(size) if size is not None else b.size,
        offset=xy(off) if off is not None else b.offset,
        rratio=float(rr) if rr is not None else b.rratio,
        chamfer_ratio=float(cr) if cr is not None else b.chamfer_ratio,
        chamfered=[a for a in atoms(ch) if a in _CHAMFER_CORNERS] if ch is not None
        else b.chamfered,
        delta=xy(delta) if delta is not None else b.delta,
        anchor=anchor or b.anchor,
        primitives=child(node, "primitives") or b.primitives)


def drill(node: Node) -> Tuple[Tuple[float, float], Tuple[float, float], bool]:
    """``((x, y) size, (x, y) shape offset, oblong)`` of the pad's hole."""
    d = child(node, "drill")
    if d is None:
        return (0.0, 0.0), (0.0, 0.0), False
    nums = [float(a) for a in atoms(d) if a != "oval"]
    oblong = "oval" in atoms(d)
    if not nums:
        size = (0.0, 0.0)
    elif oblong and len(nums) > 1:
        size = (nums[0], nums[1])
    else:
        size = (nums[0], nums[0])
    off = child(d, "offset")
    return size, (xy(off) if off is not None else (0.0, 0.0)), oblong


class PadShapes:
    """The shapes of one pad, per copper layer name."""

    def __init__(self, node: Node):
        self.node = node
        main = _shape_from(node, None, node[3] if len(node) > 3 else "circle")
        _, offset, _ = drill(node)
        if child(node, "offset") is None and offset != (0.0, 0.0):
            main.offset = offset
        self.main = main
        self.mode = "normal"
        self.layers: Dict[str, PadLayerShape] = {}
        stack = child(node, "padstack")
        if stack is not None:
            self.mode = value(stack, "mode", "normal")
            for layer in children(stack, "layer"):
                if len(layer) > 1:
                    self.layers[layer[1]] = _shape_from(layer, main, None)

    def on(self, layer_name: str) -> PadLayerShape:
        if self.mode == "custom":
            return self.layers.get(layer_name, self.main)
        if self.mode == "front_inner_back":
            if layer_name == "B.Cu":
                return self.layers.get("B.Cu", self.main)
            if layer_name != "F.Cu":
                return self.layers.get("Inner", self.main)
        return self.main


def shape_points(s: PadLayerShape, max_error: float):
    """The outline of a non-custom pad shape in the pad's frame, centered on its offset;
    None for a custom shape."""
    w, h = s.size
    ox, oy = s.offset
    if s.shape == "circle":
        pts = circle_points(0.0, 0.0, w / 2.0, max_error)
    elif s.shape == "rect":
        pts = rounded_rect(w, h, 0.0, min(w, h) * s.chamfer_ratio, s.chamfered, max_error) \
            if s.chamfered and s.chamfer_ratio > 0 else \
            [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    elif s.shape in ("roundrect", "chamfered_rect"):
        radius = min(w, h) * s.rratio if s.shape == "roundrect" or s.rratio > 0 else 0.0
        pts = rounded_rect(w, h, radius, min(w, h) * s.chamfer_ratio, s.chamfered, max_error)
    elif s.shape == "oval":
        pts = oval(w, h, max_error)
    elif s.shape == "trapezoid":
        pts = trapezoid(w, h, *s.delta)
    else:
        return None
    return [(x + ox, y + oy) for x, y in pts]


def _custom(s: PadLayerShape, frame: Frame, max_error: float):
    """A custom pad: its anchor plus every primitive, as one shapely geometry."""
    from shapely.geometry import Polygon as SPolygon
    from shapely.ops import unary_union
    anchor = PadLayerShape(shape="rect" if s.anchor == "rect" else "circle", size=s.size,
                           offset=s.offset)
    parts = [SPolygon(frame.points(shape_points(anchor, max_error)))]
    if s.primitives is not None:
        shifted = Frame(frame(s.offset), frame.angle_deg)
        for prim in s.primitives[1:]:
            if type(prim) is not list:
                continue
            # A primitive drawn with no stroke is filled: KiCad has no use for an
            # outline of no width.
            if prim[0] in ("gr_poly", "gr_circle", "gr_rect") and child(prim, "fill") is None:
                prim = prim + [["fill", "yes"]]
            if prim[0] == "gr_circle" and stroke_width(prim) == 0:
                # A filled circle is cut like a round pad.
                c, e = xy(child(prim, "center")), xy(child(prim, "end"))
                pts = circle_points(c[0], c[1], math.dist(c, e), max_error)
                parts.append(SPolygon(shifted.points(pts)))
                continue
            g = drawn_copper(prim, shifted)
            if g is not None and not g.is_empty:
                parts.append(g)
    return unary_union(parts)


def copper(shapes: PadShapes, layer_name: str, frame: Frame, max_error: float) -> List[Poly]:
    """The pad's copper on ``layer_name``, placed by ``frame`` (the pad on the board)."""
    s = shapes.on(layer_name)
    if s.size[0] <= 0 and s.size[1] <= 0 and s.shape != "custom":
        return []
    if s.shape == "custom":
        return [Poly(outline=[_nm(p) for p in poly.outline],
                     holes=[[_nm(p) for p in h] for h in poly.holes])
                for poly in shapely_to_polys(_custom(s, frame, max_error))]
    pts = shape_points(s, max_error)
    if not pts or len(pts) < 3:
        return []
    return [Poly(outline=frame.points(pts))]


def _nm(p) -> tuple:
    return (round(p[0], NM), round(p[1], NM))


def npth_has_copper(shapes: PadShapes, layer_name: str, node: Node) -> bool:
    """A non-plated hole has copper only where its pad is bigger than the hole
    (``PAD::FlashLayer``)."""
    s = shapes.on(layer_name)
    size, _, oblong = drill(node)
    if s.offset != (0.0, 0.0):
        return True
    if s.shape == "circle" and not oblong:
        return size[0] < s.size[0]
    if s.shape == "oval" and oblong:
        return size[0] < s.size[0] or size[1] < s.size[1]
    return True


def removes_unused_layers(node: Node) -> Tuple[bool, bool]:
    """``(remove_unused_layers, keep_end_layers)``."""
    return flag(node, "remove_unused_layers"), flag(node, "keep_end_layers")
