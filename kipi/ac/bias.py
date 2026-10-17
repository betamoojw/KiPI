"""DC bias on the capacitor models (#74).

A class II ceramic loses capacitance with the DC voltage across it. Models that estimate
DC bias are derated part by part, at the voltage the last DC solve found across each
part's pads; without one (or for a part off the DC solve's copper), at the rail's VRM set
point.

Models with a dielectric are also aged (#109), when the sweep settings say for how long.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from kipi.core.model.board import BoardSnapshot

# A pad's voltage is read at the nearest mesh node on its layers, if one is this close.
NEAREST_NODE_MM = 2.0


class _GroupLookup:
    """Mesh node voltages of one solved DC net group, found by place and layer."""

    def __init__(self, net_result):
        self.mesh = net_result.mesh
        self.v = net_result.solution.node_v
        self._trees: Dict[int, tuple] = {}

    def _layer(self, lid: int):
        if lid not in self._trees:
            from scipy.spatial import cKDTree
            idx = np.flatnonzero(self.mesh.node_layer == lid)
            self._trees[lid] = (cKDTree(self.mesh.points[idx]) if len(idx) else None, idx)
        return self._trees[lid]

    def pad_v(self, pad) -> float:
        """The pad's voltage: the mean over its own nodes when it was a terminal of the
        solve, else the nearest node on one of its layers; NaN if it was not meshed."""
        per_layer = self.mesh.pad_nodes.get(pad.id)
        if per_layer:
            v = self.v[np.concatenate(list(per_layer.values()))]
            return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        # A pad that only joins the net through a via may have no copper of its own in
        # the solve (an isolated pad is left out): then the copper under it.
        for layers in (pad.layers, self.mesh.layers):
            best = (NEAREST_NODE_MM, float("nan"))
            for lid in layers:
                tree, idx = self._layer(lid)
                if tree is None:
                    continue
                d, i = tree.query(pad.position)
                if d <= best[0] and np.isfinite(self.v[idx[i]]):
                    best = (d, float(self.v[idx[i]]))
            if np.isfinite(best[1]):
                return best[1]
        return float("nan")


def dc_volts(snap: BoardSnapshot, dc_result, caps: List[tuple]) -> Dict[str, float]:
    """The DC voltage across each capacitor in ``caps`` (as :func:`kipi.ac.checks.rail_caps`
    lists them) in the DC solve ``dc_result`` (a power-tree result). Parts whose rail pads
    the DC solve did not reach are left out; a ground the DC solve did not model is 0 V."""
    if dc_result is None or not getattr(dc_result, "groups", None):
        return {}
    by_net = {}
    for g in dc_result.groups.values():
        look = _GroupLookup(g.result)
        for n in g.nets:
            by_net[n] = look
    pads = {p.id: p for p in snap.pads}

    def mean_v(ids) -> Optional[float]:
        vs = [by_net[pads[i].net].pad_v(pads[i]) for i in ids
              if i in pads and pads[i].net in by_net]
        vs = [v for v in vs if np.isfinite(v)]
        return float(np.mean(vs)) if vs else None

    out: Dict[str, float] = {}
    for ref, _model, _net, a, b in caps:
        va = mean_v(a)
        if va is None:
            continue
        vb = mean_v(b) if pads.get(b[0]) is not None and pads[b[0]].net in by_net else 0.0
        if vb is not None:
            out[ref] = abs(va - vb)
    return out


def apply_derating(ports, setup, vrm, volts: Dict[str, float]
                   ) -> Tuple[Dict[str, Tuple[Optional[float], float]], List[str]]:
    """Derate the capacitor ports, in place, for DC bias (models that estimate it, at the
    voltage across each part: ``volts``, else the VRM's set point) and aging (#109, models
    with a dielectric, when the sweep settings age them). Each derated part gets its own
    copy of its model.

    Returns {reference: (DC volts or None, farads)} for the parts derated, and notes for
    the log.
    """
    s = setup.ac
    hours = s.aging_hours if s.aging else None
    out: Dict[str, Tuple[Optional[float], float]] = {}
    n_bias = from_dc = n_aged = 0
    kept: List[float] = []
    for port in ports:
        m = port.element
        if port.kind != "cap":
            continue
        biased = s.dc_bias and m.dc_bias and bool(m.rated_v)
        aged = m.aged(hours) < 1.0
        if not (biased or aged):
            continue
        v = None
        if biased:
            n_bias += 1
            v = volts.get(port.name)
            from_dc += v is not None
            v = abs(vrm.voltage) if v is None else v
        n_aged += aged
        c = m.capacitance_at(v or 0.0, hours)
        port.element = replace(m, capacitance_f=c)
        out[port.name] = (v, c)
        kept.append(c / m.capacitance_f if m.capacitance_f else 1.0)
    notes = []
    if out:
        what = []
        if n_bias:
            if from_dc == n_bias:
                where = "the voltages the last DC solve found across them"
            elif from_dc:
                where = (f"the last DC solve's voltages ({from_dc} of them) or else the "
                         f"{vrm.name} set point, {vrm.voltage:g} V")
            else:
                where = (f"the {vrm.name} set point, {vrm.voltage:g} V (no DC solve of this "
                         "rail)")
            what.append(f"{n_bias} for DC bias at {where}")
        if n_aged:
            what.append(f"{n_aged} aged {hours:g} hours")
        lo, hi = min(kept), max(kept)
        span = f"{lo:.0%}" if hi - lo < 0.005 else f"{lo:.0%} to {hi:.0%}"
        notes.append(f"Capacitors derated: {'; '.join(what)}. They keep {span} of their "
                     "capacitance")
    return out, notes
