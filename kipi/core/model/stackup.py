"""Building a :class:`~kipi.core.model.board.Stackup` from the layers a board file lists,
or a default one when the file does not give a complete stackup."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from kipi.core.model.board import CopperLayer, DielectricLayer, Stackup
from kipi.core.progress import NULL, ProgressReporter

DEFAULT_BOARD_THICKNESS_MM = 1.6
DEFAULT_OUTER_CU_MM = 0.035
DEFAULT_INNER_CU_MM = 0.0175

# One stackup layer, top to bottom: ("cu", copper layer id, thickness_mm, "", 0.0) or
# ("diel", -1, thickness_mm, material, epsilon_r).
Entry = Tuple[str, int, float, str, float]


def default_entries(cu_ids: Sequence[int]) -> List[Entry]:
    """A :data:`DEFAULT_BOARD_THICKNESS_MM` board: 1 oz outer and 1/2 oz inner copper,
    with the rest split evenly into FR4 dielectrics."""
    n = len(cu_ids)
    cu_t = [DEFAULT_OUTER_CU_MM if i in (0, n - 1) else DEFAULT_INNER_CU_MM for i in range(n)]
    diel_t = (DEFAULT_BOARD_THICKNESS_MM - sum(cu_t)) / max(1, n - 1)
    entries: List[Entry] = []
    for i, lid in enumerate(cu_ids):
        entries.append(("cu", lid, cu_t[i], "", 0.0))
        if i < n - 1:
            entries.append(("diel", -1, diel_t, "FR4", 4.5))
    return entries


def build_stackup(entries: Sequence[Entry], cu_ids: Sequence[int], names: Dict[int, str],
                  progress: ProgressReporter = NULL) -> Stackup:
    """The stackup of ``entries`` (top to bottom), or the default stackup for ``cu_ids``
    when ``entries`` does not hold every copper layer. ``names`` maps layer id to the name
    shown for it."""
    from_board = sum(1 for e in entries if e[0] == "cu") == len(cu_ids) and len(cu_ids) > 0
    if not from_board:
        progress.warn("Stackup incomplete; synthesizing a default "
                      f"{DEFAULT_BOARD_THICKNESS_MM} mm stackup")
        entries = default_entries(cu_ids)

    total = sum(e[2] for e in entries)
    z = total
    copper: List[CopperLayer] = []
    dielectrics: List[DielectricLayer] = []
    last_cu = -1
    pending_diel: List[DielectricLayer] = []
    for kind, lid, t, mat, er in entries:
        z -= t
        if kind == "cu":
            copper.append(CopperLayer(id=lid, name=names.get(lid, str(lid)), index=len(copper),
                                      thickness_mm=t, z_bottom_mm=z))
            for d in pending_diel:
                d.below_layer = lid
            pending_diel = []
            last_cu = lid
        else:
            d = DielectricLayer(thickness_mm=t, material=mat, epsilon_r=er, above_layer=last_cu)
            dielectrics.append(d)
            pending_diel.append(d)

    progress.info(f"Stackup: {len(copper)} copper layers, {total:.3f} mm total" +
                  ("" if from_board else " (default)"))
    for c in copper:
        progress.info(f"  {c.name:<8} t={c.thickness_mm * 1000:.1f} um  z={c.z_mid_mm:.3f} mm")
    return Stackup(copper=copper, dielectrics=dielectrics, board_thickness_mm=total,
                   from_board=from_board)
