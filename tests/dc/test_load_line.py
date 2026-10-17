"""VRM load line (#94): the regulated voltage droops by R_LL times the output current its
connected loads (and the VRMs it feeds) actually draw."""

import pytest

from kipi.dc.pi.model import Load, PinRef, PISetup, Vrm, validate
from kipi.dc.pi.tree import solve_tree
from tree_boards import R, board, by_name, pad, setup


def test_load_line_droops_the_output_pins():
    s = setup("ldo")
    s.vrms[1].load_line_mohm = 10.0
    res = solve_tree(board(), s)
    u1, ld = by_name(res.vrms, "U1"), by_name(res.loads, "U2")
    assert u1.v_droop == pytest.approx(0.01, rel=1e-9)
    assert u1.v_target == pytest.approx(3.29, rel=1e-9)
    assert u1.v_out == pytest.approx(3.29, rel=1e-9)
    assert ld.v_supply == pytest.approx(3.29 - R(8), rel=1e-9)
    # Drop is still measured from the set point, so the droop counts against tolerance.
    assert ld.nominal == 3.3 and ld.drop == pytest.approx(0.01 + R(8), rel=1e-9)


def test_load_line_with_remote_sense_droops_the_sense_pins():
    s = setup("ldo", sense=True)
    s.vrms[1].load_line_mohm = 10.0
    u1 = by_name(solve_tree(board(), s).vrms, "U1")
    assert u1.v_sense == pytest.approx(3.29, abs=1e-9)
    assert u1.v_out == pytest.approx(3.29 + R(8), rel=1e-9)


def test_load_line_follows_the_downstream_vrm_input_current():
    # J1 feeds a switcher, whose input current depends on the voltage J1's droop leaves.
    s = setup("smps", eff=0.85, iq=0.01)
    s.vrms[0].load_line_mohm = 100.0
    res = solve_tree(board(), s)
    j1, u1 = by_name(res.vrms, "J1"), by_name(res.vrms, "U1")
    assert j1.i_out == pytest.approx(u1.i_in, rel=1e-9)
    assert j1.v_droop == pytest.approx(0.1 * j1.i_out, rel=1e-9)
    assert j1.v_out == pytest.approx(12.0 - 0.1 * j1.i_out, rel=1e-9)
    assert u1.v_in == pytest.approx(12.0 - (0.1 + R(8)) * u1.i_in, rel=1e-8)


def test_disconnected_loads_add_no_droop():
    from shapely.geometry import box
    snap = board()
    snap.pads.append(pad("u3_1", "+3V3", "U3", "1", box(30, 0, 31, 1)))   # an island
    s = setup("ldo")
    s.vrms[1].load_line_mohm = 10.0
    s.loads.append(Load("U3", PinRef("U3", "+3V3", ["1"]), 5.0))
    u1 = by_name(solve_tree(snap, s).vrms, "U1")
    assert u1.i_out == pytest.approx(1.0, rel=1e-9)
    assert u1.v_droop == pytest.approx(0.01, rel=1e-9)


def test_negative_rail_droops_toward_zero():
    s = setup()
    s.vrms = [Vrm("U1", PinRef("U1", "+3V3", ["2"]), -3.3, "source", load_line_mohm=10.0)]
    s.loads[0].current = -1.0
    u1 = by_name(solve_tree(board(), s).vrms, "U1")
    assert u1.v_out == pytest.approx(-3.29, rel=1e-9)


def test_no_load_line_leaves_the_set_point():
    u1 = by_name(solve_tree(board(), setup("ldo")).vrms, "U1")
    assert u1.v_droop == 0.0 and u1.v_out == pytest.approx(3.3, abs=1e-12)


def test_load_line_is_saved_and_checked():
    s = setup("ldo")
    s.vrms[1].load_line_mohm = 1.5
    assert PISetup.from_dict(s.to_dict()).vrms[1].load_line_mohm == 1.5
    old = s.to_dict()
    del old["vrms"][1]["load_line_mohm"]                   # a setup from before #94
    assert PISetup.from_dict(old).vrms[1].load_line_mohm == 0.0
    s.vrms[1].load_line_mohm = -1.0
    assert any("load line cannot be negative" in e for e in validate(s))


def test_old_results_read_without_a_load_line():
    import pickle
    u1 = by_name(solve_tree(board(), setup("ldo")).vrms, "U1")
    state = dict(u1.__dict__)
    del state["load_line_mohm"], state["v_droop"]
    old = object.__new__(type(u1))
    old.__setstate__(state)
    assert old.v_droop == 0.0 and old.v_target == old.v_set
    assert pickle.loads(pickle.dumps(old)).v_target == old.v_set


def test_reports_show_the_load_line(tmp_path):
    import csv
    from kipi.dc.pi.report import export_csv
    snap = board()
    s = setup("ldo")
    s.vrms[1].load_line_mohm = 10.0
    tree = solve_tree(snap, s)
    export_csv(tree, snap, tmp_path)
    with open(tmp_path / "kipi_vrms.csv", encoding="utf-8") as fh:
        row = next(r for r in csv.DictReader(fh) if r["vrm"] == "U1")
    assert float(row["load_line_mOhm"]) == 10.0
    assert float(row["v_target_V"]) == pytest.approx(3.29, rel=1e-9)


def test_editor_and_notes_show_the_load_line():
    import os
    pytest.importorskip("qtpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    from kipi.dc.gui.pi_panel import PIPanel
    _app = QApplication.instance() or QApplication([])  # noqa: F841
    s = setup("ldo")
    panel = PIPanel()
    panel.set_board(board(), s, "test")
    panel.vrm_list.lst.setCurrentRow(1)
    ed = panel.vrm_list.editor
    assert ed.load_line.text() == "none"
    ed.load_line.setValue(10.0)
    assert s.vrms[1].load_line_mohm == 10.0
    tree = solve_tree(board(), s)
    panel.show_results(tree, next(iter(tree.groups)))
    notes = panel.txt_notes.toPlainText()
    assert "U1 load line: 10 mΩ × 1 A = 10 mV droop, regulating to 3.29000 V" in notes
    panel.close()


def test_multiphase_load_line_droops_the_mean_phase_output():
    from phase_boards import board as phase_board, two_phase
    s = two_phase()
    s.vrms[1].load_line_mohm = 5.0
    u1 = by_name(solve_tree(phase_board(), s).vrms, "U1")
    assert u1.v_droop == pytest.approx(5e-3 * u1.i_out, rel=1e-9)
    assert u1.v_out == pytest.approx(u1.v_set - u1.v_droop, abs=1e-9)   # mean of the phases
