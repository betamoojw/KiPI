"""KiPI started by KiCad (plugin) or on its own (standalone): how it tells, and what each
looks like."""

import os
import time

import pytest

pytest.importorskip("qtpy")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QMessageBox, QToolBar  # noqa: E402

from kipi import host  # noqa: E402
from kipi.core.model.snapshot_io import save  # noqa: E402
from kipi.host import Host  # noqa: E402
from synthetic import make_board  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def board_file(tmp_path):
    snap = make_board()
    snap.project_dir = str(tmp_path)
    return save(snap, tmp_path / "board.kipi.json.gz")


def _idle(app, w, timeout=30.0):
    """Wait until the window stays idle: a task's thread stops before its result reaches
    the window, and the result can start the next task."""
    t0, quiet = time.time(), 0
    while time.time() - t0 < timeout:
        app.processEvents()
        quiet = quiet + 1 if not w._busy() and not w._sync_timer.isActive() else 0
        if quiet >= 10:
            return
        time.sleep(0.01)
    raise AssertionError("window did not go idle")


def _toolbar(w):
    return [a.text() for a in w.findChild(QToolBar).actions() if a.isVisible() and a.text()]


# --- Telling them apart ------------------------------------------------------------------

def test_kicad_starting_us_means_plugin():
    assert host.detect({}) is Host.STANDALONE
    assert host.detect({"KICAD_API_SOCKET": "ipc:///tmp/kicad/api.sock"}) is Host.PLUGIN
    assert host.detect({"KICAD_API_SOCKET": ""}) is Host.STANDALONE


def test_the_override_wins():
    assert host.detect({"KIPI_HOST": "plugin"}) is Host.PLUGIN
    assert host.detect({"KIPI_HOST": " Standalone ",
                        "KICAD_API_SOCKET": "ipc:///x"}) is Host.STANDALONE
    assert host.detect({"KIPI_HOST": "nonsense"}) is Host.STANDALONE   # ignored


def test_the_entry_point_decides():
    assert host.current() is Host.STANDALONE            # detected (see conftest)
    host.set_current(Host.PLUGIN)
    assert host.current() is Host.PLUGIN


def test_plugin_and_standalone_lock_apart():
    from kipi.gui import single_instance as si
    standalone = (si.lock_path(), si.server_name())
    host.set_current(Host.PLUGIN)
    plugin = (si.lock_path(), si.server_name())
    assert plugin[0] != standalone[0] and plugin[1] != standalone[1]
    assert "standalone" in standalone[1] and "standalone" not in plugin[1]


def test_about_names_the_mode():
    from kipi.gui import about
    assert ("Mode", "Standalone") in about.versions(solver=False)
    host.set_current(Host.PLUGIN)
    assert ("Mode", "KiCad plugin") in about.versions(solver=False)


# --- Board files -------------------------------------------------------------------------

def test_the_registry_knows_snapshots(board_file):
    from kipi.sources import registry
    from kipi.sources.snapshot_file import SnapshotFileSource
    assert registry.format_for("B.KIPI.JSON.GZ").name == "KiPI snapshot"
    assert registry.format_for("board.kicad_pcb") is None
    src = registry.open_board(board_file)
    assert isinstance(src, SnapshotFileSource) and src.caps.reload
    assert not src.caps.refill_zones and not src.caps.select_items
    snap = src.load()
    assert snap.board_name == make_board().board_name and snap.source == str(board_file)
    with pytest.raises(ValueError, match="cannot open"):
        registry.open_board("board.brd")
    assert registry.file_filter().startswith("Boards (*.json.gz *.json);;KiPI snapshot (")


# --- The window --------------------------------------------------------------------------

def test_standalone_opens_files_and_hides_kicad(app, no_3d, board_file):
    from kipi.gui.main_window import MainWindow
    w = MainWindow()
    assert w.host is Host.STANDALONE
    assert "(standalone)" in w.windowTitle()
    assert _toolbar(w)[:2] == ["Open…", "Save snapshot…"]          # nothing to reload yet
    assert "Open snapshot…" not in _toolbar(w)
    assert not w._act_refill.isVisible() and w.btn_select.isHidden()
    assert w.chk_models.isHidden()                                  # no KiCad for models

    w.open_path(str(board_file))
    _idle(app, w)
    assert w.snapshot is not None and w.snapshot.source == str(board_file)
    assert _toolbar(w)[:3] == ["Open…", "Reload", "Save snapshot…"]
    assert board_file.name in w.act_reload.toolTip()
    assert not w._act_refill.isVisible() and w.btn_select.isHidden()
    assert w.windowTitle().endswith(f"{w.snapshot.board_name} (standalone)")
    w.close()


def test_standalone_says_when_it_cannot_open_a_file(app, no_3d, monkeypatch):
    from kipi.gui.main_window import MainWindow
    said = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: said.append(a[2]))
    w = MainWindow()
    w.open_path("board.brd")
    assert said and "cannot open board.brd" in said[0]
    assert w._task is None and w._source is None
    w.close()


def test_the_plugin_keeps_its_toolbar(app, no_3d, plugin_host):
    from test_sources import FakeSource
    from kipi.gui.main_window import MainWindow
    w = MainWindow(source=FakeSource())
    _idle(app, w)
    bar = _toolbar(w)
    assert bar[:3] == ["Load from Fake", "Open snapshot…", "Save snapshot…"]
    assert "Open…" not in bar and "(standalone)" not in w.windowTitle()
    assert w._act_refill.isVisible()
    w.close()


def test_models_show_only_with_a_tool_that_has_them(app, no_3d):
    from kipi.gui.main_window import MainWindow
    from kipi.sources.kicad_ipc.source import InstalledKiCad
    w = MainWindow(tool=InstalledKiCad())
    assert not w.chk_models.isHidden()
    w.close()


# --- python -m kipi ----------------------------------------------------------------------

@pytest.fixture
def started(monkeypatch):
    """What ``python -m kipi`` handed the window, with no window and no lock."""
    from kipi.gui import app as kipi_app, single_instance

    class Guard:
        def release(self):
            pass

    got = {}
    monkeypatch.setattr(kipi_app, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(single_instance, "claim", lambda: Guard())
    monkeypatch.setattr(kipi_app, "run", lambda **kw: got.update(kw) or 0)
    return got


def test_from_a_terminal_kipi_is_standalone(started, board_file):
    from kipi.__main__ import main
    from kipi.sources.snapshot_file import SnapshotFileSource
    assert main([str(board_file)]) == 0
    assert host.current() is Host.STANDALONE
    assert isinstance(started["source"], SnapshotFileSource)
    assert started.get("tool") is None                  # no KiCad colors or models

    started.clear()
    assert main([]) == 0
    assert started["source"] is None                    # opens files from the window


def test_a_file_kipi_cannot_read_stops_it(started, tmp_path, capsys):
    from kipi.__main__ import main
    assert main([str(tmp_path / "gone.kipi.json.gz")]) == 2
    other = tmp_path / "board.brd"
    other.write_text("")
    assert main([str(other)]) == 2
    assert "No such file" in capsys.readouterr().err
    assert started == {}
