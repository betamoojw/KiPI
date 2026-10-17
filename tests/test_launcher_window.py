"""The launcher's setup window. Skipped where tkinter is missing (KiCad's own Python on
Windows has none) or there is no display (Linux CI)."""

import sys
import time
from pathlib import Path

import pytest

from kipi.launcher import deps, state, system
from kipi.launcher.install import Canceled, InstallError
from kipi.launcher.setup_env import Summary
from kipi.launcher.system import Python

if not system.can_show_windows():
    pytest.skip("no tkinter or no display", allow_module_level=True)

import tkinter as tk  # noqa: E402

from kipi.launcher.ui_tk import TkUI  # noqa: E402


@pytest.fixture
def ui():
    try:
        u = TkUI()
    except tk.TclError as e:
        pytest.skip(f"Tk cannot open a window here: {e}")
    yield u
    u.close()


def _summary(offered=True, default=False):
    plan = deps.Plan([deps.Package("numpy", "2.5.4", 13_000_000),
                      deps.Package("vtk", "9.7.1", 80_000_000)],
                     [deps.Package("mkl", "2026.2.0", 185_000_000)], "live")
    return Summary("KiPI's environment is not set up yet",
                   Python(sys.executable, (3, 13, 1), True), Path("/home/me/.local/share/kipi/venv"),
                   plan, offered, "" if offered else "Intel MKL is not available for Darwin",
                   default)


def _widgets(root, cls):
    found, todo = [], [root]
    while todo:
        w = todo.pop()
        todo += w.winfo_children()
        if w.winfo_class() == cls:
            found.append(w)
    return found


def _text(root):
    return " ".join(str(w.cget("text")) for w in _widgets(root, "TLabel")
                    + _widgets(root, "TCheckbutton") + _widgets(root, "TLabelframe"))


def _button(root, label):
    return next(b for b in _widgets(root, "TButton") if b.cget("text") == label)


def test_the_window_lists_every_package_and_installs(ui):
    def look_then_install():
        table = _widgets(ui.root, "Treeview")[0]
        rows = [table.item(i, "text") for i in table.get_children()]
        assert rows == ["vtk", "numpy"]                    # biggest first
        assert "Total: 2 packages: 93 MB to download" in _text(ui.root)
        _button(ui.root, "Install").invoke()
    ui.root.after(300, look_then_install)
    assert ui.confirm(_summary()) is False


def test_ticking_pardiso_adds_it(ui):
    def tick():
        _widgets(ui.root, "TCheckbutton")[0].invoke()
        assert "With PARDISO: 3 packages: 278 MB" in _text(ui.root)
        _button(ui.root, "Install").invoke()
    ui.root.after(300, tick)
    assert ui.confirm(_summary()) is True


def test_cancel_or_closing_the_window_installs_nothing(ui):
    ui.root.after(300, lambda: _button(ui.root, "Cancel").invoke())
    assert ui.confirm(_summary()) is None
    ui.root.after(300, ui._close_clicked)
    assert ui.confirm(_summary(default=True)) is None


def test_where_pardiso_is_not_offered_it_says_why(ui):
    def look():
        assert not _widgets(ui.root, "TCheckbutton")
        assert "Not available here: Intel MKL is not available for Darwin" in _text(ui.root)
        _button(ui.root, "Install").invoke()
    ui.root.after(300, look)
    assert ui.confirm(_summary(offered=False)) is False


def test_the_install_runs_beside_the_window(ui):
    def task(on_line, on_fraction, register):
        for i in range(5):
            on_line(f"Collecting package {i}")
            on_fraction(i / 5)
            time.sleep(0.02)
        return state.State(pardiso=True)
    assert ui.install(task).pardiso


def test_a_failed_install_raises_its_error(ui):
    def task(on_line, on_fraction, register):
        raise InstallError("pip failed")
    with pytest.raises(InstallError, match="pip failed"):
        ui.install(task)


def test_cancel_stops_the_install(ui):
    class Slow:
        canceled = False

        def cancel(self):
            Slow.canceled = True

    def task(on_line, on_fraction, register):
        register(Slow())
        t0 = time.monotonic()
        while not Slow.canceled and time.monotonic() - t0 < 10:
            time.sleep(0.02)
        raise Canceled()
    ui.root.after(400, lambda: _button(ui.root, "Cancel").invoke())
    with pytest.raises(Canceled):
        ui.install(task)
    assert Slow.canceled
