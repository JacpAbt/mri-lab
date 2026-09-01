from __future__ import annotations

from pathlib import Path

import pytest

from mrilab.settings import Settings, assert_native_path


def test_workspace_on_windows_mount_is_refused() -> None:
    with pytest.raises(ValueError, match="/mnt"):
        assert_native_path(Path("/mnt/c/Users/whoever/mri-lab-data"))


def test_native_workspace_accepted() -> None:
    assert_native_path(Path("/home/jacopo/mri-lab-data"))  # must not raise


def test_settings_refuse_mounted_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRILAB_WORKSPACE", "/mnt/c/tmp/ws")
    with pytest.raises(Exception, match="/mnt"):
        Settings()


def test_derived_layout_hangs_off_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRILAB_WORKSPACE", "/home/jacopo/somewhere")
    s = Settings()
    assert s.rows_dir == Path("/home/jacopo/somewhere/rows")
    assert s.db_path == Path("/home/jacopo/somewhere/mrilab.db")
    # the pool subtree must mirror the research repo's relative layout exactly,
    # so vendored instruments' hardcoded "experiments/regimes/..." paths resolve
    assert s.pools_dir == Path("/home/jacopo/somewhere/runtime/experiments/regimes")
    # hf_cache defaults to None -> subprocesses inherit ~/.cache/huggingface
    assert s.hf_cache is None


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRILAB_API_PORT", "9999")
    assert Settings().api_port == 9999
