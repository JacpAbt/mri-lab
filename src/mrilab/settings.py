"""Service-side settings (pydantic-settings). Torch-side code never imports this;
the executor passes everything a GPU subprocess needs through argv + a whitelist env.

All paths are WSL paths — the service only runs inside WSL. The Windows tree is
the edit side and never executes anything.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRILAB_", env_file=".env", extra="ignore")

    # --- the two interpreters -------------------------------------------------
    # Environment of record for measurements (produced every golden row).
    # Checked against runtime_lock.json before any golden-comparable job.
    gpu_python: Path = Path("/home/jacopo/experiments/.venv/bin/python")
    # WSL twin of this repo; PYTHONPATH for GPU subprocesses is code_root/src.
    code_root: Path = Path("/home/jacopo/mri-lab")

    # --- state ----------------------------------------------------------------
    # Everything mutable lives here (rows, pools, reports, logs, DB, HF cache).
    # Must be ext4 (never /mnt/*): SQLite WAL locking is unreliable on 9p mounts.
    workspace: Path = Path("/home/jacopo/mri-lab-data")
    hf_cache: Path | None = None  # default: workspace/hfcache

    # --- GPU ops (values from the research repo's measured practice) ----------
    thermal_limit_c: int = 78
    thermal_retries: int = 3
    thermal_wait_s: int = 90
    power_cap_w: int = 300  # resets on reboot; checked at worker start, not enforced

    # --- API -------------------------------------------------------------------
    api_host: str = "0.0.0.0"  # WSL2: bind all so the Windows browser reaches it
    api_port: int = 8321
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("workspace")
    @classmethod
    def _workspace_not_on_windows_mount(cls, v: Path) -> Path:
        assert_native_path(v)
        return v

    # --- derived layout (single source for every component) --------------------
    @property
    def runtime_cwd(self) -> Path:
        """cwd for instrument subprocesses. Contains experiments/regimes/ so the
        vendored instruments' relative pool-cache paths keep working unmodified."""
        return self.workspace / "runtime"

    @property
    def pools_dir(self) -> Path:
        return self.runtime_cwd / "experiments" / "regimes"

    @property
    def rows_dir(self) -> Path:
        return self.workspace / "rows"

    @property
    def reports_dir(self) -> Path:
        return self.workspace / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def db_path(self) -> Path:
        return self.workspace / "mrilab.db"

    @property
    def queue_dir(self) -> Path:
        return self.workspace / "queues"

    @property
    def hf_cache_dir(self) -> Path:
        return self.hf_cache if self.hf_cache is not None else self.workspace / "hfcache"

    def ensure_workspace(self) -> None:
        for p in (
            self.runtime_cwd,
            self.pools_dir,
            self.rows_dir,
            self.reports_dir,
            self.logs_dir,
            self.queue_dir,
            self.hf_cache_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


def assert_native_path(p: Path) -> None:
    """Refuse workspace/state paths on the Windows mount (SQLite WAL + mmap are
    unreliable and slow on /mnt/* 9p mounts; state never crosses back to Windows
    except via scripts/backup_workspace.sh)."""
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "/" and parts[1] == "mnt":
        raise ValueError(
            f"workspace path {p} is on a Windows mount (/mnt/*); "
            "state must live on ext4, e.g. /home/jacopo/mri-lab-data"
        )


def get_settings() -> Settings:
    return Settings()
