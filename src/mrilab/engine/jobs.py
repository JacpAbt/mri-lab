"""JobSpec: a fully-declared instrument run whose content hash is the
idempotency key. dtype is REQUIRED and "auto" is rejected (ModelInspector's
silent auto->bf16 default must never decide a measurement's precision).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, field_validator

from mrilab.instruments._registry import INSTRUMENTS
from mrilab.settings import Settings


class Subject(BaseModel):
    repo: str | None = None       # HF repo id
    revision: str | None = None   # None == HF main (meaningful, kept)
    path: str | None = None       # local checkpoint dir (alternative to repo)

    @property
    def label(self) -> str:
        if self.path:
            return Path(self.path).name
        assert self.repo is not None
        return f"{self.repo.split('/')[-1]}@{self.revision or 'main'}"

    @field_validator("repo")
    @classmethod
    def _repo_or_path(cls, v, info):
        return v


class JobSpec(BaseModel):
    instrument: str
    subject: Subject
    dtype: str  # fp32 | bf16 | mixed — must match the instrument registry
    params: dict = {}

    @field_validator("instrument")
    @classmethod
    def _known_instrument(cls, v: str) -> str:
        if v not in INSTRUMENTS:
            raise ValueError(f"unknown instrument '{v}' (see instruments/_registry.py)")
        return v

    @field_validator("dtype")
    @classmethod
    def _no_auto(cls, v: str) -> str:
        if v == "auto":
            raise ValueError('dtype "auto" is rejected: precision must be chosen, '
                             "not defaulted (ModelInspector auto->bf16)")
        if v not in ("fp32", "bf16", "mixed"):
            raise ValueError(f"dtype must be fp32|bf16|mixed, got {v!r}")
        return v

    @property
    def lane(self) -> str:
        return INSTRUMENTS[self.instrument]["lane"]

    def content_hash(self) -> str:
        canon = json.dumps(
            {"instrument": self.instrument, "subject": self.subject.model_dump(),
             "dtype": self.dtype, "params": self.params},
            sort_keys=True, separators=(",", ":"),
        )
        return "job_" + hashlib.sha256(canon.encode()).hexdigest()[:20]

    # ---- CLI materialization -------------------------------------------------

    def output_dir(self, settings: Settings) -> Path:
        return settings.rows_dir / self.instrument / self.subject.label

    def cli_args(self, settings: Settings) -> list[str]:
        """Instrument argv. params keys map to --kebab-case flags; True -> bare
        flag; lists -> repeated values (nargs)."""
        args: list[str] = []
        if self.subject.path:
            args += ["--path", self.subject.path]
        else:
            assert self.subject.repo
            args += ["--repo", self.subject.repo]
            if self.subject.revision:
                # icl_readiness/induction_ladder take --revisions (plural)
                if self.instrument in ("icl_readiness", "induction_ladder", "icl_ladder"):
                    args += ["--revisions", self.subject.revision]
                else:
                    args += ["--revision", self.subject.revision]
        for key, val in sorted(self.params.items()):
            flag = "--" + key.replace("_", "-")
            if isinstance(val, bool):
                if val:
                    args.append(flag)
            elif isinstance(val, (list, tuple)):
                args.append(flag)
                args += [str(v) for v in val]
            else:
                args += [flag, str(val)]
        needs_output = self.instrument not in ("z_ter",)  # z_ter: fixed OUT dir
        if needs_output and "--output" not in args:
            args += ["--output", str(self.output_dir(settings))]
        return args
