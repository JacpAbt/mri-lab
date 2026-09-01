"""Engine DB session plumbing (SQLite WAL on ext4 — enforced by settings)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from mrilab.settings import Settings

_engines: dict[str, object] = {}


def get_engine(settings: Settings):
    key = str(settings.db_path)
    if key not in _engines:
        settings.ensure_workspace()
        engine = create_engine(f"sqlite:///{settings.db_path}",
                               connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):  # WAL for reader/writer concurrency
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=10000")
            cur.close()

        SQLModel.metadata.create_all(engine)
        _engines[key] = engine
    return _engines[key]


@contextmanager
def session(settings: Settings) -> Iterator[Session]:
    with Session(get_engine(settings)) as s:
        yield s
        s.commit()
