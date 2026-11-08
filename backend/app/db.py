from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)


def init_engine(db_path: Path) -> Engine:
    """(Re)create the engine for the given SQLite file and create tables."""
    global _engine
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):  # pragma: no cover - trivial
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    from . import models  # noqa: F401  (register tables)

    Base.metadata.create_all(engine)
    _upgrade_schema(engine)
    SessionLocal.configure(bind=engine)
    _engine = engine
    return engine


def _upgrade_schema(engine: Engine) -> None:
    """create_all() never alters existing tables: add columns newer versions need."""
    with engine.begin() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(annotations)")}
        if "polygon" not in columns:
            conn.exec_driver_sql("ALTER TABLE annotations ADD COLUMN polygon JSON")


def get_engine() -> Engine:
    assert _engine is not None, "init_engine() has not been called"
    return _engine


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
