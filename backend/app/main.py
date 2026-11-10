"""VisionLoop API server.

Run (development):  uvicorn app.main:app --reload   (from the backend folder)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import __version__
from .config import settings
from .db import init_engine
from .plugins import registry
from .routers import annotations, dataset, images, jobs, projects, sam, updates
from .services.jobs import runner
from .services.sam_interactive import session as sam_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    init_engine(settings.db_path)
    registry.load_plugins(settings.plugin_dir)
    runner.start()
    yield
    sam_session.stop()


app = FastAPI(title="VisionLoop", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (projects.router, images.router, annotations.router, dataset.router, jobs.router, sam.router,
          updates.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


# ------------------------------------------------- built frontend (production)
@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(404)
    dist: Path = settings.frontend_dist
    index = dist / "index.html"
    if not index.exists():
        return {
            "message": "前端尚未建置。開發模式請另外執行 `npm run dev`（frontend 資料夾），"
            "或執行 `npm run build` 後重新整理此頁。",
            "api_docs": "/docs",
        }
    target = (dist / full_path).resolve()
    if full_path and target.is_file() and dist.resolve() in target.parents:
        return FileResponse(target)
    return FileResponse(index)
