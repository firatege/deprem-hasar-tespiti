from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.data import router as data_router
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.config.settings import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.tiles_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.quarantine_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Earthquake Damage Backend", version="0.1.0", lifespan=lifespan)

app.include_router(health_router)
app.include_router(jobs_router)
app.include_router(data_router)