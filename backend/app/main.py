import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.database.session import engine, Base, SessionLocal
from backend.app.api.routes import router
from backend.app.services.pipeline_service import PipelineService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voiceminer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    logger.info("Initializing VoiceMiner SQLite database tables...")
    Base.metadata.create_all(bind=engine)

    # Preload sample dataset so the UI has immediate live research data ready
    sample_file = "data/samples/airline_sample_1000.csv"
    if os.path.exists(sample_file):
        logger.info("Preloading benchmark dataset: %s", sample_file)
        try:
            service = PipelineService.get_instance()
            db = SessionLocal()
            df, stats = service.ingest_file(sample_file)
            service.run_full_pipeline(df, n_clusters=6, db=db)
            db.close()
            logger.info("Pipeline preloaded with %d records.", len(df))
        except Exception as e:
            logger.warning("Could not auto-preload sample dataset: %s", e)

    yield
    logger.info("Shutting down VoiceMiner backend.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Customer Voice Intelligence and Evidence-Grounded Emerging Root-Cause Discovery",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
