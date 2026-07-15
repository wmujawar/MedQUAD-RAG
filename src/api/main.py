from fastapi import FastAPI

from src.api.v1.router import api_router as v1_router
from src.utils.logger import setup_logging

app = FastAPI()


setup_logging()

app.include_router(v1_router, prefix="/api/v1")
