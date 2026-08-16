from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import wetlands

app = FastAPI(title="NeerRikshan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(wetlands.router)


@app.get("/health")
def health():
    return {"status": "ok"}
