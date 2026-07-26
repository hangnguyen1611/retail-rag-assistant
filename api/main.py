import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat
from api.dependencies import init_dependencies

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",") if o.strip()
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dependencies()
    yield


app = FastAPI(title="Retail RAG Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
