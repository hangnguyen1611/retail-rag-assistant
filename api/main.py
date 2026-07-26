from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat
from api.dependencies import init_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load retriever + generator MỘT LẦN khi server khởi động
    init_dependencies()
    yield


app = FastAPI(title="Retail RAG Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: siết lại domain cụ thể khi deploy thật
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# Chạy: uvicorn api.main:app --reload
