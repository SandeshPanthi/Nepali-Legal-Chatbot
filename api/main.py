from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import chat

app = FastAPI(title="Legal RAG API")

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include chat routes (this also handles the root "/" now)
app.include_router(chat.router)