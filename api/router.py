from fastapi import APIRouter
from api.routes import chat, typesense


api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(typesense.router)

