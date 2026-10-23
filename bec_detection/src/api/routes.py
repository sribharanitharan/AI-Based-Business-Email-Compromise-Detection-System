"""
routes.py — Modular APIRouter declarations for BEC endpoints.
"""
from fastapi import APIRouter
from .schemas import EmailAnalysisRequest

router = APIRouter(prefix="/api/v1")

@router.get("/ping")
def ping():
    return {"status": "ok", "service": "BEC Shield Core API"}
