"""
middleware.py — FastAPI Security, Rate Limiting, and CORS Middleware.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class RequestTimerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        duration = round((time.time() - start) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(duration)
        return response
