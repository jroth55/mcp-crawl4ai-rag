"""
Authentication wrapper for FastMCP using class extension.
This extends FastMCP to add authentication middleware properly.
"""
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate API requests using X-API-Key header."""
    
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key
    
    async def dispatch(self, request: Request, call_next):
        # Only check authentication for SSE endpoint
        if request.url.path == "/sse" and self.api_key:
            # Extract and validate API key
            client_api_key = request.headers.get("X-API-Key", "")
            client_host = request.client.host if request.client else "unknown"
            
            # Check for missing API key
            if not client_api_key:
                logger.info(f"Missing API Key. Access denied to {request.url.path} from {client_host}")
                return Response(
                    content="event: error\ndata: {\"error\": \"Missing API Key\"}\n\n",
                    status_code=401,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "X-API-Key, Content-Type"
                    }
                )
            
            # Check for invalid API key
            if client_api_key != self.api_key:
                logger.info(f"Invalid API Key. Access denied to {request.url.path} from {client_host}")
                return Response(
                    content="event: error\ndata: {\"error\": \"Invalid API Key\"}\n\n",
                    status_code=401,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "X-API-Key, Content-Type"
                    }
                )
        
        # API key valid or not required, proceed with request
        return await call_next(request)

class AuthenticatedFastMCP(FastMCP):
    """FastMCP server with authentication middleware."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = os.getenv("MCP_SERVER_API_KEY")
        
        if self.api_key:
            logger.info("MCP_SERVER_API_KEY is set. API Key authentication enabled.")
        else:
            logger.info("MCP_SERVER_API_KEY not set. Running without API Key authentication.")
    
    def sse_app(self) -> Starlette:
        """Override to inject authentication middleware and add health check endpoint."""
        app = super().sse_app()
        
        # Add health check endpoint
        async def health_check(request: Request):
            """Health check endpoint for monitoring."""
            return JSONResponse({
                "status": "healthy",
                "service": "mcp-crawl4ai-rag",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transport": os.getenv("TRANSPORT", "sse"),
                "authentication": "enabled" if self.api_key else "disabled"
            })
        
        # Add the health route
        app.routes.insert(0, Route("/health", health_check, methods=["GET"]))
        
        # Add authentication middleware
        if self.api_key:
            app.add_middleware(AuthMiddleware, api_key=self.api_key)
        return app