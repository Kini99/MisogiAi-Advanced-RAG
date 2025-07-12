"""Main entry point for the Advanced Assessment Generation System."""

import uvicorn
from src.api import app
from src.config import settings


def main():
    """Main function to run the application."""
    print("🚀 Starting Advanced Assessment Generation System...")
    print(f"📊 API will be available at: http://{settings.api_host}:{settings.api_port}")
    print(f"📚 Documentation at: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔍 Health check at: http://{settings.api_host}:{settings.api_port}/health")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main() 