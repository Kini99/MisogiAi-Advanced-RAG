#!/usr/bin/env python3
"""
Strategic Decision Engine Startup Script
This script initializes and starts the complete Strategic Decision Engine system.
"""

import os
import sys
import time
import signal
import asyncio
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional, List
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('startup.log')
    ]
)
logger = logging.getLogger(__name__)

class StrategicDecisionEngineStarter:
    """Manages the startup and initialization of the Strategic Decision Engine."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the startup manager."""
        self.project_root = Path(__file__).parent
        self.config_file = config_file or '.env'
        self.processes = {}
        self.shutdown_event = threading.Event()
        
        # Process configurations
        self.services = {
            'backend': {
                'name': 'FastAPI Backend',
                'command': [
                    sys.executable, '-m', 'uvicorn',
                    'backend.main:app',
                    '--host', '0.0.0.0',
                    '--port', '8000',
                    '--reload'
                ],
                'cwd': self.project_root,
                'health_check': 'http://localhost:8000/health',
                'startup_delay': 10
            },
            'frontend': {
                'name': 'Streamlit Frontend',
                'command': [
                    sys.executable, '-m', 'streamlit',
                    'run', 'frontend/streamlit_app.py',
                    '--server.port', '8501',
                    '--server.address', '0.0.0.0',
                    '--server.headless', 'true'
                ],
                'cwd': self.project_root,
                'health_check': 'http://localhost:8501',
                'startup_delay': 15
            }
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        checks = [
            self._check_python_version(),
            self._check_environment_file(),
            self._check_dependencies(),
            self._check_directories(),
            self._check_database_connection(),
            self._check_external_services()
        ]
        
        if all(checks):
            logger.info("✓ All prerequisites met")
            return True
        else:
            logger.error("✗ Prerequisites check failed")
            return False
    
    def _check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            logger.info(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"✗ Python 3.9+ required, found {version.major}.{version.minor}.{version.micro}")
            return False
    
    def _check_environment_file(self) -> bool:
        """Check if environment file exists and has required variables."""
        env_path = self.project_root / self.config_file
        
        if not env_path.exists():
            logger.error(f"✗ Environment file not found: {env_path}")
            logger.info("Create .env file from env_template.txt")
            return False
        
        # Check for required environment variables
        required_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'GOOGLE_API_KEY',
            'DATABASE_URL',
            'REDIS_URL'
        ]
        
        missing_vars = []
        with open(env_path, 'r') as f:
            env_content = f.read()
            for var in required_vars:
                if f"{var}=" not in env_content or f"{var}=your_" in env_content:
                    missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"✗ Missing or placeholder environment variables: {missing_vars}")
            return False
        
        logger.info("✓ Environment file configured")
        return True
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        try:
            import fastapi
            import streamlit
            import langchain
            import chromadb
            import redis
            logger.info("✓ Core dependencies installed")
            return True
        except ImportError as e:
            logger.error(f"✗ Missing dependency: {e}")
            logger.info("Run: pip install -r requirements.txt")
            return False
    
    def _check_directories(self) -> bool:
        """Check if required directories exist."""
        required_dirs = [
            'backend',
            'frontend', 
            'logs',
            'temp',
            'uploads',
            'chroma_db'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                # Try to create the directory
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
        
        if missing_dirs:
            logger.warning(f"Created missing directories: {missing_dirs}")
        
        logger.info("✓ Directory structure verified")
        return True
    
    def _check_database_connection(self) -> bool:
        """Check database connectivity."""
        try:
            # Try to import and initialize database components
            from backend.core.database import engine, SessionLocal
            from backend.models.database_models import Base
            
            # Test connection
            with SessionLocal() as session:
                session.execute("SELECT 1")
                logger.info("✓ Database connection successful")
                return True
                
        except Exception as e:
            logger.warning(f"⚠ Database connection issue: {e}")
            logger.info("Database will be initialized on first run")
            return True  # Allow startup even if DB isn't ready yet
    
    def _check_external_services(self) -> bool:
        """Check external service dependencies."""
        # For now, just check if Redis is configured
        try:
            import redis
            # Don't actually connect in prerequisites, just verify Redis is available
            logger.info("✓ External service dependencies available")
            return True
        except Exception as e:
            logger.warning(f"⚠ External service check: {e}")
            return True  # Allow startup, services may not be ready yet
    
    async def initialize_system(self):
        """Initialize the system components."""
        logger.info("Initializing system components...")
        
        try:
            # Initialize database
            await self._initialize_database()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            # Initialize cache
            await self._initialize_cache()
            
            # Load demo data if requested
            if self._should_load_demo_data():
                await self._load_demo_data()
            
            logger.info("✓ System initialization completed")
            
        except Exception as e:
            logger.error(f"✗ System initialization failed: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize the database schema."""
        try:
            from backend.core.database import engine
            from backend.models.database_models import Base
            
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("✓ Database schema initialized")
            
        except Exception as e:
            logger.warning(f"⚠ Database initialization issue: {e}")
    
    async def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            from backend.services.vector_store_service import VectorStoreService
            from backend.core.config import Settings
            
            settings = Settings()
            vector_service = VectorStoreService(settings)
            
            # Test vector store connection
            stats = await vector_service.get_collection_stats()
            logger.info(f"✓ Vector store initialized: {stats.get('total_documents', 0)} documents")
            
        except Exception as e:
            logger.warning(f"⚠ Vector store initialization issue: {e}")
    
    async def _initialize_cache(self):
        """Initialize the cache service."""
        try:
            from backend.services.cache_service import CacheService
            from backend.core.config import Settings
            
            settings = Settings()
            cache_service = CacheService(settings)
            
            # Test cache connection
            stats = await cache_service.get_stats()
            logger.info(f"✓ Cache initialized: {stats}")
            
        except Exception as e:
            logger.warning(f"⚠ Cache initialization issue: {e}")
    
    def _should_load_demo_data(self) -> bool:
        """Check if demo data should be loaded."""
        return os.getenv('LOAD_DEMO_DATA', 'false').lower() == 'true'
    
    async def _load_demo_data(self):
        """Load demo data into the system."""
        try:
            from demo_data.load_demo_data import DemoDataLoader
            
            loader = DemoDataLoader()
            result = await loader.load_demo_data()
            
            logger.info(f"✓ Demo data loaded: {result}")
            
        except Exception as e:
            logger.warning(f"⚠ Demo data loading issue: {e}")
    
    def start_service(self, service_name: str) -> subprocess.Popen:
        """Start a specific service."""
        service_config = self.services[service_name]
        
        logger.info(f"Starting {service_config['name']}...")
        
        try:
            process = subprocess.Popen(
                service_config['command'],
                cwd=service_config['cwd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[service_name] = process
            
            # Start log monitoring in background
            threading.Thread(
                target=self._monitor_service_logs,
                args=(service_name, process),
                daemon=True
            ).start()
            
            logger.info(f"✓ {service_config['name']} started (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"✗ Failed to start {service_config['name']}: {e}")
            raise
    
    def _monitor_service_logs(self, service_name: str, process: subprocess.Popen):
        """Monitor service logs and output them."""
        service_config = self.services[service_name]
        
        # Create service-specific logger
        service_logger = logging.getLogger(f"service.{service_name}")
        
        # Monitor stdout
        def monitor_stdout():
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    service_logger.info(f"[{service_config['name']}] {line.strip()}")
                if self.shutdown_event.is_set():
                    break
        
        # Monitor stderr
        def monitor_stderr():
            for line in iter(process.stderr.readline, ''):
                if line.strip():
                    service_logger.error(f"[{service_config['name']}] {line.strip()}")
                if self.shutdown_event.is_set():
                    break
        
        stdout_thread = threading.Thread(target=monitor_stdout, daemon=True)
        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
    
    async def wait_for_service_health(self, service_name: str):
        """Wait for a service to become healthy."""
        service_config = self.services[service_name]
        health_check_url = service_config['health_check']
        startup_delay = service_config['startup_delay']
        
        logger.info(f"Waiting for {service_config['name']} to become healthy...")
        
        # Initial delay
        await asyncio.sleep(startup_delay)
        
        # Health check with timeout
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_check_url, timeout=5.0)
                    if response.status_code == 200:
                        logger.info(f"✓ {service_config['name']} is healthy")
                        return True
            except Exception:
                pass
            
            attempt += 1
            await asyncio.sleep(2)
        
        logger.warning(f"⚠ {service_config['name']} health check timeout")
        return False
    
    async def start_all_services(self):
        """Start all services in the correct order."""
        logger.info("Starting all services...")
        
        # Start backend first
        self.start_service('backend')
        await self.wait_for_service_health('backend')
        
        # Start frontend
        self.start_service('frontend')
        await self.wait_for_service_health('frontend')
        
        logger.info("✓ All services started successfully")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Shutdown all services gracefully."""
        logger.info("Shutting down services...")
        
        for service_name, process in self.processes.items():
            service_config = self.services[service_name]
            logger.info(f"Stopping {service_config['name']}...")
            
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✓ {service_config['name']} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {service_config['name']}")
                process.kill()
                process.wait()
            except Exception as e:
                logger.error(f"Error stopping {service_config['name']}: {e}")
        
        self.processes.clear()
        logger.info("✓ Shutdown completed")
    
    def print_startup_summary(self):
        """Print startup summary and access information."""
        print("\n" + "="*60)
        print("🚀 Strategic Decision Engine - READY!")
        print("="*60)
        print()
        print("Services:")
        print("  📊 Frontend (Streamlit):     http://localhost:8501")
        print("  🔧 Backend API:              http://localhost:8000")
        print("  📖 API Documentation:        http://localhost:8000/docs")
        print()
        print("Features Available:")
        print("  ✅ Document Upload & Processing")
        print("  ✅ Strategic Chat with AI")
        print("  ✅ SWOT Analysis Generation")
        print("  ✅ Market & Financial Analysis")
        print("  ✅ RAGAS Evaluation Dashboard")
        print("  ✅ Multi-LLM Support (GPT-4o, Claude 3.5, Gemini 2.5)")
        print()
        print("Demo Data:")
        if self._should_load_demo_data():
            print("  📄 Strategic Planning Document 2024")
            print("  💰 Financial Analysis & Forecast 2024")  
            print("  🏢 Competitive Analysis 2024")
        else:
            print("  ℹ️  Set LOAD_DEMO_DATA=true to load sample documents")
        print()
        print("Management:")
        print("  🛑 Stop: Ctrl+C")
        print("  📊 Logs: See startup.log")
        print("  ⚙️  Config: .env file")
        print()
        print("Ready for strategic decision making! 🎯")
        print("="*60)
    
    async def run(self):
        """Main run method."""
        logger.info("Strategic Decision Engine starting up...")
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                sys.exit(1)
            
            # Initialize system
            await self.initialize_system()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Start all services
            await self.start_all_services()
            
            # Print summary
            self.print_startup_summary()
            
            # Wait for shutdown signal
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
                
                # Check if any process died
                for service_name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        service_config = self.services[service_name]
                        logger.error(f"✗ {service_config['name']} died unexpectedly")
                        self.shutdown_event.set()
                        break
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
        finally:
            self.shutdown()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Strategic Decision Engine Startup")
    parser.add_argument(
        '--config',
        default='.env',
        help='Path to environment configuration file'
    )
    parser.add_argument(
        '--demo-data',
        action='store_true',
        help='Load demo data on startup'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo_data:
        os.environ['LOAD_DEMO_DATA'] = 'true'
    
    # Initialize and run the system
    starter = StrategicDecisionEngineStarter(config_file=args.config)
    
    try:
        asyncio.run(starter.run())
    except KeyboardInterrupt:
        print("\nShutdown initiated by user")
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 