#!/usr/bin/env python3
"""
Database Initialization Script for Strategic Decision Engine
This script helps set up the PostgreSQL database and initialize tables.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Make sure you have installed all requirements: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Database initialization helper class"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.db_name = None
        self.parse_database_url()
    
    def parse_database_url(self):
        """Parse database URL to extract database name and connection details"""
        try:
            # Extract database name from URL
            # Format: postgresql://username:password@host:port/database_name
            if '/' in self.database_url:
                self.db_name = self.database_url.split('/')[-1]
            else:
                self.db_name = 'strategic_db'
            logger.info(f"Database name: {self.db_name}")
        except Exception as e:
            logger.error(f"Error parsing database URL: {e}")
            raise
    
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect to PostgreSQL server (not to specific database)
            base_url = self.database_url.rsplit('/', 1)[0] + '/postgres'
            
            logger.info("Connecting to PostgreSQL server...")
            engine = create_engine(base_url)
            
            with engine.connect() as conn:
                # Set autocommit mode
                conn.execute(text("COMMIT"))
                
                # Check if database exists
                result = conn.execute(text(
                    "SELECT 1 FROM pg_database WHERE datname = :db_name"
                ), {"db_name": self.db_name})
                
                if not result.fetchone():
                    logger.info(f"Creating database: {self.db_name}")
                    conn.execute(text(f"CREATE DATABASE {self.db_name}"))
                    logger.info(f"Database {self.db_name} created successfully")
                else:
                    logger.info(f"Database {self.db_name} already exists")
            
            engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False
    
    def test_connection(self):
        """Test database connection"""
        try:
            logger.info("Testing database connection...")
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                logger.info(f"✓ Database connection successful!")
                logger.info(f"PostgreSQL Version: {version}")
            
            engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")
            return False
    
    def initialize_tables(self):
        """Initialize database tables"""
        try:
            logger.info("Initializing database tables...")
            
            # Import database models
            from backend.core.database import Base, engine
            
            # Create all tables
            Base.metadata.create_all(bind=engine)
            logger.info("✓ Database tables created successfully")
            
            # List created tables
            with engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ))
                tables = [row[0] for row in result.fetchall()]
                logger.info(f"Created tables: {', '.join(tables)}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error initializing tables: {e}")
            return False
    
    def create_extensions(self):
        """Create useful PostgreSQL extensions"""
        try:
            logger.info("Creating PostgreSQL extensions...")
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Create extensions
                extensions = [
                    "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"",
                    "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\"",
                ]
                
                for ext in extensions:
                    conn.execute(text(ext))
                    conn.commit()
                
                logger.info("✓ PostgreSQL extensions created")
            
            engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"✗ Error creating extensions: {e}")
            return False


def load_environment():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    
    if not env_path.exists():
        logger.warning("No .env file found. Please create one from env_template.txt")
        logger.info("You can copy env_template.txt to .env and fill in your values")
        return None
    
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info("✓ Environment variables loaded from .env file")
        return True
    except ImportError:
        logger.warning("python-dotenv not installed. Using system environment variables.")
        return True
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        return False


def get_database_url():
    """Get database URL from environment variables"""
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        logger.error("DATABASE_URL environment variable not set!")
        logger.info("Please set DATABASE_URL in your .env file or environment variables")
        logger.info("Example: DATABASE_URL=postgresql://username:password@localhost:5432/strategic_db")
        return None
    
    return database_url


def main():
    """Main initialization function"""
    print("=" * 60)
    print("Strategic Decision Engine - Database Initialization")
    print("=" * 60)
    
    # Load environment
    if load_environment() is None:
        return False
    
    # Get database URL
    database_url = get_database_url()
    if not database_url:
        return False
    
    # Initialize database
    initializer = DatabaseInitializer(database_url)
    
    # Step 1: Create database if it doesn't exist
    if not initializer.create_database_if_not_exists():
        logger.error("Failed to create database")
        return False
    
    # Step 2: Test connection
    if not initializer.test_connection():
        logger.error("Database connection test failed")
        return False
    
    # Step 3: Create extensions
    if not initializer.create_extensions():
        logger.warning("Failed to create extensions (this might be OK)")
    
    # Step 4: Initialize tables
    if not initializer.initialize_tables():
        logger.error("Failed to initialize tables")
        return False
    
    print("\n" + "=" * 60)
    print("✓ Database initialization completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Make sure your .env file has the correct API keys")
    print("2. Start Redis server if you haven't already")
    print("3. Run the application: python start.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 