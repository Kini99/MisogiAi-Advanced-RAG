"""Setup script for the Advanced Assessment Generation System."""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "uploads",
        "chroma_db",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_redis():
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print("❌ Redis connection failed")
        print("Please install and start Redis server:")
        print("  Ubuntu/Debian: sudo apt-get install redis-server")
        print("  macOS: brew install redis")
        print("  Windows: Download from https://redis.io/download")
        print("  Or use Docker: docker run -d -p 6379:6379 redis:latest")
        return False


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your OpenAI API key")
        return True
    else:
        print("❌ env.example file not found")
        return False


def run_basic_tests():
    """Run basic tests to verify installation."""
    try:
        print("🧪 Running basic tests...")
        subprocess.check_call([sys.executable, "test_basic.py"])
        print("✅ Basic tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic tests failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🎓 Advanced Assessment Generation System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check Redis
    if not check_redis():
        print("\n⚠️  Redis is required for caching. Please install and start Redis before running the application.")
    
    # Create .env file
    create_env_file()
    
    # Run basic tests
    if not run_basic_tests():
        print("\n⚠️  Some tests failed. Please check the installation.")
    
    print("\n🎉 Setup completed!")
    print("\n📚 Next steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Start Redis server if not already running")
    print("3. Run the demo: python demo.py")
    print("4. Start the API server: python main.py")
    print("5. Visit http://localhost:8000/docs for API documentation")


if __name__ == "__main__":
    main() 