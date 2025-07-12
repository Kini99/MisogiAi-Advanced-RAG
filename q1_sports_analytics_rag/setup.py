#!/usr/bin/env python3
"""
Setup script for Sports Analytics RAG System

This script helps users set up the system and configure environment variables.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file with user input."""
    print("\n🔧 Setting up environment variables...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists")
        overwrite = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Skipping .env file creation")
            return True
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        print("❌ OpenAI API key is required")
        return False
    
    # Optional configurations
    print("\nOptional configurations (press Enter to use defaults):")
    
    openai_model = input("OpenAI model (default: gpt-4-turbo-preview): ").strip() or "gpt-4-turbo-preview"
    chunk_size = input("Chunk size (default: 1000): ").strip() or "1000"
    api_port = input("API port (default: 8000): ").strip() or "8000"
    
    # Create .env file
    env_content = f"""# Sports Analytics RAG System - Environment Variables
OPENAI_API_KEY={api_key}
OPENAI_MODEL={openai_model}
CHUNK_SIZE={chunk_size}
API_PORT={api_port}

# Default settings (can be modified later)
CHROMA_PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=sports_analytics
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=10
TOP_K_RERANK=5
MAX_SUB_QUESTIONS=5
COMPRESSION_RATIO=0.7
API_HOST=0.0.0.0
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_setup():
    """Test the setup by running the test script."""
    print("\n🧪 Testing setup...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Setup test passed")
            return True
        else:
            print("❌ Setup test failed")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run test: {e}")
        return False

def main():
    """Main setup function."""
    print("🏈 Sports Analytics RAG System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Test setup
    if not test_setup():
        print("\n⚠️  Setup completed with warnings")
        print("You may need to check your configuration manually")
    else:
        print("\n🎉 Setup completed successfully!")
    
    print("\n📖 Next steps:")
    print("1. Run the demo: python main.py")
    print("2. Start the API server: python main.py --api")
    print("3. Check the README.md for more information")

if __name__ == "__main__":
    main() 