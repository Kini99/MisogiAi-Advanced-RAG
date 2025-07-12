#!/bin/bash

# Strategic Decision Engine Deployment Script
# This script handles deployment of the complete application stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_FILE="deploy.log"

# Functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$LOG_FILE"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    success "Dependencies check passed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    if [ ! -f "$ENV_FILE" ]; then
        log "Creating .env file from template..."
        cp env_template.txt "$ENV_FILE"
        
        # Generate random passwords
        POSTGRES_PASSWORD=$(openssl rand -base64 32)
        GRAFANA_PASSWORD=$(openssl rand -base64 16)
        
        # Update .env file
        sed -i "s/your_postgres_password_here/$POSTGRES_PASSWORD/g" "$ENV_FILE"
        sed -i "s/your_grafana_password_here/$GRAFANA_PASSWORD/g" "$ENV_FILE"
        
        warning "Please update the .env file with your API keys before continuing"
        warning "Required API keys: OpenAI, Anthropic, Google, Alpha Vantage, FRED, Quandl"
        
        read -p "Press Enter after updating the .env file..."
    fi
    
    # Validate required environment variables
    source "$ENV_FILE"
    
    required_vars=(
        "OPENAI_API_KEY"
        "ANTHROPIC_API_KEY"
        "GOOGLE_API_KEY"
        "POSTGRES_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    success "Environment setup complete"
}

# Create directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "logs"
        "uploads"
        "temp"
        "chroma_db"
        "nginx/ssl"
        "monitoring"
        "scripts"
        "$BACKUP_DIR"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done
    
    success "Directories created"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    if [ ! -f "nginx/ssl/cert.pem" ] || [ ! -f "nginx/ssl/key.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        success "SSL certificates generated"
    else
        log "SSL certificates already exist"
    fi
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'strategic-decision-engine'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['db:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
EOF
    
    # Grafana datasource configuration
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    success "Monitoring configuration setup complete"
}

# Database initialization
setup_database() {
    log "Setting up database initialization..."
    
    cat > scripts/init_db.sql << EOF
-- Strategic Decision Engine Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE strategic_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'strategic_db');

-- Connect to strategic_db
\c strategic_db;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS documents;
CREATE SCHEMA IF NOT EXISTS chat;
CREATE SCHEMA IF NOT EXISTS analysis;
CREATE SCHEMA IF NOT EXISTS evaluation;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO public, documents, chat, analysis, evaluation, monitoring;

-- Create initial admin user (optional)
-- INSERT INTO users (id, username, email, is_admin) VALUES 
-- (uuid_generate_v4(), 'admin', 'admin@example.com', true);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO postgres;
EOF
    
    success "Database initialization script created"
}

# Build and start services
deploy_services() {
    log "Building and starting services..."
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build application image
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    success "Services deployed"
}

# Wait for services to be ready
wait_for_services() {
    log "Waiting for services to be ready..."
    
    services=(
        "http://localhost:8000/health:Backend API"
        "http://localhost:8501:Frontend"
        "http://localhost:5432:Database"
        "http://localhost:6379:Redis"
        "http://localhost:8002/api/v1/heartbeat:ChromaDB"
    )
    
    for service in "${services[@]}"; do
        url=$(echo "$service" | cut -d':' -f1,2)
        name=$(echo "$service" | cut -d':' -f3)
        
        log "Waiting for $name to be ready..."
        
        timeout=300  # 5 minutes
        counter=0
        
        until curl -f -s "$url" > /dev/null 2>&1; do
            if [ $counter -ge $timeout ]; then
                error "$name failed to start within timeout"
                exit 1
            fi
            
            sleep 5
            counter=$((counter + 5))
        done
        
        success "$name is ready"
    done
}

# Run health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check application health
    if curl -f -s "http://localhost:8000/health" > /dev/null; then
        success "Backend API health check passed"
    else
        error "Backend API health check failed"
        exit 1
    fi
    
    # Check database connection
    if docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; then
        success "Database health check passed"
    else
        error "Database health check failed"
        exit 1
    fi
    
    # Check Redis connection
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        success "Redis health check passed"
    else
        error "Redis health check failed"
        exit 1
    fi
    
    success "All health checks passed"
}

# Show deployment summary
show_deployment_summary() {
    log "Deployment Summary"
    echo "=================="
    echo
    echo "🚀 Strategic Decision Engine has been successfully deployed!"
    echo
    echo "Services:"
    echo "  📊 Frontend (Streamlit):     http://localhost:8501"
    echo "  🔧 Backend API:              http://localhost:8000"
    echo "  📖 API Documentation:        http://localhost:8000/docs"
    echo "  🗄️  Database (PostgreSQL):   localhost:5432"
    echo "  🔄 Cache (Redis):            localhost:6379"
    echo "  🔍 Vector DB (ChromaDB):     http://localhost:8002"
    echo
    echo "Monitoring:"
    echo "  📈 Prometheus:               http://localhost:9090"
    echo "  📊 Grafana:                  http://localhost:3000"
    echo "  🔍 Elasticsearch:            http://localhost:9200"
    echo "  📋 Kibana:                   http://localhost:5601"
    echo
    echo "Management Commands:"
    echo "  🛑 Stop services:            docker-compose down"
    echo "  🔄 Restart services:         docker-compose restart"
    echo "  📊 View logs:                docker-compose logs -f"
    echo "  📈 Monitor resources:        docker stats"
    echo
    echo "Configuration:"
    echo "  ⚙️  Environment:             $ENV_FILE"
    echo "  🔧 Docker Compose:          $COMPOSE_FILE"
    echo "  📝 Deployment Log:          $LOG_FILE"
    echo
    echo "For support, check the README.md file or deployment logs."
}

# Main deployment function
main() {
    log "Starting Strategic Decision Engine deployment..."
    
    # Pre-deployment checks
    check_dependencies
    setup_environment
    create_directories
    generate_ssl_certificates
    setup_monitoring
    setup_database
    
    # Deploy services
    deploy_services
    wait_for_services
    run_health_checks
    
    # Show summary
    show_deployment_summary
    
    success "Deployment completed successfully!"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "update")
        log "Updating deployment..."
        docker-compose -f "$COMPOSE_FILE" pull
        docker-compose -f "$COMPOSE_FILE" build --no-cache
        docker-compose -f "$COMPOSE_FILE" up -d
        success "Update completed"
        ;;
    "stop")
        log "Stopping services..."
        docker-compose -f "$COMPOSE_FILE" down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting services..."
        docker-compose -f "$COMPOSE_FILE" restart
        success "Services restarted"
        ;;
    "logs")
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    "status")
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    "cleanup")
        log "Cleaning up Docker resources..."
        docker-compose -f "$COMPOSE_FILE" down -v
        docker system prune -f
        success "Cleanup completed"
        ;;
    "backup")
        log "Creating backup..."
        mkdir -p "$BACKUP_DIR"
        docker-compose exec -T db pg_dump -U postgres strategic_db > "$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql"
        tar -czf "$BACKUP_DIR/chroma_backup_$(date +%Y%m%d_%H%M%S).tar.gz" chroma_db/
        success "Backup created in $BACKUP_DIR"
        ;;
    "help")
        echo "Strategic Decision Engine Deployment Script"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  deploy    - Deploy the complete application stack (default)"
        echo "  update    - Update and restart services"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  logs      - Show service logs"
        echo "  status    - Show service status"
        echo "  cleanup   - Clean up Docker resources"
        echo "  backup    - Create database and vector store backup"
        echo "  help      - Show this help message"
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac 