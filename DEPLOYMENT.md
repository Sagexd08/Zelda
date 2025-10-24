# Deployment Guide

This guide covers various deployment options for the Facial Authentication System.

## Table of Contents

1. [Development Deployment](#development-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Maintenance](#monitoring--maintenance)

## Development Deployment

### Quick Start

Use the provided startup scripts for instant development setup:

**Linux/macOS:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```bash
start.bat
```

This automatically:
- Creates virtual environment
- Installs dependencies
- Starts backend on port 8000
- Starts frontend on port 3000

### Manual Development Setup

#### Backend

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Docker Deployment

### Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Full Stack Docker Deployment

```bash
# Build and start all services including frontend, backend, database, and monitoring
docker-compose -f docker-compose.fullstack.yml up -d

# Scale backend for high load
docker-compose -f docker-compose.fullstack.yml up -d --scale backend=3
```

This includes:
- **Backend API** - FastAPI application
- **Frontend** - React app served by Nginx
- **PostgreSQL** - Production database
- **Redis** - Caching layer
- **Prometheus** - Metrics collection
- **Grafana** - Monitoring dashboards

Access points:
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

### Custom Docker Build

#### Backend Only
```bash
docker build -f deployment/Dockerfile -t facial-auth-backend .
docker run -p 8000:8000 \
  -v $(pwd)/weights:/app/weights:ro \
  -v $(pwd)/data:/app/data \
  facial-auth-backend
```

#### Frontend Only
```bash
cd frontend
docker build -t facial-auth-frontend .
docker run -p 80:80 facial-auth-frontend
```

## Production Deployment

### Environment Configuration

Create `.env` file:

```bash
# App Configuration
ENVIRONMENT=production
DEBUG=False
APP_NAME="Facial Authentication System"
APP_VERSION="2.0.0"

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/facial_auth

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=["https://yourdomain.com"]

# Features
ENABLE_DEPTH_ESTIMATION=True
ENABLE_TEMPORAL_LIVENESS=True
ENABLE_ONLINE_LEARNING=True
ENABLE_ADAPTIVE_THRESHOLD=True
ENABLE_BIAS_MONITORING=True

# Monitoring
PROMETHEUS_ENABLED=True
```

### Backend Production Setup

```bash
# Install production dependencies
pip install -r requirements.txt

# Run with Gunicorn (production ASGI server)
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Frontend Production Build

```bash
cd frontend

# Build for production
npm run build

# The build output will be in frontend/dist/
# Serve with Nginx, Apache, or any static file server
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/your_cert.pem;
    ssl_certificate_key /etc/ssl/private/your_key.pem;

    # Frontend
    location / {
        root /var/www/facial-auth/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/facial-auth-backend.service`:

```ini
[Unit]
Description=Facial Authentication Backend
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/facial-auth
Environment="PATH=/opt/facial-auth/venv/bin"
ExecStart=/opt/facial-auth/venv/bin/gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable facial-auth-backend
sudo systemctl start facial-auth-backend
sudo systemctl status facial-auth-backend
```

## Cloud Deployment

### AWS Deployment

#### Using EC2

1. Launch EC2 instance (t3.xlarge or larger)
2. Install Docker and Docker Compose
3. Clone repository
4. Run with Docker Compose

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Clone and deploy
git clone <repository>
cd facial-auth-system
docker-compose -f docker-compose.fullstack.yml up -d
```

#### Using ECS (Elastic Container Service)

1. Build and push Docker images to ECR
2. Create ECS task definitions
3. Configure load balancer
4. Deploy with Fargate or EC2 instances

#### Using Elastic Beanstalk

```bash
# Initialize EB application
eb init -p docker facial-auth-system

# Create environment
eb create facial-auth-prod

# Deploy
eb deploy
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/facial-auth-backend

# Deploy
gcloud run deploy facial-auth \
  --image gcr.io/PROJECT_ID/facial-auth-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Microsoft Azure

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name facial-auth-rg --location eastus

# Deploy container
az container create \
  --resource-group facial-auth-rg \
  --name facial-auth-backend \
  --image yourregistry.azurecr.io/facial-auth-backend:latest \
  --ports 8000 \
  --dns-name-label facial-auth-unique
```

### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create facial-auth-system

# Deploy
git push heroku main

# Scale
heroku ps:scale web=2
```

## Monitoring & Maintenance

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# System info
curl http://localhost:8000/api/v1/system/info
```

### Metrics

Access Prometheus metrics:
```bash
curl http://localhost:8000/metrics
```

View in Grafana:
- URL: http://localhost:3001
- Default credentials: admin/admin

### Logging

```bash
# View backend logs
docker-compose logs -f backend

# View all logs
docker-compose logs -f

# Filter by service
docker-compose logs -f backend frontend
```

### Backup Database

```bash
# Backup SQLite
cp facial_auth.db facial_auth_backup_$(date +%Y%m%d).db

# Backup PostgreSQL
docker-compose exec postgres pg_dump -U facial_auth_user facial_auth > backup.sql

# Restore PostgreSQL
docker-compose exec -T postgres psql -U facial_auth_user facial_auth < backup.sql
```

### Update Deployment

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f docker-compose.fullstack.yml up -d --build

# Or for zero-downtime
docker-compose -f docker-compose.fullstack.yml up -d --no-deps --build backend
```

### Performance Tuning

#### Backend
- Increase workers: Set `API_WORKERS` environment variable
- Enable caching: Use Redis for session storage
- Optimize database: Add indexes, use connection pooling
- Use GPU: Install CUDA-enabled PyTorch

#### Frontend
- Enable CDN for static assets
- Implement code splitting
- Use lazy loading for routes
- Optimize images

### Security Hardening

1. **SSL/TLS**: Always use HTTPS in production
2. **Firewall**: Limit access to necessary ports only
3. **Secrets**: Use environment variables, never commit secrets
4. **Updates**: Keep dependencies up to date
5. **Rate Limiting**: Configure in `app/core/config.py`
6. **Database**: Use strong passwords, enable encryption at rest

### Troubleshooting

#### Backend won't start
```bash
# Check logs
docker-compose logs backend

# Verify dependencies
pip list

# Test database connection
python -c "from app.core.database import init_db; init_db()"
```

#### Frontend build fails
```bash
# Clear cache
rm -rf node_modules frontend/dist
npm install

# Check Node version
node --version  # Should be 18+
```

#### High memory usage
- Reduce API workers
- Enable response compression
- Optimize model loading (lazy load)
- Increase swap space

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review documentation
- Open an issue on GitHub

## License

This project is licensed under the MIT License.


