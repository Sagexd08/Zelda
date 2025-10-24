# Full Stack Web Application Guide

## Overview

The Facial Authentication System is now a complete full-stack web application with:

### Frontend (React + Vite)
- **Location:** `frontend/`
- **Port:** 3000 (development)
- **Technology:** React 18, Tailwind CSS, Framer Motion, Recharts
- **Features:**
  - Beautiful, responsive UI
  - Real-time camera integration
  - WebSocket support for live authentication
  - Comprehensive analytics dashboard
  - User management interface
  - System settings panel

### Backend (FastAPI + PyTorch)
- **Location:** `app/`
- **Port:** 8000
- **Technology:** FastAPI, PyTorch, SQLAlchemy
- **Features:**
  - RESTful API endpoints
  - WebSocket connections
  - ML model serving
  - Database management
  - Prometheus metrics

## Project Structure

```
facial-auth-system/
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── api/                # API client
│   │   ├── components/         # Reusable components
│   │   │   ├── Navbar.jsx
│   │   │   ├── Card.jsx
│   │   │   ├── Button.jsx
│   │   │   └── WebcamCapture.jsx
│   │   ├── pages/             # Page components
│   │   │   ├── Home.jsx       # Landing page
│   │   │   ├── Register.jsx   # User registration
│   │   │   ├── Authenticate.jsx  # Face authentication
│   │   │   ├── Dashboard.jsx  # Analytics
│   │   │   ├── Users.jsx      # User management
│   │   │   └── Settings.jsx   # System config
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── Dockerfile
├── app/                        # Backend Application
│   ├── api/
│   │   ├── routes.py          # REST endpoints
│   │   └── websocket.py       # WebSocket handler
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   └── security.py
│   ├── models/                # ML models
│   └── services/              # Business logic
├── deployment/                # Deployment configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── nginx-frontend.conf
├── start.sh                   # Quick start (Linux/macOS)
├── start.bat                  # Quick start (Windows)
├── docker-compose.fullstack.yml  # Full stack deployment
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # Main documentation
```

## Quick Start

### Option 1: Automated Scripts (Recommended)

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
1. Creates Python virtual environment
2. Installs all Python dependencies
3. Installs all Node.js dependencies  
4. Initializes database
5. Starts backend on http://localhost:8000
6. Starts frontend on http://localhost:3000

### Option 2: Docker (Production-Ready)

```bash
# Full stack with all services
docker-compose -f docker-compose.fullstack.yml up -d

# View logs
docker-compose -f docker-compose.fullstack.yml logs -f

# Stop all services
docker-compose -f docker-compose.fullstack.yml down
```

This includes:
- Frontend (Nginx)
- Backend (FastAPI)
- PostgreSQL database
- Redis cache
- Prometheus monitoring
- Grafana dashboards

### Option 3: Manual Setup

**Backend:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Features Guide

### 1. Home Page (`/`)
- Landing page with feature showcase
- System statistics
- Quick navigation to other sections
- Technology stack overview

### 2. User Registration (`/register`)
- **Camera Capture:** Real-time face capture with multiple images
- **File Upload:** Upload existing photos
- **Face Detection:** Automatic face detection and validation
- **Quality Assessment:** Real-time quality scoring
- **Liveness Check:** Anti-spoofing verification

**Usage:**
1. Enter a unique User ID
2. Choose camera or upload mode
3. Capture/upload 3+ face images
4. System validates and registers user
5. View registration success with metrics

### 3. Authentication (`/authenticate`)
- **Live Video Mode:** Continuous face monitoring
- **Single Shot Mode:** One-time capture and verify
- **Real-time Detection:** Instant face detection feedback
- **Liveness Verification:** Active anti-spoofing
- **Confidence Scoring:** Detailed authentication results

**Usage:**
1. Enter User ID to authenticate
2. Start camera
3. Position face in frame
4. System automatically authenticates
5. View detailed results with confidence scores

### 4. Dashboard (`/dashboard`)
- **Real-time Metrics:** Authentication activity, success rate
- **Performance Charts:** FPS, accuracy, latency over time
- **Model Comparison:** Performance across different models
- **Liveness Analytics:** Pie chart of detection results
- **System Health:** Status of all services
- **Recent Activity:** Timeline of authentication events

### 5. User Management (`/users`)
- **User Database:** View all registered users
- **Search Functionality:** Find users by ID
- **User Details:** View complete user information
- **Delete Users:** Remove users and their data
- **Statistics:** Authentication history, success rate per user

### 6. Settings (`/settings`)
- **Detection Parameters:** Configure thresholds and sizes
- **Feature Toggles:** Enable/disable system features
- **Performance Settings:** Workers, caching, logging
- **Database Info:** View database status and size
- **Model Configuration:** Adjust ML model parameters

## API Endpoints

### REST API

```
POST   /api/v1/register          # Register new user
POST   /api/v1/authenticate      # Authenticate user
POST   /api/v1/identify          # Identify unknown face
POST   /api/v1/delete_user       # Delete user
GET    /api/v1/system/info       # System information
GET    /health                   # Health check
GET    /metrics                  # Prometheus metrics
GET    /docs                     # API documentation (Swagger)
```

### WebSocket

```
WS     /ws/{client_id}           # Real-time face recognition
```

**WebSocket Message Format:**
```json
{
  "type": "frame",
  "frame": "base64_encoded_image",
  "user_id": "john_doe",
  "session_id": "unique_session_id"
}
```

**Response:**
```json
{
  "status": "authenticated",
  "authenticated": true,
  "confidence": 0.92,
  "liveness_score": 0.87,
  "threshold": 0.6
}
```

## Development Workflow

### Making Frontend Changes

1. Edit files in `frontend/src/`
2. Changes auto-reload in browser
3. Test in browser at http://localhost:3000
4. Build for production: `npm run build`

### Making Backend Changes

1. Edit files in `app/`
2. Server auto-reloads with `--reload` flag
3. Test API at http://localhost:8000/docs
4. Run tests: `pytest`

### Adding New Features

**Frontend Component:**
```jsx
// frontend/src/components/NewComponent.jsx
import { motion } from 'framer-motion'
import Card from './Card'

const NewComponent = () => {
  return (
    <Card>
      <h2>New Feature</h2>
    </Card>
  )
}

export default NewComponent
```

**Backend Endpoint:**
```python
# app/api/routes.py
@router.post("/new_endpoint")
async def new_endpoint(data: RequestModel):
    # Process data
    return {"result": "success"}
```

## Customization

### Styling

Edit `frontend/tailwind.config.js` to customize colors:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        500: '#your-color',
        600: '#your-darker-color',
      }
    }
  }
}
```

### Configuration

Backend config in `app/core/config.py`:
```python
VERIFICATION_THRESHOLD = 0.6  # Adjust similarity threshold
LIVENESS_THRESHOLD = 0.5      # Adjust liveness threshold
MIN_REGISTRATION_SAMPLES = 3  # Min images for registration
```

## Troubleshooting

### Frontend Issues

**Port already in use:**
```bash
# Change port in package.json or vite.config.js
vite --port 3001
```

**Dependencies error:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### Backend Issues

**Import errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Database errors:**
```bash
# Reset database
rm facial_auth.db
python -c "from app.core.database import init_db; init_db()"
```

### CORS Issues

If frontend can't reach backend, update CORS settings in `app/core/config.py`:

```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:3000"
]
```

## Production Deployment

### Build Frontend

```bash
cd frontend
npm run build
# Output in frontend/dist/
```

### Build Backend

```bash
docker build -f deployment/Dockerfile -t facial-auth-backend .
```

### Deploy with Docker

```bash
docker-compose -f docker-compose.fullstack.yml up -d
```

### Environment Variables

Create `.env` file:
```bash
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key
CORS_ORIGINS=["https://yourdomain.com"]
```

## Performance Optimization

### Frontend
- Code splitting (already configured)
- Lazy loading routes
- Image optimization
- CDN for static assets

### Backend
- Increase API workers: `API_WORKERS=8`
- Enable Redis caching
- Use PostgreSQL instead of SQLite
- Enable GPU acceleration

## Security

### Best Practices
1. Use HTTPS in production
2. Set strong SECRET_KEY
3. Enable rate limiting
4. Use environment variables for secrets
5. Regular security updates
6. Monitor with Prometheus/Grafana

### Rate Limiting

Already configured in backend:
- Enabled by default
- Configurable in `app/core/config.py`
- Per-IP limiting

## Monitoring

### Prometheus Metrics

Access at http://localhost:9090

Key metrics:
- `facial_auth_requests_total` - Total requests
- `facial_auth_request_latency_seconds` - Request latency

### Grafana Dashboards

Access at http://localhost:3001 (Docker deployment)

Pre-configured dashboards:
- System Overview
- API Performance
- Authentication Analytics

## Testing

### Backend Tests
```bash
pytest
pytest --cov=app tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Integration Tests
```bash
# Start both servers
./start.sh

# Run E2E tests
npm run test:e2e
```

## Support & Resources

- **Documentation:** See `README.md` and `DEPLOYMENT.md`
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **System Info:** http://localhost:8000/api/v1/system/info

## Next Steps

1. **Customize branding** - Update colors, logos, text
2. **Add authentication** - Implement user login system
3. **Enhance security** - Add JWT tokens, RBAC
4. **Scale deployment** - Use Kubernetes, load balancers
5. **Monitor production** - Set up alerts, logging
6. **Optimize models** - Fine-tune for your use case

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

MIT License - See LICENSE file for details


