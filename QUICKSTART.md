# Quick Start Guide

Get the Facial Authentication System running in 5 minutes!

## Prerequisites

- Python 3.9+
- 4GB+ RAM
- (Optional) NVIDIA GPU with CUDA 11.8+

## Installation Steps

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/facial-auth-system.git
cd facial-auth-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use nano, vim, or any editor)
nano .env
```

**Important settings to change:**
- `SECRET_KEY` - Generate with: `openssl rand -hex 32`
- `ENCRYPTION_KEY` - Generate with: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- `DATABASE_URL` - Keep default for SQLite, or use PostgreSQL URL

### 3. Download Model Weights

```bash
# Create weights directory
mkdir -p weights

# Download FaceNet (auto-downloads on first use)
# Download ArcFace
wget https://github.com/deepinsight/insightface/releases/download/v0.7/arcface_resnet100.pth -O weights/arcface_resnet100.pth

# Download MobileFaceNet
wget https://github.com/sirius-ai/MobileFaceNet_Pytorch/raw/master/model.pth -O weights/mobilefacenet.pth
```

See `weights/README.md` for complete download instructions.

### 4. Initialize Database

```bash
python -c "from app.core.database import init_db; init_db()"
```

### 5. Start Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: **http://localhost:8000**

## Test the API

### Check Health

```bash
curl http://localhost:8000/health
```

### View API Documentation

Open in browser: **http://localhost:8000/docs**

### Register a User (Python)

```python
import requests

url = "http://localhost:8000/api/v1/register"

# Prepare 5+ face images
files = [
    ('images', open('face1.jpg', 'rb')),
    ('images', open('face2.jpg', 'rb')),
    ('images', open('face3.jpg', 'rb')),
    ('images', open('face4.jpg', 'rb')),
    ('images', open('face5.jpg', 'rb')),
]

data = {'user_id': 'john_doe'}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Authenticate User

```python
import requests

url = "http://localhost:8000/api/v1/authenticate"

files = {'image': open('test_face.jpg', 'rb')}
data = {'user_id': 'john_doe'}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Authenticated: {result['authenticated']}")
print(f"Confidence: {result['confidence']}")
```

## Run Demo Application

### Streamlit Web Demo

```bash
streamlit run demo/streamlit_app.py
```

Open: **http://localhost:8501**

### Evaluation Notebook

```bash
jupyter notebook notebooks/evaluate_system.ipynb
```

## Docker Deployment (Alternative)

### Quick Start with Docker

```bash
cd deployment
docker-compose up -d
```

Services available at:
- API: http://localhost:8000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Common Issues

### Issue: Models not loading
**Solution**: Ensure weights are downloaded to `weights/` directory

### Issue: CUDA out of memory
**Solution**: Set `DEVICE=cpu` in `.env` file

### Issue: No face detected
**Solution**: Ensure image has clear frontal face with good lighting

### Issue: Port already in use
**Solution**: Change port in `.env`: `API_PORT=8001`

## Next Steps

1. **Training**: See `training/README.md` for model training
2. **Deployment**: See `deployment/` for production setup
3. **API Docs**: Visit http://localhost:8000/docs for interactive API
4. **Documentation**: Read `README.md` for complete system documentation

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/facial-auth-system/issues)
- **Documentation**: Full docs in `README.md`
- **Examples**: Check `notebooks/` and `demo/` directories

---

**Ready to authenticate!** 🔐

