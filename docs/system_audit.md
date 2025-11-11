# Zelda Facial Auth System – Scope Review Audit

## Executive Summary
- Backend FastAPI service implements registration/authentication/identification flows with rich model hooks but lacks auth/role enforcement, resilience around database failures, and production-ready secrets or config validation (`app/api/routes.py`, `app/core/config.py`).
- Frontend React app offers registration/authentication UX but simulates live detection via timers and omits websocket-driven status, so advertised real-time checks are not wired to backend (`frontend/src/pages/Authenticate.jsx`).
- ML training scripts cover fusion, liveness, temporal pipelines yet rely on synthetic data generation and placeholder features, leaving no reproducible path from raw data to deployed weights (`training/train_fusion.py`, `training/dataset_loaders.py`).
- Deployment artifacts span Docker Compose, Kubernetes, and CI, but environment expectations (GPU libs, secrets, health endpoints) diverge across manifests and lack alignment with `requirements.txt` and runtime health checks (`deployment/Dockerfile`, `docker-compose.fullstack.yml`).

## Backend Inventory & Gaps
- **API surface**: `/api/v1` routes cover register/authenticate/identify/delete/challenge/adaptive plus `/system` info (`app/api/routes.py`). No auth tokens or RBAC guard the endpoints; sensitive operations (delete) depend on user_id query only.
- **Input handling**: File uploads decoded in-memory without size limits; invalid images silently skipped during registration leading to potential all-invalid uploads returning 400 (`decode_image`). Missing explicit DB transaction rollbacks around commits.
- **Service layer**: Authentication service logs to `AuditLog` and updates counters, but assumes presence of primary embedding; failure paths for DB errors or decryption exceptions bubble up as 500s without catch (`app/services/authentication_service.py`).
- **Configuration**: Pydantic settings default to insecure secrets and enable many features (adaptive/bias/redis) regardless of environment. No validation that referenced weight files exist; redis toggle default false despite services importing cache (`app/core/config.py`).
- **Database**: SQLAlchemy models cover user embeddings, liveness, audit, bias metrics. Uses SQLite static pool by default; no migrations/alembic scripts despite dependency. When switching to Postgres, connection URL must be set manually; no automated schema migrations (`app/core/database.py`).
- **Observability/security**: Prometheus middleware enabled but no authentication on `/metrics`. Audit logging exists but no PII hashing for IP/UA unless populated upstream. Rate limiting via SlowAPI configured globally but tests absent.

## Frontend Inventory & Gaps
- **Feature coverage**: Pages for Home, Authenticate, Register, Dashboard, Users, Settings exist; only Authenticate examined shows simulated liveness overlay and random confidence data without backend websocket integration (`frontend/src/pages/Authenticate.jsx`).
- **API integration**: Axios client targets `/api/v1` endpoints with query-string `user_id` parameters, matching backend signature but lacking retry/backoff or error surface translation. Delete endpoint uses POST instead of DELETE, consistent with backend but non-RESTful (`frontend/src/api/client.js`).
- **State management**: Local component state only; no global store or persisted session. MFA/voice features advertised in UI not backed by API responses.
- **Build artifacts**: `frontend/dist` checked in, implying manual build; deployment references Nginx static config but no automated CI build pipeline.

## Training & ML Pipeline Status
- **Fusion training**: `training/train_fusion.py` can extract embeddings from dataset but defaults to synthetic random embeddings, hindering reproducibility. No experiment tracking or config serialization; outputs weight file to `weights/fusion_mlp.pth` without checksum/versioning.
- **Dataset loaders**: Provide dataset classes but return zero tensors when files missing, silently degrading training quality. Temporal liveness dataset returns random feature vectors instead of computed features, acting as placeholder (`training/dataset_loaders.py`).
- **AutomL**: `app/automl/hyperparameter_tuning.py` (not yet reviewed) likely stub; no orchestration scripts tying training -> evaluation -> deployment. No pipeline definitions (e.g., Airflow/Prefect) despite README implying automation.
- **Weight management**: `weights/` contains sample `fusion_mlp.pth` but no provenance; settings expect numerous models (`arcface`, `retinaface`, etc.) that are absent, creating runtime failure risk.

## Deployment & Infrastructure Review
- **Docker**: Production Dockerfile installs CPU-only stack on `python:3.10-slim`, but requirements include GPU-focused packages (`torch>=2.6.0`, `timm`) that may lack wheels; healthcheck hits `/health` via `requests` though service exposes unauthenticated HTTP. GPU stage references `python3` but omits dependency install (`requirements.txt`); no conditional build instructions.
- **Docker Compose**: Fullstack compose provisions Postgres, Redis, Prometheus, Grafana, backend, frontend. Backend env values hardcode `ENVIRONMENT=production` but do not supply SECRET_KEY, encryption keys, or weights volume beyond read-only `./weights`; DB volume mapping duplicates with `facial_auth_db`. Compose healthcheck expects `/health` but service requires DB migrations to run first (`docker-compose.fullstack.yml`).
- **Kubernetes**: Manifests in `deployment/kubernetes/` need review for parity with Docker Compose (likely not referencing Redis). Potential mismatch between Render/Vercel configs and AWS stack.
- **CI/CD**: `.github/workflows/deploy.yml` currently calls deprecated `aws ecs create-deployment`; fix pending to switch to `aws ecs update-service`. Pipeline likely missing lint/test stages before deploy.

## README Claims vs Reality
- README advertises live adaptive learning, voice auth, federated features, and production readiness. Implementation stubs exist (`app/federated/*`, `app/models/voice_stub.py`), but no integration endpoints or training data; features effectively placeholders pending pipeline work.
- Documentation references GPU acceleration and on-device components, yet repository lacks CUDA build instructions or edge deployment configs.

## Environment Parity Assessment
- `requirements.txt` pins heavy CV/ML libs requiring system packages (OpenCV, torch). Dockerfile installs `libopencv-dev` but not `ffmpeg`, `libgl1`, `libssl`, which torch/insightface may need.
- Local `.env.example` not propagated into deployment manifests; secrets must be set manually. Compose/Render configs diverge in hostnames and ports causing potential CORS conflicts (`frontend/src/api/client.js` vs `app/core/config.py`).

## Immediate Gap List
1. Harden backend by adding authentication/authorization, enforcing upload limits, validating config secrets, and covering DB failure paths.
2. Replace frontend mock states with actual backend signal integration (websocket) and ensure feature toggles mirror backend capabilities.
3. Build reproducible ML pipeline with real datasets, evaluation metrics, and tracked artifact export for `weights/`.
4. Align infrastructure manifests and CI/CD to consistent environment (Postgres vs SQLite, Redis enabled, secrets managed) and update ECS deployment command.
5. Update documentation to reflect true system capabilities and provide clear setup instructions for data/models.
