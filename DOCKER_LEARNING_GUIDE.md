# Docker Learning Guide: Using MediaPipe ASL Recognition Project as Example

**A beginner's guide to understanding Docker using a real-world project**

---

## Table of Contents

1. [What is Docker? (The Big Picture)](#what-is-docker-the-big-picture)
2. [Core Concepts](#core-concepts)
3. [How Docker Works in This Project](#how-docker-works-in-this-project)
4. [Breaking Down the Dockerfile](#breaking-down-the-dockerfile)
5. [Understanding docker-compose](#understanding-docker-compose)
6. [Common Docker Commands](#common-docker-commands)
7. [Development vs Production](#development-vs-production)
8. [Troubleshooting & Tips](#troubleshooting--tips)

---

## What is Docker? (The Big Picture)

### The Problem Docker Solves

Imagine you've built an amazing ASL recognition application on your laptop. It works perfectly! But when your friend tries to run it on their computer, they get errors:
- Missing Python packages
- Different Python version
- Missing system libraries (like `libgl1`, `libsm6`)
- Different OS (Windows vs Linux)

### The Solution: Docker

**Docker is like a shipping container for software.** 

Just like physical shipping containers:
- ✅ Standardized
- ✅ Portable (works on any ship/harbor)
- ✅ Contains everything needed inside
- ✅ Isolated from the outside world

**Docker containers** work the same way:
- ✅ Contain your entire application
- ✅ Include all dependencies (Python packages, system libraries)
- ✅ Run identically everywhere (your laptop, friend's laptop, AWS server)
- ✅ Isolated from other applications on the same system

### Key Benefit

> "It works on my machine" → becomes → "It works everywhere"

---

## Core Concepts

### 1. **Image** 📦
A **Docker image** is like a recipe or blueprint. It's a read-only template that describes:
- What OS to use
- What software to install
- What files to copy
- What commands to run
- Environment variables

**Think of it like:** A detailed recipe for baking a cake

**In this project:** The `Dockerfile` creates an image for the ASL recognition API

### 2. **Container** 🏃
A **Docker container** is a running instance of an image.

**Think of it like:** An actual cake you baked using the recipe

**In this project:** When you run `docker compose up`, it creates containers from images

```
Image (recipe) ──build──> Container (running app)
```

### 3. **Layer** 📚
Docker images are built in layers, like a stack of pancakes:

```
Layer 1: Python 3.10 base OS
    ↓
Layer 2: System libraries installed
    ↓
Layer 3: Python packages installed
    ↓
Layer 4: Application code copied
```

Each layer is cached. If you rebuild, Docker skips unchanged layers (faster!).

### 4. **Dockerfile** 📄
A text file with instructions to build an image. Like a script that says:
1. Start with Python 3.10
2. Install these system packages
3. Copy this code
4. Run this command

### 5. **docker-compose** 🎼
A tool to define and run **multiple containers** together.

**Think of it like:** Managing an orchestra (API container + Nginx container + databases, etc.)

---

## How Docker Works in This Project

### Project Architecture

```
Your Computer (or AWS Server)
│
└── Docker Daemon (Background service)
    │
    ├── Container 1: ASL API (Python + TensorFlow + MediaPipe)
    │
    └── Container 2: Nginx (Web server)
        │
        └── Serves at localhost:80 and localhost:443
```

### Step-by-Step: What Happens When You Run Docker

#### **Step 1: Building the Image**

```bash
docker compose build
```

This runs all the instructions in the `Dockerfile`:

```
1. Download Python 3.10 slim image
2. Install system dependencies (libgl1, libsm6, etc.)
3. Create Python virtual environment
4. Install Python packages from requirements.txt
5. Copy your code into the container
```

**Result:** A Docker image (about 3-4 GB) saved on your computer

#### **Step 2: Running the Container**

```bash
docker compose up
```

This:
1. Takes the image you built
2. Creates a container from it
3. Starts the application inside
4. Maps ports (8000 → 8000)
5. Sets environment variables

**Result:** Your app is running!

---

## Breaking Down the Dockerfile

Let's understand the actual [Dockerfile](Dockerfile) used in this project:

### Part 1: The Builder Stage (Multi-stage Build)

```dockerfile
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS builder
```

**What it does:**
- `FROM`: Start with Python 3.10 slim image (official Python image, minimal OS)
- `AS builder`: Name this stage "builder" (used for multi-stage build)

**Why slim?** Smaller, faster, only includes essentials (no games, text editors, etc.)

---

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
```

**What it does:**
- `PYTHONDONTWRITEBYTECODE=1`: Don't create `.pyc` files (saves space)
- `PYTHONUNBUFFERED=1`: Show logs immediately (don't buffer)
- `PIP_NO_CACHE_DIR=1`: Don't save pip cache (saves space)

**Why?** Make the image smaller and logs easier to debug

---

```dockerfile
WORKDIR /app
```

**What it does:**
- Creates `/app` directory inside the container
- All commands after this run from `/app`

**Think of it like:** `cd /app` on Linux

---

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    ...
```

**What it does:**
- `RUN`: Execute a command during build
- `apt-get`: Ubuntu package manager (like `npm` for system packages)
- `--no-install-recommends`: Only install necessary packages (saves space)
- Installs libraries needed for:
  - `build-essential`: Compiling Python packages with C extensions
  - `libgl1`, `libsm6`: Required by OpenCV (computer vision library)
  - `libxext6`, `libxrender1`: X11 display libraries

**Why?** These libraries are required by TensorFlow, MediaPipe, and OpenCV

---

```dockerfile
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
```

**What it does:**
- Creates a Python virtual environment (isolated Python installation)
- Updates `PATH` so `python` and `pip` commands use the venv

**Think of it like:** Having a separate Python installation just for this app

---

```dockerfile
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
```

**What it does:**
- `COPY`: Copy `requirements.txt` from your computer into the container
- `pip install`: Install all Python packages (TensorFlow, MediaPipe, FastAPI, etc.)

**Result:** All Python dependencies are installed

---

### Part 2: The Runtime Stage

```dockerfile
FROM python:${PYTHON_VERSION}-slim AS runtime
```

**What it does:**
- Start fresh with a new, clean Python image
- Name this stage "runtime"

**Why separate stages?**
- **Builder stage:** Includes build tools (large files)
- **Runtime stage:** Only includes what's needed to run (small and clean)
- **Result:** Final image is 70% smaller! 🚀

---

```dockerfile
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="ASL Recognition API with MediaPipe and TensorFlow"
```

**What it does:**
- Metadata about the image
- Useful for documentation

---

```dockerfile
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MEDIAPIPE_DISABLE_GPU=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HOME=/tmp
```

**What it does:**
- `TF_CPP_MIN_LOG_LEVEL=2`: Reduce TensorFlow logging (less spam)
- `MEDIAPIPE_DISABLE_GPU=1`: Use CPU only (not all servers have GPU)
- `MPLCONFIGDIR=/tmp/matplotlib`: Fix matplotlib config directory
- `HOME=/tmp`: Set home directory for temporary files

**Why?** Configure libraries for container environment

---

```dockerfile
RUN adduser \
    --disabled-password \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    appuser
```

**What it does:**
- Creates a non-root user called `appuser`
- `--disabled-password`: Can't login with password (for security)
- `--shell "/sbin/nologin"`: Can't get a shell (more secure)

**Why?** Security! Running as root inside a container is dangerous. If someone compromises the app, they have full system access. A limited user has fewer permissions.

---

```dockerfile
COPY --from=builder /app/venv /app/venv
```

**What it does:**
- Copies the Python virtual environment from the builder stage
- `--from=builder`: Reference the builder stage

**Why?** Avoids reinstalling all packages in the runtime stage

---

```dockerfile
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser models/ ./models/
```

**What it does:**
- Copies application files from your computer into the container
- `--chown=appuser:appuser`: Sets owner to the non-root user

**What's copied:**
- `main.py`: FastAPI application
- `realtime_predict.py`: ASL recognition logic
- `models/`: Pre-trained TensorFlow and encoder models
- `utils/`: Helper functions

---

```dockerfile
USER appuser
```

**What it does:**
- Switch to running as `appuser` (not root)

**Why?** Everything after this runs with limited permissions

---

```dockerfile
EXPOSE 8000
```

**What it does:**
- Declares that this container listens on port 8000
- `EXPOSE` doesn't actually publish the port—it's documentation

**Think of it like:** "This app will listen on port 8000"

---

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

**What it does:**
- Every 30 seconds, run `curl http://localhost:8000/health`
- If it fails 3 times, mark the container as unhealthy
- `--start-period=5s`: Wait 5 seconds before first check (let app start)

**Why?** Docker can automatically restart unhealthy containers

---

## Understanding docker-compose

`docker-compose` makes it easy to run multiple containers together. Let's look at [docker-compose.yml](docker-compose.yml):

### The API Service

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: asl-api
```

**What it does:**
- Defines a service called `api`
- `build`: Build from the Dockerfile in current directory
- `container_name`: Name the container `asl-api`

---

```yaml
    restart: unless-stopped
```

**What it does:**
- If the container crashes, automatically restart it
- `unless-stopped`: Don't restart if you manually stopped it

---

```yaml
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
      - PYTHONUNBUFFERED=1
```

**What it does:**
- Sets environment variables inside the container
- `TF_CPP_MIN_LOG_LEVEL=2`: Reduce TensorFlow logging

**Think of it like:** Setting environment variables on Linux

---

```yaml
    expose:
      - "8000"
```

**What it does:**
- The container listens on port 8000
- `expose`: Only visible to other containers (not to host)

**Difference:**
- `EXPOSE` in Dockerfile: Documentation
- `expose` in compose: Actually make port available to other containers

---

```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

**What it does:**
- Every 30 seconds, check if the API is healthy
- Run `curl http://localhost:8000/health`
- If it returns non-200 status, count as failure
- After 3 failures, mark as unhealthy

---

```yaml
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

**What it does:**
- `limits`: Maximum CPU and memory the container can use
- `reservations`: Minimum resources to guarantee

**Why?** Prevent one container from using all system resources

---

### The Nginx Service

```yaml
  nginx:
    image: nginx:alpine
    container_name: asl-nginx
    restart: unless-stopped
```

**What it does:**
- Uses the official `nginx:alpine` image (pre-built, not building from scratch)
- `alpine`: Minimal Linux image (very small)

**Why two containers?**
- **API container:** Runs your Python app (internal only)
- **Nginx container:** Reverse proxy, handles HTTPS, serves on ports 80/443

---

```yaml
    ports:
      - "80:80"
      - "443:443"
```

**What it does:**
- Maps host ports to container ports
- `80:80` means: listen on port 80 on your computer, forward to port 80 in container
- External users access port 80 → Nginx on port 80 → forwards to API on port 8000

**Flow:**
```
Internet (Port 80) 
    ↓
Nginx Container (Port 80) 
    ↓
API Container (Port 8000)
```

---

```yaml
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
```

**What it does:**
- Mount files/folders from your computer into the container
- `./nginx/nginx.conf` → `/etc/nginx/nginx.conf:ro`
  - `ro` = read-only (can't modify from inside container)

**Why?** Nginx configuration and SSL certificates are on your computer, not in the image

---

```yaml
    depends_on:
      api:
        condition: service_healthy
```

**What it does:**
- Don't start Nginx until the API is healthy
- Checks the `healthcheck` we defined

**Why?** Prevents Nginx from crashing if it can't reach the API

---

### Networks

```yaml
networks:
  asl-network:
    driver: bridge
```

**What it does:**
- Creates a Docker network called `asl-network`
- All containers connected to this network can communicate with each other by name

**How containers talk:**
```
Container A → uses name "api" → resolves to IP of API container
```

---

## Development vs Production

### Production Compose ([docker-compose.yml](docker-compose.yml))

```yaml
# No volumes - uses code from image
# No ports on API - only Nginx is exposed
# Restart policy enabled
# Resource limits set
```

**For deployment on AWS:** Optimized for speed and stability

### Development Compose ([docker-compose.dev.yml](docker-compose.dev.yml))

```yaml
volumes:
  - ./main.py:/app/main.py:ro
  - ./realtime_predict.py:/app/realtime_predict.py:ro
  - ./utils:/app/utils:ro
```

**What it does:**
- Mounts source code from your computer
- Changes to files on your computer are reflected in container immediately

---

```yaml
command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**What it does:**
- Override the default command
- `--reload`: Auto-restart when files change (hot reload)

**Perfect for development:** Edit code → auto-reloads → test → repeat

---

```yaml
ports:
  - "8000:8000"
```

**What it does:**
- Expose port 8000 on your computer
- Can access at `http://localhost:8000`

---

### How to Use Both

```bash
# Production (single compose file)
docker compose up

# Development (overlay development on production)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**How it works:**
- Base configuration from `docker-compose.yml`
- Overrides/additions from `docker-compose.dev.yml`
- Result: All production settings + hot reload for development

---

## Common Docker Commands

### Building

```bash
# Build the image
docker compose build

# Build without cache (fresh build)
docker compose build --no-cache

# Build specific service
docker compose build api
```

### Running

```bash
# Start containers in background
docker compose up -d

# Start containers and show logs
docker compose up

# Start with development overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Run one-time command
docker compose run api python train_classifier.py
```

### Managing

```bash
# View running containers
docker compose ps

# View logs
docker compose logs
docker compose logs -f        # Follow (like 'tail -f')
docker compose logs api       # Specific service

# Stop containers
docker compose stop

# Stop and remove containers
docker compose down

# Remove volumes too (destructive!)
docker compose down -v

# Restart
docker compose restart api
```

### Debugging

```bash
# Connect to running container
docker compose exec api bash

# Run command in container
docker compose exec api python -c "import tensorflow; print(tensorflow.__version__)"

# View container info
docker compose inspect api

# View top (CPU/memory usage)
docker compose top api
```

### Images

```bash
# List images
docker image ls

# Remove image
docker image rm asl-api

# View image layers
docker image history asl-api

# Inspect image details
docker image inspect asl-api
```

---

## Troubleshooting & Tips

### Problem: "Address already in use"

**Error:**
```
Error: bind: address already in use
```

**Solution:**
Another service is using port 80 or 443. Find and stop it:

```bash
# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 80).OwningProcess

# Then stop it, or use different port:
docker compose -p myport up -d
```

---

### Problem: "Out of memory"

**Solution:**
Docker containers are using too much memory. Check:

```bash
docker compose stats
```

Increase Docker's memory limit in Docker Desktop settings, or use resource limits in compose:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

---

### Problem: "Models not loading"

**Solution:**
Files aren't being copied correctly:

```bash
# Check what's in the container
docker compose exec api ls -la /app/models

# Check if Dockerfile is copying correctly
docker compose build --no-cache
```

---

### Problem: "Permissions denied"

**Solution:**
Files belong to root instead of appuser:

```bash
# Fix in Dockerfile
COPY --chown=appuser:appuser models/ ./models/
```

---

### Tip: View Docker Disk Usage

```bash
docker system df
```

Shows:
- Images (total size)
- Containers (disk used)
- Volumes (persistent storage)

---

### Tip: Clean Up

```bash
# Remove unused containers/images/networks
docker system prune

# Also remove volumes
docker system prune -a --volumes
```

---

## Key Takeaways

| Concept | Think Of It As | In This Project |
|---------|--------|---|
| **Image** | Recipe | Dockerfile (blueprint for API) |
| **Container** | Cooked dish | Running API + Nginx |
| **Layer** | Ingredient added | Python → packages → code |
| **Volume** | External storage | nginx.conf, SSL files |
| **Network** | Telephone line | Containers talk via `asl-network` |
| **Port** | Door | 80/443 → 8000 |
| **Healthcheck** | Pulse monitor | Checks if API is responding |

---

## Next Steps

1. ✅ Run `docker compose up` and access the API at `http://localhost`
2. ✅ Make a code change in `main.py` → see hot reload in dev mode
3. ✅ Check logs: `docker compose logs -f`
4. ✅ Connect to container: `docker compose exec api bash`
5. ✅ Read official Docker docs: https://docs.docker.com/

---

## Resources

- **Docker Official Docs:** https://docs.docker.com/
- **Dockerfile Reference:** https://docs.docker.com/engine/reference/builder/
- **Compose File Reference:** https://docs.docker.com/compose/compose-file/
- **Best Practices:** https://docs.docker.com/develop/dev-best-practices/

---

**Happy containerizing! 🐳**
