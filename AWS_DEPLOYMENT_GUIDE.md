# 🚀 AWS EC2 Deployment Guide

## Complete Guide: Deploying ASL Recognition API to AWS EC2

This comprehensive guide walks you through deploying the ASL Recognition API to AWS EC2 with Docker, Nginx reverse proxy, SSL/TLS, and CI/CD automation.

---

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Architecture Overview](#-architecture-overview)
3. [AWS Setup](#-aws-setup)
   - [Create IAM User](#1-create-iam-user)
   - [Create ECR Repository](#2-create-ecr-repository)
   - [Launch EC2 Instance](#3-launch-ec2-instance)
   - [Configure Security Groups](#4-configure-security-groups)
4. [EC2 Instance Setup](#-ec2-instance-setup)
   - [Install Docker](#1-install-docker)
   - [Configure AWS CLI](#2-configure-aws-cli)
   - [Deploy Application](#3-deploy-application)
5. [Domain & SSL Setup](#-domain--ssl-setup)
6. [CI/CD with GitHub Actions](#-cicd-with-github-actions)
7. [Monitoring & Logging](#-monitoring--logging)
8. [Troubleshooting](#-troubleshooting)
9. [Cost Estimation](#-cost-estimation)
10. [Security Best Practices](#-security-best-practices)

---

## 📦 Prerequisites

Before starting, ensure you have:

- [ ] **AWS Account** with billing enabled
- [ ] **GitHub Account** (for CI/CD)
- [ ] **Domain Name** (optional, for custom domain)
- [ ] **Local Tools**:
  - Docker Desktop installed
  - AWS CLI installed (`aws --version`)
  - Git installed

### Local Testing First

Test your Docker setup locally before deploying:

```powershell
# Build and run locally
docker compose build
docker compose up -d

# Test the API
curl http://localhost/health

# Check logs
docker compose logs -f

# Stop
docker compose down
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Cloud (ap-southeast-1)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   VPC (10.0.0.0/16)                        │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              Public Subnet (10.0.1.0/24)             │  │  │
│  │  │                                                       │  │  │
│  │  │  ┌─────────────────────────────────────────────┐     │  │  │
│  │  │  │            EC2 Instance (m7i-flex.large)       │     │  │  │
│  │  │  │  ┌───────────────┐  ┌─────────────────────┐ │     │  │  │
│  │  │  │  │    Nginx      │  │   ASL API (Docker)  │ │     │  │  │
│  │  │  │  │  :80 / :443   │──│      :8000          │ │     │  │  │
│  │  │  │  │               │  │                     │ │     │  │  │
│  │  │  │  │  - SSL/TLS    │  │  - FastAPI          │ │     │  │  │
│  │  │  │  │  - WebSocket  │  │  - MediaPipe        │ │     │  │  │
│  │  │  │  │  - Rate Limit │  │  - TensorFlow       │ │     │  │  │
│  │  │  │  └───────────────┘  └─────────────────────┘ │     │  │  │
│  │  │  └─────────────────────────────────────────────┘     │  │  │
│  │  │                         │                             │  │  │
│  │  └─────────────────────────┼─────────────────────────────┘  │  │
│  │                            │                                 │  │
│  └────────────────────────────┼─────────────────────────────────┘  │
│                               │                                    │
│  ┌────────────────────────────▼────────────────────────────────┐  │
│  │                        ECR Registry                          │  │
│  │                   asl-recognition-api:latest                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Your Next.js App (Vercel) ──────WebSocket────────► EC2 /ws/recognize
                          ──────HTTP POST──────────► EC2 /predict-image/
```

---

## ☁️ AWS Setup

### 1. Create IAM User

Create a dedicated IAM user for deployments:

```bash
# Create user
aws iam create-user --user-name asl-deployer

# Attach necessary policies
aws iam attach-user-policy --user-name asl-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess

aws iam attach-user-policy --user-name asl-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess

aws iam attach-user-policy --user-name asl-deployer \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMFullAccess

# Create access keys
aws iam create-access-key --user-name asl-deployer
```

**Save the `AccessKeyId` and `SecretAccessKey`** - you'll need these for GitHub Secrets.

### 2. Create ECR Repository

```bash
# Create ECR repository
aws ecr create-repository \
    --repository-name asl-recognition-api \
    --region ap-southeast-1 \
    --image-scanning-configuration scanOnPush=true

# Get the repository URI (save this!)
aws ecr describe-repositories \
    --repository-names asl-recognition-api \
    --query 'repositories[0].repositoryUri' \
    --output text
```

**Example Output:** `123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/asl-recognition-api`

### 3. Launch EC2 Instance

#### Option A: AWS Console (Recommended for Beginners)

1. Go to **EC2 Dashboard** → **Launch Instance**
2. Configure:

   - **Name**: `asl-api-server`
   - **AMI**: Amazon Linux 2023 (or Ubuntu 22.04)
   - **Instance Type**: `m7i-flex.large` (2 vCPU, 8GB RAM)
   - **Key Pair**: Create new or select existing
   - **Network Settings**:
     - Allow SSH (port 22) from your IP
     - Allow HTTP (port 80) from anywhere
     - Allow HTTPS (port 443) from anywhere
   - **Storage**: 30GB gp3

3. Click **Launch Instance**

#### Option B: AWS CLI

```bash
# Create key pair
aws ec2 create-key-pair \
    --key-name asl-api-key \
    --query 'KeyMaterial' \
    --output text > asl-api-key.pem

chmod 400 asl-api-key.pem

# Create security group
aws ec2 create-security-group \
    --group-name asl-api-sg \
    --description "Security group for ASL API"

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-name asl-api-sg \
    --protocol tcp --port 22 --cidr YOUR_IP/32

aws ec2 authorize-security-group-ingress \
    --group-name asl-api-sg \
    --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-name asl-api-sg \
    --protocol tcp --port 443 --cidr 0.0.0.0/0

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type m7i-flex.large \
    --key-name asl-api-key \
    --security-groups asl-api-sg \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=asl-api-server}]'
```

### 4. Configure Security Groups

Ensure these ports are open:

| Port | Protocol | Source    | Purpose        |
| ---- | -------- | --------- | -------------- |
| 22   | TCP      | Your IP   | SSH access     |
| 80   | TCP      | 0.0.0.0/0 | HTTP traffic   |
| 443  | TCP      | 0.0.0.0/0 | HTTPS traffic  |
| 8000 | TCP      | VPC only  | API (internal) |

---

## 🖥️ EC2 Instance Setup

### 1. Connect to Your Instance

```bash
# SSH into your instance
ssh -i asl-api-key.pem ec2-user@YOUR_EC2_PUBLIC_IP

# Or for Ubuntu
ssh -i asl-api-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### 2. Install Docker

**For Amazon Linux 2023:**

```bash
# Update system
sudo dnf update -y

# Install Docker
sudo dnf install docker -y

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Log out and back in for group changes
exit
```

**For Ubuntu 22.04:**

```bash
# Update and install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in
exit
```

### 3. Configure AWS CLI on EC2

```bash
# Reconnect
ssh -i asl-api-key.pem ec2-user@YOUR_EC2_PUBLIC_IP

# Install AWS CLI (if not present)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
# Enter your Access Key ID, Secret Access Key, region (ap-southeast-1), and output format (json)
```

### 4. Deploy Application

```bash
# Create application directory
mkdir -p ~/asl-api
cd ~/asl-api

# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com
```

Create `docker-compose.prod.yml` on the EC2 instance:

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    image: 079079338393.dkr.ecr.ap-southeast-1.amazonaws.com/asl-recognition-api:latest
    container_name: asl-api
    restart: unless-stopped
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
      - PYTHONUNBUFFERED=1
      - ALLOWED_ORIGINS=https://sign-bridge-snowy.vercel.app,http://localhost:3000
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    networks:
      - asl-network
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 3G
        reservations:
          cpus: '0.5'
          memory: 1G

  nginx:
    image: nginx:alpine
    container_name: asl-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      api:
        condition: service_healthy
    networks:
      - asl-network

networks:
  asl-network:
    driver: bridge
EOF
```

Create Nginx config:

```bash
mkdir -p nginx

cat > nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent"';
    access_log /var/log/nginx/access.log main;

    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 10M;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    # WebSocket upgrade map
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    upstream asl_api {
        server api:8000;
        keepalive 32;
    }

    server {
        listen 80;
        server_name _;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;

        location /health {
            proxy_pass http://asl_api/health;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /predict-image/ {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://asl_api;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws/ {
            proxy_pass http://asl_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
            proxy_read_timeout 86400;
        }

        location / {
            proxy_pass http://asl_api;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
```

### 5. First Deployment (Manual)

Push your Docker image from local machine first:

```powershell
# On your local machine

# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

# Build the image
docker build -t asl-recognition-api .

# Tag for ECR
docker tag asl-recognition-api:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/asl-recognition-api:latest

# Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/asl-recognition-api:latest
```

Then on EC2:

```bash
# Pull and run
cd ~/asl-api
docker compose pull
docker compose up -d

# Check status
docker compose ps
docker compose logs -f
```

### 6. Verify Deployment

```bash
# Health check
curl http://localhost/health

# From your local machine
curl http://YOUR_EC2_PUBLIC_IP/health

# Check API docs
# Open in browser: http://YOUR_EC2_PUBLIC_IP/docs
```

---

## 🔐 Domain & SSL Setup (Optional)

> **Note:** If you're using the **API Proxy Pattern** (recommended), you **don't need SSL** on EC2!
> The proxy handles HTTPS on Vercel's side. Skip this section unless you need a custom domain.

### 1. Configure Domain (Route 53 or External DNS)

Point your domain to your EC2 Elastic IP:

```
Type: A
Name: api.yourdomain.com
Value: YOUR_EC2_ELASTIC_IP
TTL: 300
```

### 2. Install SSL with Let's Encrypt

```bash
# Install Certbot
sudo dnf install certbot python3-certbot-nginx -y  # Amazon Linux
# OR
sudo apt install certbot python3-certbot-nginx -y  # Ubuntu

# Stop nginx temporarily
docker compose down

# Get SSL certificate
sudo certbot certonly --standalone -d api.yourdomain.com

# Copy certificates to nginx folder
sudo cp /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem ~/asl-api/nginx/ssl/
sudo cp /etc/letsencrypt/live/api.yourdomain.com/privkey.pem ~/asl-api/nginx/ssl/
sudo chown $USER:$USER ~/asl-api/nginx/ssl/*
```

### 3. Update Nginx for HTTPS

Update `nginx/nginx.conf`:

```nginx
# Add this server block for HTTPS
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # ... rest of location blocks ...
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### 4. Auto-Renew SSL

```bash
# Create renewal script
cat > ~/renew-ssl.sh << 'EOF'
#!/bin/bash
cd ~/asl-api
docker compose down
certbot renew
cp /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem ~/asl-api/nginx/ssl/
cp /etc/letsencrypt/live/api.yourdomain.com/privkey.pem ~/asl-api/nginx/ssl/
docker compose up -d
EOF

chmod +x ~/renew-ssl.sh

# Add to crontab (runs monthly)
(crontab -l 2>/dev/null; echo "0 0 1 * * /home/ec2-user/renew-ssl.sh") | crontab -
```

---

## 🔄 CI/CD with GitHub Actions

### 1. Add GitHub Secrets

Go to your repository → **Settings** → **Secrets and variables** → **Actions**

Add these secrets:

| Secret Name             | Value                                          |
| ----------------------- | ---------------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | Your IAM access key                            |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key                            |
| `ECR_REGISTRY`          | `123456789012.dkr.ecr.us-east-1.amazonaws.com` |
| `EC2_INSTANCE_ID`       | `i-0abc123def456789`                           |
| `EC2_PUBLIC_IP`         | `12.34.56.78`                                  |

### 2. Workflow File

The workflow file is already created at `.github/workflows/deploy.yml`. It will:

1. **Test** - Validate code loads correctly
2. **Build** - Build Docker image and push to ECR
3. **Deploy** - SSH to EC2 and pull new image
4. **Verify** - Health check the deployment

### 3. Enable SSM for EC2 (Required for CI/CD)

```bash
# Attach SSM role to EC2 instance
aws iam create-role --role-name EC2-SSM-Role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

aws iam attach-role-policy --role-name EC2-SSM-Role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Create instance profile
aws iam create-instance-profile --instance-profile-name EC2-SSM-Profile
aws iam add-role-to-instance-profile \
    --instance-profile-name EC2-SSM-Profile \
    --role-name EC2-SSM-Role

# Associate with EC2 instance
aws ec2 associate-iam-instance-profile \
    --instance-id YOUR_INSTANCE_ID \
    --iam-instance-profile Name=EC2-SSM-Profile
```

---

## 📊 Monitoring & Logging

### 1. CloudWatch Logs

```bash
# Install CloudWatch agent on EC2
sudo yum install amazon-cloudwatch-agent -y

# Configure agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### 2. Docker Logs

```bash
# View real-time logs
docker compose logs -f

# View specific service logs
docker compose logs -f api
docker compose logs -f nginx

# Limit log output
docker compose logs --tail=100 api
```

### 3. Set Up Log Rotation

```bash
# Create docker log rotation config
sudo cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# Restart Docker
sudo systemctl restart docker
```

### 4. Basic Monitoring Script

```bash
cat > ~/monitor.sh << 'EOF'
#!/bin/bash
echo "=== ASL API Status ==="
echo ""
echo "Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Health Check:"
curl -s http://localhost/health | jq
echo ""
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
EOF

chmod +x ~/monitor.sh
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker compose logs api

# Check if models exist
docker compose exec api ls -la /app/models/

# Verify image
docker images
```

#### 2. WebSocket Connection Fails

```bash
# Check nginx config
docker compose exec nginx nginx -t

# Verify WebSocket headers
curl -v -H "Upgrade: websocket" -H "Connection: Upgrade" http://localhost/ws/recognize
```

#### 3. Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Increase swap (if needed)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. ECR Login Issues

```bash
# Re-authenticate
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

# Check credentials
aws sts get-caller-identity
```

#### 5. SSL Certificate Issues

```bash
# Test SSL
openssl s_client -connect api.yourdomain.com:443

# Check certificate expiry
openssl x509 -enddate -noout -in ~/asl-api/nginx/ssl/fullchain.pem
```

### Useful Commands

```bash
# Restart all services
docker compose restart

# Force rebuild
docker compose up -d --force-recreate

# Remove all and start fresh
docker compose down -v
docker system prune -a
docker compose up -d

# Check resource usage
htop
docker stats

# View network
docker network ls
docker network inspect asl-api_asl-network
```

---

## 💰 Cost Estimation

### EC2 Costs (ap-southeast-1)

| Instance Type  | vCPU | RAM | On-Demand/Month | Notes               |
| -------------- | ---- | --- | --------------- | ------------------- |
| t3.micro       | 2    | 1GB | ~$8             | ❌ Too small for ML |
| t3.small       | 2    | 2GB | ~$15            | ❌ Risky for ML     |
| c7i.flex-large | 2    | 4GB | ~$35            | ✅ Workable         |
| m7i-flex.large | 2    | 8GB | ~$45            | ✅ **Recommended**  |

### Additional Costs

| Service             | Estimated Cost     |
| ------------------- | ------------------ |
| ECR Storage         | ~$0.10/GB/month    |
| Data Transfer (out) | ~$0.09/GB          |
| Elastic IP          | Free (if attached) |
| Route 53            | ~$0.50/hosted zone |
| SSL (Let's Encrypt) | Free               |

### Recommended for Your Use Case

**Recommended:** m7i-flex.large (~$45/month) - Best for TensorFlow + MediaPipe
**Budget Option:** c7i.flex-large (~$35/month) - Works but monitor memory usage

---

## 🔒 Security Best Practices

### 1. Network Security

- [ ] Use Security Groups to restrict SSH access to your IP only
- [ ] Keep API port (8000) internal only
- [ ] Enable VPC Flow Logs

### 2. Instance Security

```bash
# Keep system updated
sudo dnf update -y  # Amazon Linux
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Enable automatic security updates
sudo dnf install dnf-automatic -y
sudo systemctl enable --now dnf-automatic.timer
```

### 3. Docker Security

- [ ] Run containers as non-root user (already configured in Dockerfile)
- [ ] Use read-only file systems where possible
- [ ] Limit container resources
- [ ] Scan images for vulnerabilities

### 4. API Security

Update your `main.py` CORS configuration:

```python
# In production, specify exact origins
ALLOWED_ORIGINS = [
    "https://your-vercel-app.vercel.app",
    "https://yourdomain.com"
]
```

### 5. Secrets Management

- [ ] Use AWS Secrets Manager for sensitive data
- [ ] Never commit secrets to Git
- [ ] Rotate access keys regularly

---

## 🔗 Connect from Next.js (Vercel)

### Important: Use API Proxy Pattern

Since Vercel uses HTTPS and EC2 uses HTTP, browsers will block direct calls (mixed content).
**Solution:** Create an API route in Next.js that proxies requests to EC2 server-side.

```
Browser → Vercel API (HTTPS) → EC2 (HTTP) → Response
       ✅ No mixed content issues!
```

### 1. Environment Variables

**In Vercel Dashboard** (Settings → Environment Variables):

| Name         | Value                  |
| ------------ | ---------------------- |
| `ML_API_URL` | `http://54.169.39.128` |

**In `.env.local`** (for local development):

```env
# Server-side API URL (used by API proxy route)
ML_API_URL=http://54.169.39.128
```

> **Note:** This is a server-side variable (no `NEXT_PUBLIC_` prefix), so it's not exposed to the browser.

### 2. Create API Proxy Route

Create `src/app/api/ml/predict/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";

// EC2 ML API URL (server-side, no CORS/mixed content issues)
const ML_API_URL = process.env.ML_API_URL || "http://54.169.39.128";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();

    // Forward the request to EC2 ML API
    const response = await fetch(`${ML_API_URL}/predict-image/`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("ML API error:", errorText);
      return NextResponse.json(
        { error: "ML API request failed", details: errorText },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error calling ML API:", error);
    return NextResponse.json(
      { error: "Failed to connect to ML API" },
      { status: 500 }
    );
  }
}
```

### 3. Call the Proxy from Frontend

In your component (e.g., `CameraCapture.tsx`):

```typescript
// Call Vercel API proxy to avoid mixed content issues
const res = await fetch("/api/ml/predict", {
  method: "POST",
  body: formData,
});

const data = await res.json();
if (data.success && data.label) {
  setPrediction(`${data.label} (${(data.confidence * 100).toFixed(1)}%)`);
}
```

### Why This Works

| Connection       | Protocol | Status                                        |
| ---------------- | -------- | --------------------------------------------- |
| Browser → Vercel | HTTPS    | ✅ Secure                                     |
| Vercel → EC2     | HTTP     | ✅ Server-to-server (no browser restrictions) |

**No SSL certificate needed on EC2!**

```

---

## ✅ Deployment Checklist

### AWS Setup
- [ ] AWS Account created and IAM user configured
- [ ] ECR repository created
- [ ] EC2 instance launched with correct instance type (m7i-flex.large recommended)
- [ ] Security groups configured (22, 80, 443)
- [ ] Docker and Docker Compose installed on EC2
- [ ] AWS CLI configured on EC2
- [ ] Application deployed and running
- [ ] Health check passing (`/health` returns 200)

### Vercel Integration (API Proxy Pattern)
- [ ] Created `src/app/api/ml/predict/route.ts` proxy route
- [ ] Updated frontend to call `/api/ml/predict`
- [ ] Added `ML_API_URL` environment variable in Vercel
- [ ] Deployed and tested on phone/other devices

### Optional
- [ ] Domain configured
- [ ] SSL certificate installed
- [ ] GitHub Actions CI/CD configured
- [ ] Monitoring and logging set up

---

## 📚 Additional Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Nginx WebSocket Proxying](https://nginx.org/en/docs/http/websocket.html)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

---

## 🆘 Need Help?

If you encounter issues:

1. Check Docker logs: `docker compose logs -f`
2. Verify health endpoint: `curl http://localhost/health`
3. Review security group settings
4. Check EC2 instance metrics in AWS Console

---

**Good luck with your deployment! 🎉**
```
