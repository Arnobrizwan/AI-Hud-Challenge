# AWS Deployment Guide for AI News Hub

## 🚀 **AWS App Runner** (Recommended - Easiest)

### Why AWS App Runner?
- ✅ **Free tier**: 1000 vCPU minutes/month
- ✅ **Automatic scaling** - scales to zero when not used
- ✅ **Docker native** - perfect for your setup
- ✅ **Managed service** - no server management
- ✅ **HTTPS included** - secure by default

### Deploy Steps:

#### Option 1: AWS Console (Easiest)
1. Go to [AWS App Runner Console](https://console.aws.amazon.com/apprunner)
2. Click "Create service"
3. Choose "Source" → "Container registry" → "Public"
4. Enter your Docker image URL (after building)
5. Configure service settings
6. Deploy!

#### Option 2: AWS CLI
```bash
# Build and push to ECR
aws ecr create-repository --repository-name ai-news-hub
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker build -t ai-news-hub .
docker tag ai-news-hub:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-news-hub:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-news-hub:latest

# Deploy with App Runner
aws apprunner create-service --service-name ai-news-hub --source-configuration '{
  "ImageRepository": {
    "ImageIdentifier": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-news-hub:latest",
    "ImageConfiguration": {
      "Port": "8000",
      "RuntimeEnvironmentVariables": {
        "PORT": "8000"
      }
    },
    "ImageRepositoryType": "ECR"
  }
}'
```

### Cost: **FREE** (within limits) 🆓

---

## 🐳 **AWS Elastic Beanstalk** (Alternative)

### Why Elastic Beanstalk?
- ✅ **Free tier**: 750 hours/month
- ✅ **Easy deployment** - just upload code
- ✅ **Auto-scaling** - handles traffic spikes
- ✅ **Load balancing** - built-in

### Deploy Steps:
1. Go to [AWS Elastic Beanstalk Console](https://console.aws.amazon.com/elasticbeanstalk)
2. Click "Create application"
3. Choose "Docker" platform
4. Upload your code as ZIP file
5. Deploy!

### Cost: **FREE** (within limits) 🆓

---

## ⚡ **AWS Lambda** (Serverless - Advanced)

### Why Lambda?
- ✅ **Free tier**: 1 million requests/month
- ✅ **Pay per use** - only pay for requests
- ✅ **Auto-scaling** - handles any load
- ✅ **Serverless** - no server management

### Deploy Steps:
1. Install AWS SAM CLI
2. Create `template.yaml`:
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Resources:
  AINewsHubAPI:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: app.handler
      Runtime: python3.11
      Events:
        Api:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
```

3. Deploy: `sam build && sam deploy --guided`

### Cost: **FREE** (within limits) 🆓

---

## 🖥️ **AWS EC2** (Full Control)

### Why EC2?
- ✅ **Free tier**: 750 hours/month of t2.micro
- ✅ **Full control** - install anything
- ✅ **Persistent storage** - data stays
- ✅ **Custom configuration** - optimize as needed

### Deploy Steps:
1. Launch t2.micro instance (free tier)
2. Install Docker: `sudo yum install docker`
3. Clone your repo: `git clone https://github.com/Arnobrizwan/AI-Hud-Challenge.git`
4. Build and run: `docker build -t ai-news-hub . && docker run -p 80:8000 ai-news-hub`

### Cost: **FREE** (within limits) 🆓

---

## 📊 **Size Optimization for AWS**

Your project is 331MB. To optimize:

1. **Use .dockerignore**:
```dockerignore
__pycache__/
*.pyc
.git/
tests/
*.md
.env
node_modules/
```

2. **Multi-stage build** (already done)

3. **Remove unnecessary files**:
```bash
# Remove test files
find . -name "tests" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

---

## 🎯 **My Recommendation**

**Start with AWS App Runner** because:
- Easiest AWS deployment
- Handles your project size
- Great free tier
- Auto-scaling
- No server management

**Backup plan**: AWS Elastic Beanstalk (more traditional but reliable)

---

## 🚀 **Quick Start with AWS App Runner**

1. **Build your Docker image**:
   ```bash
   docker build -f Dockerfile.aws -t ai-news-hub .
   ```

2. **Push to ECR** (or use Docker Hub):
   ```bash
   # Tag for ECR
   docker tag ai-news-hub:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-news-hub:latest
   
   # Push to ECR
   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/ai-news-hub:latest
   ```

3. **Deploy with App Runner**:
   - Go to AWS App Runner Console
   - Create service
   - Use your ECR image
   - Deploy!

**Total time**: 15 minutes
**Cost**: $0 (within free tier)
**Result**: Live AI News Hub on AWS! 🚀

---

## 💡 **Pro Tips**

- **Use ECR** for private images
- **Use Docker Hub** for public images (easier)
- **Set up CloudWatch** for monitoring
- **Use Route 53** for custom domain
- **Enable CloudFront** for global CDN
