# 🔧 Troubleshooting Guide

## 🚨 Build Issues

### **Problem: Build Stuck for 10+ Seconds**

**Root Causes:**
1. **Circular Dependencies**: Multiple services building from same directory
2. **Missing Health Endpoints**: Services don't respond to health checks
3. **Resource Conflicts**: Memory/CPU issues with heavy ML services
4. **Infinite Wait Loops**: No timeout in health check scripts

**Solutions:**

#### **Option 1: Use Fast Development Mode (Recommended)**
```bash
# Start only core services (3-5 minutes)
make dev-up-fast

# Check logs
make dev-logs-fast

# Stop when done
make dev-down-fast
```

#### **Option 2: Build Services Individually**
```bash
# Start infrastructure first
docker-compose -f docker-compose.dev-simple.yml up redis postgres -d

# Wait for them to be ready
sleep 30

# Start services one by one
docker-compose -f docker-compose.dev-simple.yml up foundations-guards -d
docker-compose -f docker-compose.dev-simple.yml up content-extraction -d
docker-compose -f docker-compose.dev-simple.yml up ingestion-normalization -d
docker-compose -f docker-compose.dev-simple.yml up api-gateway -d
```

#### **Option 3: Debug Specific Service**
```bash
# Check what's running
docker-compose -f docker-compose.dev-simple.yml ps

# Check logs for specific service
docker-compose -f docker-compose.dev-simple.yml logs foundations-guards

# Check if service is responding
curl http://localhost:8001/health
```

### **Problem: Out of Memory Errors**

**Solution:**
```bash
# Increase Docker memory limit in Docker Desktop
# Settings > Resources > Memory: 8GB+

# Or reduce service replicas
export WORKER_COUNT=1
make dev-up-fast
```

### **Problem: Port Conflicts**

**Solution:**
```bash
# Check what's using ports
lsof -i :8000-8016

# Kill conflicting processes
sudo kill -9 $(lsof -t -i:8000-8016)

# Or use different ports
export API_GATEWAY_PORT=9000
make dev-up-fast
```

### **Problem: Database Connection Issues**

**Solution:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Restart database
docker-compose -f docker-compose.dev-simple.yml restart postgres

# Check database logs
docker-compose -f docker-compose.dev-simple.yml logs postgres
```

## 🔍 Debugging Commands

### **Check Service Status**
```bash
# All services
docker-compose -f docker-compose.dev-simple.yml ps

# Specific service logs
docker-compose -f docker-compose.dev-simple.yml logs -f foundations-guards

# Health check
curl http://localhost:8001/health
```

### **Check Resource Usage**
```bash
# Docker resource usage
docker stats

# System resources
top
htop  # if installed
```

### **Clean Up and Restart**
```bash
# Stop everything
make dev-down-fast

# Clean Docker cache
docker system prune -f

# Restart
make dev-up-fast
```

## 📊 Performance Optimization

### **For Apple Silicon (M1/M2)**
```bash
# Use ARM64 platform
export DOCKER_DEFAULT_PLATFORM=linux/arm64
make dev-up-fast
```

### **For Intel Macs**
```bash
# Use AMD64 platform
export DOCKER_DEFAULT_PLATFORM=linux/amd64
make dev-up-fast
```

### **Reduce Memory Usage**
```bash
# Set environment variables
export WORKER_COUNT=1
export MAX_MEMORY=512m
export ENABLE_ML_SERVICES=false

make dev-up-fast
```

## 🚀 Quick Start (Recommended)

For fastest development experience:

```bash
# 1. Start core services only
make dev-up-fast

# 2. Wait for completion (3-5 minutes)
# 3. Test the API
curl http://localhost:8000/health

# 4. View logs if needed
make dev-logs-fast

# 5. Stop when done
make dev-down-fast
```

## 📞 Getting Help

If you're still having issues:

1. **Check the logs**: `make dev-logs-fast`
2. **Verify Docker is running**: `docker --version`
3. **Check available memory**: `docker system df`
4. **Try the simple setup**: Use `docker-compose.dev-simple.yml` directly
5. **Create an issue**: Include logs and system info

## ✅ Success Indicators

You'll know everything is working when you see:
- ✅ All services show "healthy" status
- ✅ API Gateway responds at http://localhost:8000
- ✅ Individual services respond at their ports
- ✅ No error messages in logs
- ✅ Build completes in under 5 minutes
