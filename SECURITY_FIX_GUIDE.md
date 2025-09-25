# 🚨 URGENT: Security Fix Guide

## ⚠️ Service Account Key Exposed

Google Cloud detected a service account key that was accidentally committed to the public repository. **Immediate action required!**

## 🔧 Immediate Steps to Fix

### 1. ✅ Remove Exposed Key (COMPLETED)
- [x] Removed `config/service-account-key.json` from repository
- [x] Added comprehensive `.gitignore` to prevent future exposure
- [x] Updated documentation with security best practices

### 2. 🔄 Rotate the Compromised Credentials

#### **Step A: Disable the Exposed Key**
```bash
# List service account keys
gcloud iam service-accounts keys list \
  --iam-account=news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com

# Delete the compromised key (ID: e86761e4a72dcd73be0b99dd84dbc1b0c25ad553)
gcloud iam service-accounts keys delete e86761e4a72dcd73be0b99dd84dbc1b0c25ad553 \
  --iam-account=news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com
```

#### **Step B: Create New Service Account Key**
```bash
# Create a new key for the same service account
gcloud iam service-accounts keys create new-key.json \
  --iam-account=news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com
```

#### **Step C: Update GitHub Secrets**
1. Go to your GitHub repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Update the `GCP_SA_KEY` secret with the content of `new-key.json`
4. Delete the `new-key.json` file from your local machine

### 3. 🔍 Audit and Monitor

#### **Check for Unauthorized Access**
```bash
# Check recent activity
gcloud logging read "resource.type=service_account" \
  --limit=50 \
  --format="table(timestamp,resource.labels.service_account_id,severity,textPayload)"

# Check for any suspicious API calls
gcloud logging read "protoPayload.authenticationInfo.principalEmail=news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com" \
  --limit=100 \
  --format="table(timestamp,protoPayload.methodName,protoPayload.resourceName)"
```

#### **Review IAM Permissions**
```bash
# Check current permissions
gcloud projects get-iam-policy news-hub-prod-2024 \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com"
```

### 4. 🛡️ Implement Security Best Practices

#### **Use Workload Identity (Recommended)**
Instead of service account keys, use Workload Identity:

```bash
# Enable Workload Identity
gcloud iam service-accounts add-iam-policy-binding \
  news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:news-hub-prod-2024.svc.id.goog[default/news-hub-pipeline]"
```

#### **Update GitHub Actions Workflow**
Update `.github/workflows/deploy-production.yml` to use Workload Identity:

```yaml
- name: 🔐 Setup Google Cloud
  uses: google-github-actions/setup-gcloud@v1
  with:
    project_id: ${{ env.PROJECT_ID }}
    workload_identity_provider: projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID
    service_account: news-hub-deployer@news-hub-prod-2024.iam.gserviceaccount.com
```

### 5. 📋 Verification Checklist

- [ ] Exposed key removed from repository
- [ ] Compromised key deleted from Google Cloud
- [ ] New service account key created
- [ ] GitHub Secrets updated
- [ ] Local key files deleted
- [ ] Repository access audited
- [ ] IAM permissions reviewed
- [ ] Monitoring alerts configured

### 6. 🚨 Prevention Measures

#### **Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Add to .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-added-large-files
      - id: check-json
      - id: check-yaml
      - id: detect-private-key
```

#### **GitHub Actions Security**
```yaml
# Add to workflow
permissions:
  contents: read
  id-token: write  # For Workload Identity
```

## 📞 Support

If you need help with any of these steps:
1. Check Google Cloud Console for detailed logs
2. Review the [Google Cloud Security Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)
3. Contact Google Cloud Support if needed

## ✅ Status

- [x] **Immediate threat mitigated** - Key removed from repository
- [x] **Prevention measures in place** - .gitignore updated
- [x] **Documentation updated** - Security guide created
- [ ] **Credentials rotated** - Action required
- [ ] **Monitoring configured** - Action required

**Next Priority**: Rotate the compromised credentials immediately!
