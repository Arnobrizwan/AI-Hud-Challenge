# Redis instance for rate limiting and caching

# Enable Redis API
resource "google_project_service" "redis_api" {
  project = var.project_id
  service = "redis.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Redis instance
resource "google_redis_instance" "redis_cache" {
  name           = "${local.service_name}-redis"
  project        = var.project_id
  region         = var.region
  memory_size_gb = var.redis_memory_size_gb
  tier           = var.redis_tier
  redis_version  = var.redis_version

  # Network configuration
  authorized_network = google_compute_network.vpc_network.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  # Redis configuration
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout         = "300"
  }

  # Maintenance policy
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }

  # Auth and security
  auth_enabled               = true
  transit_encryption_mode   = "SERVER_AUTHENTICATION"
  customer_managed_key      = var.environment == "prod" ? google_kms_crypto_key.redis_key[0].id : null

  labels = merge(local.labels, {
    component = "cache"
  })

  depends_on = [
    google_project_service.redis_api,
    google_compute_network.vpc_network,
    google_service_networking_connection.private_vpc_connection
  ]
}

# KMS key for Redis encryption (production only)
resource "google_kms_key_ring" "redis_keyring" {
  count    = var.environment == "prod" ? 1 : 0
  name     = "${local.service_name}-redis-keyring"
  location = var.region
  project  = var.project_id
}

resource "google_kms_crypto_key" "redis_key" {
  count    = var.environment == "prod" ? 1 : 0
  name     = "${local.service_name}-redis-key"
  key_ring = google_kms_key_ring.redis_keyring[0].id

  rotation_period = "2592000s" # 30 days

  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Redis backup (for Standard tier)
resource "google_redis_instance" "redis_backup" {
  count = var.enable_backups && var.redis_tier == "STANDARD_HA" ? 1 : 0
  
  name           = "${local.service_name}-redis-backup"
  project        = var.project_id
  region         = var.region
  memory_size_gb = var.redis_memory_size_gb
  tier           = "BASIC"
  redis_version  = var.redis_version

  # Network configuration
  authorized_network = google_compute_network.vpc_network.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  labels = merge(local.labels, {
    component = "cache-backup"
  })

  depends_on = [
    google_redis_instance.redis_cache
  ]
}

# Firewall rule for Redis (if needed for debugging)
resource "google_compute_firewall" "redis_firewall" {
  count = var.environment != "prod" ? 1 : 0
  
  name    = "${local.service_name}-redis-firewall"
  network = google_compute_network.vpc_network.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["6379"]
  }

  source_ranges = [var.subnet_cidr]
  target_tags   = ["redis-access"]

  depends_on = [google_compute_network.vpc_network]
}
