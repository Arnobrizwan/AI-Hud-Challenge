# Cloud Run service configuration

# Enable required APIs
resource "google_project_service" "cloud_run_api" {
  project = var.project_id
  service = "run.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

resource "google_project_service" "cloud_resource_manager_api" {
  project = var.project_id
  service = "cloudresourcemanager.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Cloud Run service
resource "google_cloud_run_v2_service" "foundations_guards_service" {
  name     = local.service_name
  location = var.region
  project  = var.project_id

  depends_on = [
    google_project_service.cloud_run_api,
    google_redis_instance.redis_cache
  ]

  template {
    # Scaling configuration
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    # VPC configuration
    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    # Container configuration
    containers {
      name  = "foundations-guards-container"
      image = var.container_image

      # Resource limits
      resources {
        limits = {
          cpu    = local.cloud_run_config.cpu_limit
          memory = local.cloud_run_config.memory_limit
        }
        cpu_idle                = true
        startup_cpu_boost       = true
      }

      # Ports
      ports {
        name           = "http1"
        container_port = 8080
      }

      # Environment variables
      dynamic "env" {
        for_each = merge(
          {
            ENVIRONMENT                    = var.environment
            GCP_PROJECT_ID                = var.project_id
            GCP_REGION                    = var.region
            CLOUD_RUN_SERVICE_NAME        = local.service_name
            REDIS_URL                     = "redis://${google_redis_instance.redis_cache.host}:${google_redis_instance.redis_cache.port}/0"
            FIREBASE_PROJECT_ID           = var.firebase_project_id != "" ? var.firebase_project_id : var.project_id
            ENABLE_METRICS               = "true"
            ENABLE_SECURITY_HEADERS      = "true"
            HEALTH_CHECK_DEPENDENCIES    = "true"
            LOG_LEVEL                    = var.environment == "prod" ? "INFO" : "DEBUG"
            LOG_FORMAT                   = "json"
          },
          var.environment_variables
        )
        content {
          name  = env.key
          value = env.value
        }
      }

      # Secret environment variables
      dynamic "env" {
        for_each = var.secret_environment_variables
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.app_secrets[env.key].secret_id
              version = "latest"
            }
          }
        }
      }

      # Startup probe
      startup_probe {
        http_get {
          path = "/health/live"
          port = 8080
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      # Liveness probe
      liveness_probe {
        http_get {
          path = "/health/live"
          port = 8080
        }
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 30
        failure_threshold     = 3
      }
    }

    # Service account
    service_account = google_service_account.cloud_run_sa.email

    # Request timeout
    timeout = "${var.timeout_seconds}s"

    # Execution environment
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    # Session affinity (for sticky sessions if needed)
    session_affinity = false

    # Maximum requests per container
    max_instance_request_concurrency = local.cloud_run_config.concurrency
  }

  # Traffic configuration
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }

  labels = local.labels

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      template[0].annotations
    ]
  }
}

# IAM policy for Cloud Run service
resource "google_cloud_run_service_iam_policy" "noauth" {
  location = google_cloud_run_v2_service.foundations_guards_service.location
  project  = google_cloud_run_v2_service.foundations_guards_service.project
  service  = google_cloud_run_v2_service.foundations_guards_service.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

# IAM policy data for public access
data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

# Custom domain mapping (optional)
resource "google_cloud_run_domain_mapping" "domain_mapping" {
  count    = var.custom_domain != "" ? 1 : 0
  location = var.region
  name     = var.custom_domain
  project  = var.project_id

  metadata {
    namespace = var.project_id
  }

  spec {
    route_name = google_cloud_run_v2_service.foundations_guards_service.name
  }

  depends_on = [google_cloud_run_v2_service.foundations_guards_service]
}
