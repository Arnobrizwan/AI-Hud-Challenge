# Secret Manager configuration

# Enable Secret Manager API
resource "google_project_service" "secretmanager_api" {
  project = var.project_id
  service = "secretmanager.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Create secrets for sensitive configuration
resource "google_secret_manager_secret" "app_secrets" {
  for_each = var.secret_environment_variables
  
  secret_id = "${local.service_name}-${each.key}"
  project   = var.project_id

  labels = local.labels

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [google_project_service.secretmanager_api]
}

# Create secret versions
resource "google_secret_manager_secret_version" "app_secret_versions" {
  for_each = var.secret_environment_variables
  
  secret      = google_secret_manager_secret.app_secrets[each.key].id
  secret_data = each.value
}

# Default secrets that should be created (empty by default)
locals {
  default_secrets = {
    jwt-secret-key        = ""
    firebase-credentials  = ""
    database-password     = ""
  }
}

# Create default secrets if not provided
resource "google_secret_manager_secret" "default_secrets" {
  for_each = {
    for key, value in local.default_secrets : key => value
    if !contains(keys(var.secret_environment_variables), key)
  }
  
  secret_id = "${local.service_name}-${each.key}"
  project   = var.project_id

  labels = merge(local.labels, {
    type = "default"
  })

  replication {
    automatic = true
  }

  depends_on = [google_project_service.secretmanager_api]
}

# IAM binding for Cloud Run service account to access secrets
resource "google_secret_manager_secret_iam_binding" "secret_access" {
  for_each = google_secret_manager_secret.app_secrets
  
  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"

  members = [
    "serviceAccount:${google_service_account.cloud_run_sa.email}",
  ]

  depends_on = [
    google_service_account.cloud_run_sa,
    google_secret_manager_secret.app_secrets
  ]
}

# IAM binding for default secrets
resource "google_secret_manager_secret_iam_binding" "default_secret_access" {
  for_each = google_secret_manager_secret.default_secrets
  
  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"

  members = [
    "serviceAccount:${google_service_account.cloud_run_sa.email}",
  ]

  depends_on = [
    google_service_account.cloud_run_sa,
    google_secret_manager_secret.default_secrets
  ]
}
