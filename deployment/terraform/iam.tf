# IAM configuration for service accounts and permissions

# Enable IAM API
resource "google_project_service" "iam_api" {
  project = var.project_id
  service = "iam.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Cloud Run service account
resource "google_service_account" "cloud_run_sa" {
  account_id   = "${local.service_name}-sa"
  display_name = "Service Account for ${local.service_name}"
  description  = "Service account used by the Foundations & Guards Cloud Run service"
  project      = var.project_id

  depends_on = [google_project_service.iam_api]
}

# Basic IAM roles for Cloud Run service account
resource "google_project_iam_member" "cloud_run_sa_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/cloudtrace.agent",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"

  depends_on = [google_service_account.cloud_run_sa]
}

# Firebase Admin SDK permissions
resource "google_project_iam_member" "firebase_admin" {
  count = var.firebase_project_id != "" ? 1 : 0
  
  project = var.firebase_project_id != "" ? var.firebase_project_id : var.project_id
  role    = "roles/firebase.admin"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"

  depends_on = [google_service_account.cloud_run_sa]
}

# Redis access permissions
resource "google_project_iam_member" "redis_editor" {
  project = var.project_id
  role    = "roles/redis.editor"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"

  depends_on = [google_service_account.cloud_run_sa]
}

# Custom role for specific service permissions
resource "google_project_iam_custom_role" "foundations_guards_role" {
  role_id     = "foundations_guards_service_role"
  title       = "Foundations & Guards Service Role"
  description = "Custom role for Foundations & Guards service with minimal required permissions"
  project     = var.project_id

  permissions = [
    "redis.instances.get",
    "redis.instances.list",
    "monitoring.timeSeries.create",
    "logging.logEntries.create",
    "secretmanager.versions.access",
    "cloudtrace.traces.patch"
  ]

  depends_on = [google_project_service.iam_api]
}

# Assign custom role to service account
resource "google_project_iam_member" "custom_role_assignment" {
  project = var.project_id
  role    = google_project_iam_custom_role.foundations_guards_role.id
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"

  depends_on = [
    google_service_account.cloud_run_sa,
    google_project_iam_custom_role.foundations_guards_role
  ]
}

# Workload Identity (if using GKE in the future)
resource "google_service_account_iam_binding" "workload_identity" {
  count = var.enable_workload_identity ? 1 : 0
  
  service_account_id = google_service_account.cloud_run_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/${local.service_name}]",
  ]

  depends_on = [google_service_account.cloud_run_sa]
}

# Service account key for local development (optional)
resource "google_service_account_key" "cloud_run_sa_key" {
  count = var.create_service_account_key ? 1 : 0
  
  service_account_id = google_service_account.cloud_run_sa.name
  public_key_type    = "TYPE_X509_PEM_FILE"

  depends_on = [google_service_account.cloud_run_sa]
}

# Cloud Build service account for CI/CD
resource "google_service_account" "cloud_build_sa" {
  count = var.enable_cloud_build ? 1 : 0
  
  account_id   = "${local.service_name}-build-sa"
  display_name = "Cloud Build Service Account for ${local.service_name}"
  description  = "Service account used by Cloud Build for CI/CD"
  project      = var.project_id

  depends_on = [google_project_service.iam_api]
}

# Cloud Build service account permissions
resource "google_project_iam_member" "cloud_build_sa_roles" {
  for_each = var.enable_cloud_build ? toset([
    "roles/cloudbuild.builds.builder",
    "roles/run.developer",
    "roles/iam.serviceAccountUser",
    "roles/storage.objectViewer",
  ]) : toset([])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.cloud_build_sa[0].email}"

  depends_on = [google_service_account.cloud_build_sa]
}

# Allow Cloud Build to act as Cloud Run service account
resource "google_service_account_iam_binding" "cloud_build_sa_user" {
  count = var.enable_cloud_build ? 1 : 0
  
  service_account_id = google_service_account.cloud_run_sa.name
  role               = "roles/iam.serviceAccountUser"

  members = [
    "serviceAccount:${google_service_account.cloud_build_sa[0].email}",
  ]

  depends_on = [
    google_service_account.cloud_run_sa,
    google_service_account.cloud_build_sa
  ]
}
