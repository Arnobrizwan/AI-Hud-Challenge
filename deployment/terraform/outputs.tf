# Terraform outputs

output "service_url" {
  description = "URL of the Cloud Run service"
  value       = google_cloud_run_v2_service.foundations_guards_service.uri
}

output "service_name" {
  description = "Name of the Cloud Run service"
  value       = google_cloud_run_v2_service.foundations_guards_service.name
}

output "service_region" {
  description = "Region where the service is deployed"
  value       = google_cloud_run_v2_service.foundations_guards_service.location
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.redis_cache.host
  sensitive   = true
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.redis_cache.port
}

output "redis_auth_string" {
  description = "Redis authentication string"
  value       = google_redis_instance.redis_cache.auth_string
  sensitive   = true
}

output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc_network.name
}

output "vpc_network_id" {
  description = "ID of the VPC network"
  value       = google_compute_network.vpc_network.id
}

output "subnet_name" {
  description = "Name of the subnet"
  value       = google_compute_subnetwork.subnet.name
}

output "subnet_cidr" {
  description = "CIDR range of the subnet"
  value       = google_compute_subnetwork.subnet.ip_cidr_range
}

output "service_account_email" {
  description = "Email of the Cloud Run service account"
  value       = google_service_account.cloud_run_sa.email
}

output "service_account_id" {
  description = "ID of the Cloud Run service account"
  value       = google_service_account.cloud_run_sa.account_id
}

output "vpc_connector_name" {
  description = "Name of the VPC connector"
  value       = google_vpc_access_connector.connector.name
}

output "vpc_connector_id" {
  description = "ID of the VPC connector"
  value       = google_vpc_access_connector.connector.id
}

output "log_bucket_name" {
  description = "Name of the log storage bucket"
  value       = var.enable_monitoring ? google_storage_bucket.log_bucket[0].name : null
}

output "secret_names" {
  description = "Names of created secrets"
  value       = {
    for k, v in google_secret_manager_secret.app_secrets : k => v.secret_id
  }
}

output "monitoring_dashboard_url" {
  description = "URL of the monitoring dashboard"
  value       = var.enable_monitoring ? "https://console.cloud.google.com/monitoring/dashboards/custom/${google_monitoring_dashboard.service_dashboard[0].id}?project=${var.project_id}" : null
}

output "cloud_build_sa_email" {
  description = "Email of the Cloud Build service account"
  value       = var.enable_cloud_build ? google_service_account.cloud_build_sa[0].email : null
}

# Useful for CI/CD pipelines
output "deployment_info" {
  description = "Deployment information for CI/CD"
  value = {
    project_id      = var.project_id
    region         = var.region
    service_name   = local.service_name
    service_url    = google_cloud_run_v2_service.foundations_guards_service.uri
    container_port = 8080
    health_check   = "/health/live"
    metrics_path   = "/metrics"
  }
}

# Security information
output "security_info" {
  description = "Security configuration information"
  value = {
    vpc_network       = google_compute_network.vpc_network.name
    subnet           = google_compute_subnetwork.subnet.name
    service_account  = google_service_account.cloud_run_sa.email
    secrets_created  = length(google_secret_manager_secret.app_secrets)
    redis_encrypted  = var.environment == "prod" ? true : false
  }
  sensitive = false
}

# Resource identifiers for cleanup
output "resource_ids" {
  description = "Resource IDs for cleanup and management"
  value = {
    cloud_run_service = google_cloud_run_v2_service.foundations_guards_service.id
    redis_instance    = google_redis_instance.redis_cache.id
    vpc_network       = google_compute_network.vpc_network.id
    service_account   = google_service_account.cloud_run_sa.id
    vpc_connector     = google_vpc_access_connector.connector.id
  }
}
