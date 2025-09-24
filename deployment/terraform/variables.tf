# Terraform variables for GCP deployment

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "foundations-guards-service"
}

variable "container_image" {
  description = "Container image URL"
  type        = string
}

# Cloud Run Configuration
variable "cpu_limit" {
  description = "CPU limit for Cloud Run service"
  type        = string
  default     = "1000m"
}

variable "memory_limit" {
  description = "Memory limit for Cloud Run service"
  type        = string
  default     = "512Mi"
}

variable "max_concurrency" {
  description = "Maximum concurrent requests per container"
  type        = number
  default     = 1000
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 100
}

variable "timeout_seconds" {
  description = "Request timeout in seconds"
  type        = number
  default     = 300
}

# Redis Configuration
variable "redis_memory_size_gb" {
  description = "Redis instance memory size in GB"
  type        = number
  default     = 1
}

variable "redis_tier" {
  description = "Redis service tier"
  type        = string
  default     = "STANDARD_HA"
  
  validation {
    condition     = contains(["BASIC", "STANDARD_HA"], var.redis_tier)
    error_message = "Redis tier must be either BASIC or STANDARD_HA."
  }
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_7_0"
}

# VPC Configuration
variable "vpc_name" {
  description = "VPC network name"
  type        = string
  default     = "foundations-guards-vpc"
}

variable "subnet_cidr" {
  description = "Subnet CIDR range"
  type        = string
  default     = "10.0.0.0/24"
}

# Firestore Configuration
variable "enable_firestore" {
  description = "Enable Firestore database"
  type        = bool
  default     = false
}

variable "firestore_location" {
  description = "Firestore location"
  type        = string
  default     = "us-central"
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable Cloud Monitoring and Logging"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "allowed_ingress_cidrs" {
  description = "CIDR blocks allowed to access the service"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_iap" {
  description = "Enable Identity-Aware Proxy"
  type        = bool
  default     = false
}

# Firebase Configuration
variable "firebase_project_id" {
  description = "Firebase project ID (defaults to main project)"
  type        = string
  default     = ""
}

# Environment Variables
variable "environment_variables" {
  description = "Environment variables for the service"
  type        = map(string)
  default     = {}
}

variable "secret_environment_variables" {
  description = "Secret environment variables stored in Secret Manager"
  type        = map(string)
  default     = {}
}

# Backup Configuration
variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

# Additional Variables
variable "custom_domain" {
  description = "Custom domain for the service (optional)"
  type        = string
  default     = ""
}

variable "notification_channels" {
  description = "Notification channels for alerting"
  type        = list(string)
  default     = []
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity for GKE integration"
  type        = bool
  default     = false
}

variable "create_service_account_key" {
  description = "Create service account key for local development"
  type        = bool
  default     = false
}

variable "enable_cloud_build" {
  description = "Enable Cloud Build service account and permissions"
  type        = bool
  default     = true
}
