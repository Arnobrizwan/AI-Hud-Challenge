# Terraform configuration for GCP deployment
# This creates the infrastructure for the Foundations & Guards service

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Configure backend for state management
  # Uncomment and configure for production
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "foundations-guards-service"
  # }
}

# Configure providers
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Local variables
locals {
  service_name = var.service_name
  labels = {
    environment = var.environment
    service     = "foundations-guards"
    managed-by  = "terraform"
  }
  
  # Cloud Run configuration
  cloud_run_config = {
    cpu_limit      = var.cpu_limit
    memory_limit   = var.memory_limit
    concurrency    = var.max_concurrency
    min_instances  = var.min_instances
    max_instances  = var.max_instances
  }
}
