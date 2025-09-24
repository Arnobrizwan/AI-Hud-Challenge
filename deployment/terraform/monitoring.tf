# Monitoring, logging, and alerting configuration

# Enable required APIs
resource "google_project_service" "monitoring_api" {
  count = var.enable_monitoring ? 1 : 0
  
  project = var.project_id
  service = "monitoring.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

resource "google_project_service" "logging_api" {
  count = var.enable_monitoring ? 1 : 0
  
  project = var.project_id
  service = "logging.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# Log sink for structured logging
resource "google_logging_project_sink" "app_log_sink" {
  count = var.enable_monitoring ? 1 : 0
  
  name        = "${local.service_name}-log-sink"
  project     = var.project_id
  destination = "storage.googleapis.com/${google_storage_bucket.log_bucket[0].name}"

  filter = <<-EOT
    resource.type="cloud_run_revision"
    resource.labels.service_name="${local.service_name}"
    severity >= "INFO"
  EOT

  unique_writer_identity = true

  depends_on = [
    google_project_service.logging_api,
    google_storage_bucket.log_bucket
  ]
}

# Storage bucket for logs
resource "google_storage_bucket" "log_bucket" {
  count = var.enable_monitoring ? 1 : 0
  
  name     = "${var.project_id}-${local.service_name}-logs"
  project  = var.project_id
  location = var.region

  uniform_bucket_level_access = true
  
  retention_policy {
    retention_period = var.log_retention_days * 24 * 60 * 60 # Convert days to seconds
  }

  lifecycle_rule {
    condition {
      age = var.log_retention_days
    }
    action {
      type = "Delete"
    }
  }

  labels = local.labels

  depends_on = [google_project_service.logging_api]
}

# IAM binding for log sink
resource "google_storage_bucket_iam_member" "log_sink_writer" {
  count = var.enable_monitoring ? 1 : 0
  
  bucket = google_storage_bucket.log_bucket[0].name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.app_log_sink[0].writer_identity

  depends_on = [
    google_storage_bucket.log_bucket,
    google_logging_project_sink.app_log_sink
  ]
}

# Cloud Monitoring alert policies
resource "google_monitoring_alert_policy" "high_error_rate" {
  count = var.enable_monitoring ? 1 : 0
  
  display_name = "${local.service_name} - High Error Rate"
  project      = var.project_id

  conditions {
    display_name = "High 5xx error rate"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" resource.label.service_name=\"${local.service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.05 # 5% error rate

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
        group_by_fields = ["resource.label.service_name"]
      }
    }
  }

  alert_strategy {
    auto_close = "604800s" # 7 days
  }

  combiner = "OR"
  enabled  = true

  notification_channels = var.notification_channels

  depends_on = [google_project_service.monitoring_api]
}

resource "google_monitoring_alert_policy" "high_latency" {
  count = var.enable_monitoring ? 1 : 0
  
  display_name = "${local.service_name} - High Latency"
  project      = var.project_id

  conditions {
    display_name = "High request latency"
    
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" resource.label.service_name=\"${local.service_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 1.0 # 1 second

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_95"
        cross_series_reducer = "REDUCE_MEAN"
        group_by_fields      = ["resource.label.service_name"]
      }
    }
  }

  alert_strategy {
    auto_close = "604800s" # 7 days
  }

  combiner = "OR"
  enabled  = true

  notification_channels = var.notification_channels

  depends_on = [google_project_service.monitoring_api]
}

resource "google_monitoring_alert_policy" "redis_high_memory" {
  count = var.enable_monitoring ? 1 : 0
  
  display_name = "${local.service_name} - Redis High Memory Usage"
  project      = var.project_id

  conditions {
    display_name = "Redis memory usage above 80%"
    
    condition_threshold {
      filter          = "resource.type=\"redis_instance\" resource.label.instance_id=\"${google_redis_instance.redis_cache.id}\""
      duration        = "300s"
      comparison      = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  alert_strategy {
    auto_close = "604800s" # 7 days
  }

  combiner = "OR"
  enabled  = true

  notification_channels = var.notification_channels

  depends_on = [google_project_service.monitoring_api]
}

# Custom dashboard
resource "google_monitoring_dashboard" "service_dashboard" {
  count = var.enable_monitoring ? 1 : 0
  
  project        = var.project_id
  dashboard_json = jsonencode({
    displayName = "${local.service_name} Dashboard"
    mosaicLayout = {
      tiles = [
        {
          width = 6
          height = 4
          widget = {
            title = "Request Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" resource.label.service_name=\"${local.service_name}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
              }]
            }
          }
        },
        {
          width = 6
          height = 4
          xPos = 6
          widget = {
            title = "Error Rate"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" resource.label.service_name=\"${local.service_name}\" metric.label.response_code_class=\"5xx\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_RATE"
                      crossSeriesReducer = "REDUCE_SUM"
                    }
                  }
                }
              }]
            }
          }
        },
        {
          width = 6
          height = 4
          yPos = 4
          widget = {
            title = "Response Latency (95th percentile)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloud_run_revision\" resource.label.service_name=\"${local.service_name}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_95"
                      crossSeriesReducer = "REDUCE_MEAN"
                    }
                  }
                }
              }]
            }
          }
        },
        {
          width = 6
          height = 4
          xPos = 6
          yPos = 4
          widget = {
            title = "Redis Memory Usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"redis_instance\" resource.label.instance_id=\"${google_redis_instance.redis_cache.id}\""
                    aggregation = {
                      alignmentPeriod = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
              }]
            }
          }
        }
      ]
    }
  })

  depends_on = [google_project_service.monitoring_api]
}

# Uptime check
resource "google_monitoring_uptime_check_config" "service_uptime_check" {
  count = var.enable_monitoring ? 1 : 0
  
  display_name = "${local.service_name} Uptime Check"
  project      = var.project_id

  timeout = "10s"
  period  = "60s"

  http_check {
    path         = "/health/live"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "uptime_url"
    labels = {
      project_id = var.project_id
      host       = google_cloud_run_v2_service.foundations_guards_service.uri
    }
  }

  content_matchers {
    content = "healthy"
    matcher = "CONTAINS_STRING"
  }

  checker_type = "STATIC_IP_CHECKERS"

  depends_on = [
    google_project_service.monitoring_api,
    google_cloud_run_v2_service.foundations_guards_service
  ]
}
