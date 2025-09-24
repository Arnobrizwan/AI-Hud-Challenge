# VPC and networking configuration

# Enable Compute API
resource "google_project_service" "compute_api" {
  project = var.project_id
  service = "compute.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

resource "google_project_service" "servicenetworking_api" {
  project = var.project_id
  service = "servicenetworking.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

resource "google_project_service" "vpcaccess_api" {
  project = var.project_id
  service = "vpcaccess.googleapis.com"
  
  disable_dependent_services = true
  disable_on_destroy         = false
}

# VPC Network
resource "google_compute_network" "vpc_network" {
  name                    = var.vpc_name
  project                 = var.project_id
  auto_create_subnetworks = false
  mtu                     = 1460

  depends_on = [google_project_service.compute_api]
}

# Subnet for the VPC
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.vpc_name}-subnet"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.vpc_network.id
  ip_cidr_range = var.subnet_cidr

  # Enable private Google access
  private_ip_google_access = true

  # Secondary IP ranges (if needed)
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }

  depends_on = [google_compute_network.vpc_network]
}

# VPC Access Connector for Cloud Run
resource "google_vpc_access_connector" "connector" {
  name          = "${local.service_name}-connector"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.vpc_network.id
  ip_cidr_range = "10.8.0.0/28"

  min_throughput = 200
  max_throughput = 1000

  depends_on = [
    google_project_service.vpcaccess_api,
    google_compute_network.vpc_network
  ]
}

# Private service connection for Redis
resource "google_compute_global_address" "private_ip_address" {
  name          = "${local.service_name}-private-ip"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id

  depends_on = [google_compute_network.vpc_network]
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]

  depends_on = [
    google_project_service.servicenetworking_api,
    google_compute_global_address.private_ip_address
  ]
}

# Cloud NAT for outbound internet access (if needed)
resource "google_compute_router" "router" {
  name    = "${local.service_name}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.vpc_network.id

  depends_on = [google_compute_network.vpc_network]
}

resource "google_compute_router_nat" "nat" {
  name                               = "${local.service_name}-nat"
  project                           = var.project_id
  router                            = google_compute_router.router.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }

  depends_on = [google_compute_router.router]
}

# Firewall rules
resource "google_compute_firewall" "allow_health_check" {
  name    = "${local.service_name}-allow-health-check"
  project = var.project_id
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  # Google Cloud health check ranges
  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]

  target_tags = ["cloud-run-service"]

  depends_on = [google_compute_network.vpc_network]
}

resource "google_compute_firewall" "allow_ssh" {
  count = var.environment != "prod" ? 1 : 0
  
  name    = "${local.service_name}-allow-ssh"
  project = var.project_id
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]  # IAP range
  target_tags   = ["allow-ssh"]

  depends_on = [google_compute_network.vpc_network]
}

# Firewall rule for load balancer health checks
resource "google_compute_firewall" "allow_lb_health_check" {
  name    = "${local.service_name}-allow-lb-health-check"
  project = var.project_id
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["80", "8080", "443"]
  }

  # Load balancer health check ranges
  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]

  target_tags = ["load-balancer-backend"]

  depends_on = [google_compute_network.vpc_network]
}

# Security policy (Cloud Armor) for DDoS protection
resource "google_compute_security_policy" "security_policy" {
  name    = "${local.service_name}-security-policy"
  project = var.project_id

  description = "Security policy for Foundations & Guards service"

  # Default rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }

  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Rate limit rule"
    
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      
      ban_duration_sec = 600
    }
  }

  # Block known malicious IPs (example)
  rule {
    action   = "deny(403)"
    priority = "500"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = [
          # Add known malicious IP ranges here
        ]
      }
    }
    description = "Block malicious IPs"
  }

  depends_on = [google_project_service.compute_api]
}
