#!/usr/bin/env python3
# scripts/smoke_tests.py
import requests
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVICES = [
    'safety-service', 'ingestion-service', 'content-extraction-service',
    'content-enrichment-service', 'deduplication-service', 'src', 'summarization-service', 
    'personalization-service', 'notification-service', 'feedback-service', 
    'evaluation-service', 'mlops-orchestration-service', 'storage-service',
    'realtime-interface-service', 'observability-service'
]

def test_service_health(service_name: str, base_url: str) -> bool:
    """Test if a service is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ {service_name} health check failed: {str(e)}")
        return False

def test_service_endpoints(service_name: str, base_url: str) -> bool:
    """Test basic service endpoints"""
    try:
        # Test health endpoint
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code != 200:
            return False
            
        # Test basic API endpoint if available
        try:
            api_response = requests.get(f"{base_url}/", timeout=10)
            if api_response.status_code not in [200, 404]:  # 404 is ok for root
                return False
        except:
            pass  # Some services might not have root endpoint
            
        return True
    except Exception as e:
        print(f"❌ {service_name} endpoint test failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run smoke tests for News Hub Pipeline')
    parser.add_argument('--environment', default='production', 
                       choices=['production', 'staging', 'local'],
                       help='Environment to test')
    parser.add_argument('--base-url', help='Override base URL for testing')
    parser.add_argument('--timeout', type=int, default=30, 
                       help='Timeout for each service check')
    
    args = parser.parse_args()
    
    # Determine base URL
    if args.base_url:
        base_url = args.base_url.rstrip('/')
    elif args.environment == 'production':
        base_url = "https://api.newshub.com"
    elif args.environment == 'staging':
        base_url = "https://staging-api.newshub.com"
    else:  # local
        base_url = "http://localhost:8000"
    
    print(f"🧪 Running smoke tests for {args.environment} environment...")
    print(f"🌐 Base URL: {base_url}")
    
    failed_services = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Test health endpoints
        health_futures = {
            executor.submit(test_service_health, service, f"{base_url}/{service}"): service
            for service in SERVICES
        }
        
        for future in as_completed(health_futures):
            service = health_futures[future]
            if not future.result():
                failed_services.append(f"{service} (health)")
        
        # Test basic endpoints
        endpoint_futures = {
            executor.submit(test_service_endpoints, service, f"{base_url}/{service}"): service
            for service in SERVICES
        }
        
        for future in as_completed(endpoint_futures):
            service = endpoint_futures[future]
            if not future.result():
                if f"{service} (endpoints)" not in failed_services:
                    failed_services.append(f"{service} (endpoints)")
    
    if failed_services:
        print(f"❌ {len(failed_services)} service checks failed:")
        for service in failed_services:
            print(f"   • {service}")
        sys.exit(1)
    else:
        print("✅ All smoke tests passed!")
        print(f"🎉 {len(SERVICES)} services are healthy and responding correctly!")

if __name__ == "__main__":
    main()
