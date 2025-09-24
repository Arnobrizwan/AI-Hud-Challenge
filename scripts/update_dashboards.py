#!/usr/bin/env python3
# scripts/update_dashboards.py
import requests
import json
import os
import sys
from pathlib import Path

def update_grafana_dashboard():
    """Update Grafana dashboards with latest configuration"""
    grafana_url = os.getenv('GRAFANA_URL', 'http://localhost:3001')
    grafana_user = os.getenv('GRAFANA_USER', 'admin')
    grafana_password = os.getenv('GRAFANA_PASSWORD', 'admin')
    
    # Load dashboard configuration
    dashboard_path = Path('monitoring/dashboards/news-hub-dashboard.json')
    if not dashboard_path.exists():
        print("❌ Dashboard configuration not found")
        return False
    
    with open(dashboard_path, 'r') as f:
        dashboard_config = json.load(f)
    
    # Authenticate with Grafana
    auth_url = f"{grafana_url}/api/auth/keys"
    auth_data = {
        "name": "api-key",
        "role": "Admin"
    }
    
    try:
        # Get API key
        response = requests.post(
            auth_url,
            json=auth_data,
            auth=(grafana_user, grafana_password)
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to authenticate with Grafana: {response.text}")
            return False
        
        api_key = response.json()['key']
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Update dashboard
        dashboard_url = f"{grafana_url}/api/dashboards/db"
        response = requests.post(
            dashboard_url,
            json=dashboard_config,
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Dashboard updated successfully")
            return True
        else:
            print(f"❌ Failed to update dashboard: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating dashboard: {str(e)}")
        return False

def main():
    """Main function"""
    print("📊 Updating monitoring dashboards...")
    
    if update_grafana_dashboard():
        print("🎉 Dashboard update completed successfully!")
        sys.exit(0)
    else:
        print("💥 Dashboard update failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
