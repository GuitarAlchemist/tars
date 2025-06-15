#!/usr/bin/env python3
"""
Simple deployment function for autonomous quality loop
"""

import os
import sys
import subprocess
import random
from pathlib import Path

def deploy_project_simple(project_path):
    """Deploy a specific project and return deployment info"""
    
    if not os.path.exists(project_path):
        return None
    
    project_name = os.path.basename(project_path)
    image_name = f"tars-{project_name.lower()}"
    container_name = f"tars-{project_name.lower()}-container"
    port = 5000 + random.randint(100, 999)
    
    try:
        # Stop and remove existing container
        subprocess.run(f"docker stop {container_name}", shell=True, capture_output=True)
        subprocess.run(f"docker rm {container_name}", shell=True, capture_output=True)
        
        # Build Docker image
        build_result = subprocess.run(
            f"docker build -t {image_name} .", 
            shell=True, 
            cwd=project_path,
            capture_output=True, 
            text=True,
            timeout=120
        )
        
        if build_result.returncode != 0:
            return {
                'success': False,
                'error': build_result.stderr,
                'build_logs': build_result.stderr
            }
        
        # Run container
        run_result = subprocess.run(
            f"docker run -d --name {container_name} -p {port}:5000 {image_name}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if run_result.returncode != 0:
            return {
                'success': False,
                'error': run_result.stderr
            }
        
        return {
            'success': True,
            'container_name': container_name,
            'image_name': image_name,
            'port': port,
            'ip': 'localhost',
            'url': f"http://localhost:{port}"
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Deployment timed out'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main function for standalone testing"""
    if len(sys.argv) < 2:
        print("Usage: python simple-deploy.py <project_path>")
        return 1
    
    project_path = sys.argv[1]
    result = deploy_project_simple(project_path)
    
    if result and result['success']:
        print(f"SUCCESS:{result['container_name']}:{result['ip']}:{result['port']}")
        return 0
    else:
        error = result['error'] if result else "Unknown error"
        print(f"FAILED:{error}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
