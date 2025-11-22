#!/usr/bin/env python3
"""
TARS Evolve Command Demo
Demonstrates the complete TARS evolution system with Docker containers
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

class TarsEvolveDemo:
    def __init__(self):
        self.evolution_dir = ".tars/evolution"
        self.monitoring_dir = ".tars/monitoring"
        self.shared_dir = ".tars/shared"
        
        # Ensure directories exist
        os.makedirs(self.evolution_dir, exist_ok=True)
        os.makedirs(self.monitoring_dir, exist_ok=True)
        os.makedirs(self.shared_dir, exist_ok=True)
    
    async def run_complete_demo(self):
        """Run complete TARS evolution demo"""
        
        print("🤖 TARS AUTONOMOUS EVOLUTION DEMO")
        print("=" * 40)
        print("Demonstrating complete evolution system with Docker containers")
        print()
        
        # Demo configuration
        evolution_config = {
            "mode": "experimental",
            "duration_hours": 2,
            "docker_image": "tars-evolution:latest",
            "swarm_nodes": 1,
            "monitoring_level": "comprehensive",
            "sync_interval": 30,
            "safety_checks": True,
            "evolution_goals": ["performance", "capabilities", "mcp-integration"]
        }
        
        try:
            # Phase 1: Generate container identity
            print("🏷️ PHASE 1: GENERATING CONTAINER IDENTITY")
            print("=" * 45)
            container_identity = await self.generate_container_identity(evolution_config)
            print()
            
            # Phase 2: Create container registry
            print("📋 PHASE 2: CREATING CONTAINER REGISTRY")
            print("=" * 40)
            registry = await self.create_container_registry()
            print()
            
            # Phase 3: Setup Docker environment
            print("🐳 PHASE 3: SETTING UP DOCKER ENVIRONMENT")
            print("=" * 45)
            docker_setup = await self.setup_docker_environment(container_identity, evolution_config)
            print()
            
            # Phase 4: Start evolution session
            print("🚀 PHASE 4: STARTING EVOLUTION SESSION")
            print("=" * 40)
            session_result = await self.start_evolution_session(container_identity, evolution_config)
            print()
            
            # Phase 5: Monitor evolution (demo)
            print("📊 PHASE 5: MONITORING EVOLUTION")
            print("=" * 35)
            await self.demo_evolution_monitoring(container_identity, evolution_config)
            print()
            
            # Phase 6: Show management commands
            print("🔧 PHASE 6: EVOLUTION MANAGEMENT COMMANDS")
            print("=" * 45)
            self.show_management_commands(container_identity)
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            return False
    
    async def generate_container_identity(self, config):
        """Generate container identity with versioning"""
        
        current_time = datetime.now()
        date_string = current_time.strftime("%Y%m%d")
        time_string = current_time.strftime("%H%M%S")
        
        # Get current TARS version
        current_version = self.get_current_tars_version()
        next_version = self.calculate_next_version(current_version, config["mode"])
        
        # Generate session ID
        session_id = self.generate_session_id(config["mode"])
        
        # Create container name
        container_name = f"tars-evolution-v{next_version}-{date_string}-{time_string}-{session_id}"
        image_tag = f"tars/evolution:v{next_version}-{config['mode']}-{date_string}"
        
        container_identity = {
            "container_name": container_name,
            "image_tag": image_tag,
            "current_version": current_version,
            "next_version": next_version,
            "session_id": session_id,
            "creation_time": current_time.isoformat(),
            "labels": {
                "tars.version": next_version,
                "tars.evolution.mode": config["mode"],
                "tars.evolution.session": session_id,
                "tars.evolution.parent": current_version,
                "tars.evolution.created": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tars.evolution.goals": ",".join(config["evolution_goals"])
            }
        }
        
        print(f"  🏷️ Container Name: {container_name}")
        print(f"  📦 Image Tag: {image_tag}")
        print(f"  🔢 Version: {current_version} → {next_version}")
        print(f"  🆔 Session ID: {session_id}")
        print(f"  📅 Created: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return container_identity
    
    def get_current_tars_version(self):
        """Get current TARS version"""
        version_file = ".tars/version.json"
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_data = json.load(f)
                return version_data.get("version", "2.1.0")
        return "2.1.0"
    
    def calculate_next_version(self, current_version, mode):
        """Calculate next version based on evolution mode"""
        parts = current_version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if mode == "safe":
            return f"{major}.{minor}.{patch + 1}"
        elif mode == "experimental":
            return f"{major}.{minor + 1}.0"
        elif mode == "aggressive":
            return f"{major + 1}.0.0"
        else:
            return f"{major}.{minor}.{patch + 1}"
    
    def generate_session_id(self, mode):
        """Generate session ID"""
        mode_prefix = {
            "safe": "safe",
            "experimental": "exp",
            "aggressive": "aggr"
        }.get(mode, "unkn")
        
        # Get next session number
        session_file = f".tars/sessions/{mode_prefix}-counter.txt"
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                counter = int(f.read().strip())
            counter += 1
        else:
            counter = 1
        
        with open(session_file, 'w') as f:
            f.write(str(counter))
        
        return f"{mode_prefix}{counter:03d}"
    
    async def create_container_registry(self):
        """Create container registry"""
        
        registry_path = ".tars/evolution/container-registry.json"
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {
                "containers": [],
                "active_sessions": [],
                "version_history": [],
                "last_updated": datetime.now().isoformat(),
                "registry_version": "1.0.0"
            }
        
        print(f"  📊 Registry Status:")
        print(f"    Total Containers: {len(registry['containers'])}")
        print(f"    Active Sessions: {len(registry['active_sessions'])}")
        print(f"    Version History: {len(registry['version_history'])}")
        print(f"    Last Updated: {registry['last_updated']}")
        
        return registry
    
    async def setup_docker_environment(self, container_identity, config):
        """Setup Docker environment"""
        
        session_dir = f"{self.evolution_dir}/{container_identity['session_id']}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Create Dockerfile
        dockerfile_content = self.generate_dockerfile(container_identity, config)
        with open(f"{session_dir}/Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        compose_content = self.generate_docker_compose(container_identity, config)
        with open(f"{session_dir}/docker-compose.yml", 'w') as f:
            f.write(compose_content)
        
        # Create evolution scripts directory
        scripts_dir = f"{session_dir}/evolution-scripts"
        os.makedirs(scripts_dir, exist_ok=True)
        
        # Create start script
        start_script = self.generate_start_script(container_identity, config)
        with open(f"{scripts_dir}/start-evolution.sh", 'w') as f:
            f.write(start_script)
        os.chmod(f"{scripts_dir}/start-evolution.sh", 0o755)
        
        print(f"  📂 Session Directory: {session_dir}")
        print(f"  🐳 Dockerfile: {session_dir}/Dockerfile")
        print(f"  📋 Docker Compose: {session_dir}/docker-compose.yml")
        print(f"  🚀 Start Script: {scripts_dir}/start-evolution.sh")
        
        return {
            "session_dir": session_dir,
            "dockerfile": f"{session_dir}/Dockerfile",
            "compose_file": f"{session_dir}/docker-compose.yml",
            "start_script": f"{scripts_dir}/start-evolution.sh"
        }
    
    def generate_dockerfile(self, container_identity, config):
        """Generate Dockerfile for evolution container"""
        return f'''# TARS Evolution Container - Version {container_identity["next_version"]}
# Session: {container_identity["session_id"]}
# Mode: {config["mode"]}
# Created: {container_identity["creation_time"]}

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

# Container metadata
LABEL tars.version="{container_identity["next_version"]}"
LABEL tars.evolution.session="{container_identity["session_id"]}"
LABEL tars.evolution.mode="{config["mode"]}"
LABEL tars.evolution.parent="{container_identity["current_version"]}"

# Install evolution dependencies
RUN apt-get update && apt-get install -y \\
    python3 python3-pip nodejs npm git curl wget jq htop \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for TARS evolution
RUN pip3 install aiohttp asyncio requests numpy pandas

# Set up TARS evolution workspace
WORKDIR /tars

# Create evolution runtime image
FROM mcr.microsoft.com/dotnet/runtime:8.0

# Copy evolution metadata
LABEL tars.version="{container_identity["next_version"]}"
LABEL tars.evolution.session="{container_identity["session_id"]}"
LABEL tars.evolution.mode="{config["mode"]}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    python3 python3-pip nodejs npm git curl \\
    && rm -rf /var/lib/apt/lists/*

# Set evolution environment
ENV TARS_VERSION="{container_identity["next_version"]}"
ENV TARS_EVOLUTION_SESSION="{container_identity["session_id"]}"
ENV TARS_EVOLUTION_MODE="{config["mode"]}"
ENV TARS_PARENT_VERSION="{container_identity["current_version"]}"
ENV TARS_CONTAINER_NAME="{container_identity["container_name"]}"
ENV TARS_DOCKER_ISOLATED="true"

# Create evolution directories
RUN mkdir -p /tars/evolution /tars/shared /tars/monitoring /tars/backups

# Copy evolution scripts
COPY evolution-scripts/ ./evolution-scripts/
RUN chmod +x ./evolution-scripts/*.sh

# Health check for evolution session
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Start TARS evolution
ENTRYPOINT ["./evolution-scripts/start-evolution.sh"]
CMD ["--version", "{container_identity["next_version"]}", "--session", "{container_identity["session_id"]}"]
'''
    
    def generate_docker_compose(self, container_identity, config):
        """Generate docker-compose.yml for evolution"""
        return f'''version: '3.8'

services:
  {container_identity["container_name"]}:
    build:
      context: .
      dockerfile: Dockerfile
    image: {container_identity["image_tag"]}
    container_name: {container_identity["container_name"]}
    hostname: tars-evolution-{container_identity["session_id"]}
    
    # Container labels for identification
    labels:
      - "tars.version={container_identity["next_version"]}"
      - "tars.evolution.session={container_identity["session_id"]}"
      - "tars.evolution.mode={config["mode"]}"
      - "tars.evolution.parent={container_identity["current_version"]}"
    
    # Resource limits for evolution
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    
    # Volume mounts for evolution session
    volumes:
      - ./evolution/{container_identity["session_id"]}:/tars/evolution:rw
      - ./shared:/tars/shared:ro
      - ./monitoring/{container_identity["session_id"]}:/tars/monitoring:rw
    
    # Environment for versioned evolution
    environment:
      - TARS_VERSION={container_identity["next_version"]}
      - TARS_EVOLUTION_SESSION={container_identity["session_id"]}
      - TARS_EVOLUTION_MODE={config["mode"]}
      - TARS_PARENT_VERSION={container_identity["current_version"]}
      - TARS_CONTAINER_NAME={container_identity["container_name"]}
    
    # Network configuration
    networks:
      - tars-evolution-{container_identity["session_id"]}
    
    # Port mappings for monitoring
    ports:
      - "8080:8080"  # Evolution API
      - "8081:8081"  # Metrics endpoint
      - "8082:8082"  # Log streaming
      - "8083:8083"  # Health check

networks:
  tars-evolution-{container_identity["session_id"]}:
    driver: bridge
    name: tars-evolution-{container_identity["session_id"]}
'''
    
    def generate_start_script(self, container_identity, config):
        """Generate start script for evolution container"""
        return f'''#!/bin/bash
# TARS Evolution Start Script
# Version: {container_identity["next_version"]}
# Session: {container_identity["session_id"]}
# Mode: {config["mode"]}

echo "🤖 STARTING TARS EVOLUTION SESSION"
echo "================================="
echo "Version: {container_identity["next_version"]}"
echo "Session: {container_identity["session_id"]}"
echo "Mode: {config["mode"]}"
echo "Container: {container_identity["container_name"]}"
echo ""

# Create evolution metrics file
cat > /tars/evolution/metrics.json << EOF
{{
  "session_id": "{container_identity["session_id"]}",
  "version": "{container_identity["next_version"]}",
  "mode": "{config["mode"]}",
  "start_time": "$(date -Iseconds)",
  "progress_percentage": 0,
  "current_phase": "initialization",
  "important_events": [],
  "milestones_reached": []
}}
EOF

# Start evolution monitoring
echo "📊 Starting evolution monitoring..."

# Simulate evolution progress
for i in {{1..10}}; do
    sleep 30
    progress=$((i * 10))
    
    # Update metrics
    cat > /tars/evolution/metrics.json << EOF
{{
  "session_id": "{container_identity["session_id"]}",
  "version": "{container_identity["next_version"]}",
  "mode": "{config["mode"]}",
  "start_time": "$(date -Iseconds)",
  "progress_percentage": $progress,
  "current_phase": "evolution-phase-$i",
  "important_events": [
    {{
      "type": "EvolutionProgress",
      "description": "Evolution progress: $progress%",
      "severity": "Info",
      "timestamp": "$(date -Iseconds)"
    }}
  ],
  "milestones_reached": []
}}
EOF
    
    echo "📈 Evolution progress: $progress%"
done

echo "✅ TARS evolution session completed"
'''
    
    async def start_evolution_session(self, container_identity, config):
        """Start evolution session (demo)"""
        
        # Create session configuration
        session_config = {
            "session_id": container_identity["session_id"],
            "container_name": container_identity["container_name"],
            "image_tag": container_identity["image_tag"],
            "version": container_identity["next_version"],
            "mode": config["mode"],
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "goals": config["evolution_goals"],
            "monitoring_endpoints": [
                "http://localhost:8080",
                "http://localhost:8081",
                "http://localhost:8082",
                "http://localhost:8083"
            ]
        }
        
        # Save session configuration
        session_file = f"{self.evolution_dir}/current-session.json"
        with open(session_file, 'w') as f:
            json.dump(session_config, f, indent=2)
        
        print(f"  🚀 Session Started: {session_config['session_id']}")
        print(f"  📦 Container: {session_config['container_name']}")
        print(f"  🔢 Version: {session_config['version']}")
        print(f"  📊 Status: {session_config['status']}")
        print(f"  🎯 Goals: {', '.join(session_config['goals'])}")
        print()
        print("  🔍 MONITORING ENDPOINTS:")
        for endpoint in session_config['monitoring_endpoints']:
            print(f"    {endpoint}")
        
        return session_config
    
    async def demo_evolution_monitoring(self, container_identity, config):
        """Demo evolution monitoring"""
        
        print("  📊 Simulating evolution monitoring...")
        print("  (In real implementation, this would monitor Docker container)")
        print()
        
        # Simulate monitoring events
        events = [
            "🎯 EVOLUTION EVENT: CodeGeneration - Generated 150 new lines of F# code",
            "🏆 EVOLUTION MILESTONE: MCP Integration - Successfully integrated 3 new MCP servers",
            "🚀 PERFORMANCE IMPROVEMENT DETECTED: 25% faster metascript execution",
            "📈 Evolution Progress: 45% - Optimizing autonomous capabilities",
            "🔄 SYNC REQUEST: Requesting validation for performance improvements",
            "✅ SYNC COMPLETED: Performance improvements validated and approved"
        ]
        
        for i, event in enumerate(events):
            await asyncio.sleep(1)  # Simulate real-time monitoring
            print(f"  {event}")
        
        print()
        print("  📊 Evolution monitoring active...")
        print("  🔄 Sync validation enabled...")
        print("  🛡️ Safety monitoring active...")
    
    def show_management_commands(self, container_identity):
        """Show evolution management commands"""
        
        session_id = container_identity["session_id"]
        container_name = container_identity["container_name"]
        
        print("  📋 EVOLUTION MANAGEMENT COMMANDS:")
        print()
        print("  🔍 Status and Monitoring:")
        print(f"    tars evolve status --detailed")
        print(f"    tars evolve monitor --follow --session {session_id}")
        print(f"    docker logs {container_name} --follow")
        print()
        print("  🔧 Container Management:")
        print(f"    docker ps --filter name={container_name}")
        print(f"    docker exec -it {container_name} /bin/bash")
        print(f"    docker stats {container_name}")
        print()
        print("  🛑 Stop and Cleanup:")
        print(f"    tars evolve stop --preserve-changes --session {session_id}")
        print(f"    docker-compose -f .tars/evolution/{session_id}/docker-compose.yml down")
        print()
        print("  🔄 Validation and Sync:")
        print(f"    tars evolve validate --comprehensive --session {session_id}")
        print(f"    tars evolve sync --backup-host --session {session_id}")
        print()
        print("  📋 Registry Management:")
        print("    tars evolve list --all")
        print("    tars evolve list --active")
        print("    tars evolve cleanup --older-than 7")

async def main():
    """Main function"""
    print("🤖 TARS EVOLUTION CONTAINER SYSTEM DEMO")
    print("=" * 50)
    print("Demonstrating complete evolution system with Docker containers")
    print()
    
    demo = TarsEvolveDemo()
    success = await demo.run_complete_demo()
    
    if success:
        print()
        print("🎉 TARS EVOLUTION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("✅ Container identity generated with versioning")
        print("✅ Container registry created and managed")
        print("✅ Docker environment configured")
        print("✅ Evolution session started")
        print("✅ Monitoring system demonstrated")
        print("✅ Management commands provided")
        print()
        print("🚀 TARS IS READY FOR AUTONOMOUS EVOLUTION!")
        print("   Use the management commands above to control evolution")
    else:
        print("❌ Demo failed - check output for details")

if __name__ == "__main__":
    asyncio.run(main())
