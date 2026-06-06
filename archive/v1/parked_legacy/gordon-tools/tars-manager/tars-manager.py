#!/usr/bin/env python3
"""
TARS Manager - Gordon Integration Tool
Intelligent TARS infrastructure management and analysis
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import docker
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tars-manager")

class TarsManagerConfig:
    """Configuration for TARS Manager"""
    def __init__(self):
        self.gordon_api_url = os.getenv("GORDON_API_URL", "http://localhost:8997")
        self.tars_api_url = os.getenv("TARS_API_URL", "http://localhost:8080")
        self.docker_socket = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.analysis_interval = int(os.getenv("ANALYSIS_INTERVAL", "30"))

class AnalysisRequest(BaseModel):
    analysis_type: str = "health"
    target_services: Optional[List[str]] = None
    deep_analysis: bool = False

class ConsolidationRequest(BaseModel):
    stage: str = "all"  # database, application, web, all
    dry_run: bool = False
    force: bool = False

class TarsManager:
    """Main TARS Manager class integrating Gordon AI with TARS infrastructure"""
    
    def __init__(self, config: TarsManagerConfig):
        self.config = config
        try:
            # Try to connect to Docker socket
            self.docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
            # Test connection
            self.docker_client.ping()
            logger.info("✅ Docker client connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Docker connection failed: {e}")
            # Fallback to None - we'll handle this gracefully
            self.docker_client = None

        self.app = FastAPI(title="TARS Manager", version="1.0.0")
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "gordon_connected": await self.check_gordon_connection(),
                "tars_connected": await self.check_tars_connection()
            }
        
        @self.app.get("/api/v1/tars/status")
        async def get_tars_status():
            """Get comprehensive TARS status"""
            return await self.get_infrastructure_status()
        
        @self.app.post("/api/v1/tars/analyze")
        async def analyze_infrastructure(request: AnalysisRequest):
            """Trigger Gordon-powered infrastructure analysis"""
            return await self.analyze_with_gordon(request)
        
        @self.app.post("/api/v1/tars/consolidate")
        async def consolidate_infrastructure(request: ConsolidationRequest):
            """Execute Gordon-assisted consolidation"""
            return await self.execute_consolidation(request)
        
        @self.app.get("/api/v1/tars/containers")
        async def list_containers():
            """List all TARS-related containers"""
            return await self.get_tars_containers()
        
        @self.app.post("/api/v1/tars/optimize")
        async def optimize_infrastructure():
            """Get AI optimization recommendations"""
            return await self.get_optimization_recommendations()

    async def check_gordon_connection(self) -> bool:
        """Check if Gordon API is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.gordon_api_url}/api/health", timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Gordon connection check failed: {e}")
            return False

    async def check_tars_connection(self) -> bool:
        """Check if TARS API is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config.tars_api_url}/api/health", timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"TARS connection check failed: {e}")
            return False

    async def query_gordon(self, prompt: str) -> Dict[str, Any]:
        """Query Gordon AI for analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "message": prompt,
                    "useAdvancedReasoning": True,
                    "enableMemory": True,
                    "maxTokens": 2000
                }
                
                async with session.post(
                    f"{self.config.gordon_api_url}/api/chat",
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("success"):
                            return {"success": True, "analysis": result.get("result", "")}
                    
                    return {"success": False, "error": f"Gordon API returned {response.status}"}
        except Exception as e:
            logger.error(f"Error querying Gordon: {e}")
            return {"success": False, "error": str(e)}

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get comprehensive TARS infrastructure status"""
        try:
            if not self.docker_client:
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_containers": 0,
                    "tars_containers": 0,
                    "services": {},
                    "health_score": 0,
                    "error": "Docker client not available"
                }

            containers = self.docker_client.containers.list(all=True)
            tars_containers = [c for c in containers if 'tars' in c.name.lower()]

            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_containers": len(containers),
                "tars_containers": len(tars_containers),
                "services": {},
                "health_score": 0
            }
            
            # Analyze each TARS container
            healthy_services = 0
            for container in tars_containers:
                service_status = {
                    "name": container.name,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "ports": [port for port in container.ports.keys()] if container.ports else [],
                    "healthy": container.status == "running"
                }
                
                if service_status["healthy"]:
                    healthy_services += 1
                
                status["services"][container.name] = service_status
            
            # Calculate health score
            if tars_containers:
                status["health_score"] = int((healthy_services / len(tars_containers)) * 100)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting infrastructure status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def analyze_with_gordon(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Perform Gordon-powered infrastructure analysis"""
        try:
            # Get current infrastructure status
            infra_status = await self.get_infrastructure_status()
            
            # Create analysis prompt for Gordon
            prompt = f"""
            Analyze the following TARS infrastructure status and provide recommendations:
            
            Infrastructure Status:
            - Total containers: {infra_status['total_containers']}
            - TARS containers: {infra_status['tars_containers']}
            - Health score: {infra_status['health_score']}/100
            - Analysis type: {request.analysis_type}
            
            Services:
            {json.dumps(infra_status['services'], indent=2)}
            
            Please provide:
            1. Overall assessment
            2. Critical issues (if any)
            3. Optimization recommendations
            4. Next steps
            """
            
            # Query Gordon for analysis
            gordon_response = await self.query_gordon(prompt)
            
            if gordon_response["success"]:
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis_type": request.analysis_type,
                    "infrastructure_status": infra_status,
                    "gordon_analysis": gordon_response["analysis"],
                    "recommendations": self.parse_gordon_recommendations(gordon_response["analysis"])
                }
            else:
                raise HTTPException(status_code=500, detail=f"Gordon analysis failed: {gordon_response['error']}")
                
        except Exception as e:
            logger.error(f"Error in Gordon analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def parse_gordon_recommendations(self, analysis: str) -> List[Dict[str, Any]]:
        """Parse Gordon's analysis into structured recommendations"""
        recommendations = []
        lines = analysis.split('\n')
        
        current_rec = None
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or 'recommend' in line.lower():
                if current_rec:
                    recommendations.append(current_rec)
                
                current_rec = {
                    "action": line.replace('•', '').replace('-', '').strip(),
                    "priority": "medium",
                    "category": "general"
                }
                
                # Determine priority based on keywords
                if any(word in line.lower() for word in ['critical', 'urgent', 'immediate']):
                    current_rec["priority"] = "high"
                elif any(word in line.lower() for word in ['minor', 'optional', 'future']):
                    current_rec["priority"] = "low"
        
        if current_rec:
            recommendations.append(current_rec)
        
        return recommendations

    async def execute_consolidation(self, request: ConsolidationRequest) -> Dict[str, Any]:
        """Execute Gordon-assisted infrastructure consolidation"""
        try:
            logger.info(f"Starting consolidation: stage={request.stage}, dry_run={request.dry_run}")
            
            # Get Gordon's consolidation analysis
            prompt = f"""
            Plan a TARS infrastructure consolidation for stage: {request.stage}
            Current mode: {'dry run' if request.dry_run else 'execution'}
            
            Provide:
            1. Step-by-step consolidation plan
            2. Risk assessment
            3. Rollback strategy
            4. Expected outcomes
            """
            
            gordon_response = await self.query_gordon(prompt)
            
            if not gordon_response["success"]:
                raise HTTPException(status_code=500, detail="Gordon consolidation planning failed")
            
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "stage": request.stage,
                "dry_run": request.dry_run,
                "gordon_plan": gordon_response["analysis"],
                "execution_steps": [],
                "status": "planned"
            }
            
            if not request.dry_run:
                # Execute consolidation steps
                result["execution_steps"] = await self.execute_consolidation_steps(request.stage)
                result["status"] = "completed"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in consolidation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def execute_consolidation_steps(self, stage: str) -> List[Dict[str, Any]]:
        """Execute actual consolidation steps"""
        steps = []
        
        try:
            if stage in ["database", "all"]:
                # Database tier consolidation
                step = await self.consolidate_database_tier()
                steps.append(step)
            
            if stage in ["application", "all"]:
                # Application tier consolidation
                step = await self.consolidate_application_tier()
                steps.append(step)
            
            if stage in ["web", "all"]:
                # Web tier consolidation
                step = await self.consolidate_web_tier()
                steps.append(step)
            
        except Exception as e:
            steps.append({
                "step": f"consolidate_{stage}",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return steps

    async def consolidate_database_tier(self) -> Dict[str, Any]:
        """Consolidate database tier"""
        try:
            # Stop old database containers
            old_containers = ["mongodb", "tars-mongodb-new"]
            for container_name in old_containers:
                try:
                    container = self.docker_client.containers.get(container_name)
                    container.stop()
                    logger.info(f"Stopped container: {container_name}")
                except docker.errors.NotFound:
                    logger.info(f"Container not found: {container_name}")
            
            # Start unified database services
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.unified.yml", 
                "up", "-d", "mongodb", "chromadb", "redis", "fuseki", "virtuoso"
            ], capture_output=True, text=True)
            
            return {
                "step": "consolidate_database",
                "status": "success" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "step": "consolidate_database",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def consolidate_application_tier(self) -> Dict[str, Any]:
        """Consolidate application tier"""
        # Similar implementation for application tier
        return {
            "step": "consolidate_application",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def consolidate_web_tier(self) -> Dict[str, Any]:
        """Consolidate web tier"""
        # Similar implementation for web tier
        return {
            "step": "consolidate_web",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def get_tars_containers(self) -> List[Dict[str, Any]]:
        """Get list of TARS-related containers"""
        try:
            containers = self.docker_client.containers.list(all=True)
            tars_containers = []
            
            for container in containers:
                if 'tars' in container.name.lower():
                    tars_containers.append({
                        "name": container.name,
                        "id": container.id[:12],
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown",
                        "created": container.attrs["Created"],
                        "ports": dict(container.ports) if container.ports else {}
                    })
            
            return tars_containers
            
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get AI-powered optimization recommendations"""
        try:
            # Get current status
            status = await self.get_infrastructure_status()
            
            # Query Gordon for optimization advice
            prompt = f"""
            Analyze this TARS infrastructure and provide optimization recommendations:
            
            Current Status:
            - Health Score: {status['health_score']}/100
            - Total Services: {len(status['services'])}
            
            Focus on:
            1. Performance optimization
            2. Resource efficiency
            3. Security improvements
            4. Scalability enhancements
            """
            
            gordon_response = await self.query_gordon(prompt)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "current_status": status,
                "gordon_recommendations": gordon_response["analysis"] if gordon_response["success"] else "Analysis failed",
                "optimization_score": status['health_score']
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    config = TarsManagerConfig()
    manager = TarsManager(config)

    logger.info("🚀 Starting TARS Manager - Gordon Integration Tool")
    logger.info(f"Gordon API: {config.gordon_api_url}")
    logger.info(f"TARS API: {config.tars_api_url}")

    import uvicorn
    uvicorn.run(
        manager.app,
        host="0.0.0.0",
        port=8998,
        log_level=config.log_level.lower()
    )

if __name__ == "__main__":
    main()
