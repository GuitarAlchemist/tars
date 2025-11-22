#!/usr/bin/env python3
"""
TARS CLI - Command Line Interface for TARS Manager
Provides easy access to Gordon-powered TARS management
"""

import argparse
import asyncio
import json
import sys
from typing import Dict, Any
import aiohttp

class TarsCLI:
    """Command Line Interface for TARS Manager"""
    
    def __init__(self, base_url: str = "http://localhost:8998"):
        self.base_url = base_url
        
    async def make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to TARS Manager API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        return await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        return await response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def analyze(self, analysis_type: str = "health", deep: bool = False):
        """Analyze TARS infrastructure"""
        print(f"🔍 Analyzing TARS infrastructure ({analysis_type})...")
        
        data = {
            "analysis_type": analysis_type,
            "deep_analysis": deep
        }
        
        result = await self.make_request("POST", "/api/v1/tars/analyze", data)
        
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return
        
        print(f"✅ Analysis complete!")
        print(f"📊 Health Score: {result['infrastructure_status']['health_score']}/100")
        print(f"🤖 Gordon's Analysis:")
        print(result['gordon_analysis'])
        
        if result.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                print(f"  {i}. {priority_emoji} {rec['action']}")
    
    async def consolidate(self, stage: str = "all", dry_run: bool = True, force: bool = False):
        """Execute infrastructure consolidation"""
        mode = "DRY RUN" if dry_run else "EXECUTION"
        print(f"🔄 Starting consolidation ({stage} stage) - {mode}")
        
        data = {
            "stage": stage,
            "dry_run": dry_run,
            "force": force
        }
        
        result = await self.make_request("POST", "/api/v1/tars/consolidate", data)
        
        if "error" in result:
            print(f"❌ Consolidation failed: {result['error']}")
            return
        
        print(f"✅ Consolidation {result['status']}!")
        print(f"🤖 Gordon's Plan:")
        print(result['gordon_plan'])
        
        if result.get('execution_steps'):
            print(f"\n📋 Execution Steps:")
            for step in result['execution_steps']:
                status_emoji = "✅" if step['status'] == "success" else "❌"
                print(f"  {status_emoji} {step['step']}")
    
    async def monitor(self, services: str = "all", interval: int = 30):
        """Monitor TARS infrastructure"""
        print(f"📊 Monitoring TARS services ({services})...")
        
        # Get current status
        result = await self.make_request("GET", "/api/v1/tars/status")
        
        if "error" in result:
            print(f"❌ Monitoring failed: {result['error']}")
            return
        
        print(f"🏥 Health Score: {result['health_score']}/100")
        print(f"📦 Total Containers: {result['total_containers']}")
        print(f"🚀 TARS Containers: {result['tars_containers']}")
        
        print(f"\n📋 Services Status:")
        for name, service in result['services'].items():
            status_emoji = "✅" if service['healthy'] else "❌"
            print(f"  {status_emoji} {name}: {service['status']}")
    
    async def optimize(self, target: str = "performance"):
        """Get optimization recommendations"""
        print(f"⚡ Getting optimization recommendations ({target})...")
        
        result = await self.make_request("POST", "/api/v1/tars/optimize")
        
        if "error" in result:
            print(f"❌ Optimization failed: {result['error']}")
            return
        
        print(f"✅ Optimization analysis complete!")
        print(f"📊 Current Score: {result['optimization_score']}/100")
        print(f"🤖 Gordon's Recommendations:")
        print(result['gordon_recommendations'])
    
    async def status(self):
        """Get TARS status"""
        print("📊 Getting TARS status...")

        result = await self.make_request("GET", "/api/v1/tars/status")

        if "error" in result:
            print(f"❌ Status check failed: {result['error']}")
            return

        print(f"✅ TARS Status:")
        print(f"  🏥 Health Score: {result['health_score']}/100")
        print(f"  📦 Total Containers: {result['total_containers']}")
        print(f"  🚀 TARS Containers: {result['tars_containers']}")
        print(f"  ⏰ Last Updated: {result['timestamp']}")

    async def blue_green(self, action: str = "status", target: str = "both"):
        """Manage blue-green deployment"""
        print(f"🔄 Blue-Green {action.title()} ({target})...")

        if action == "status":
            # Check blue-green deployment status
            containers = {
                'blue': 'tars-blue-production',
                'green': 'tars-green-evolution',
                'green-stable': 'tars-green-stable',
                'monitor': 'tars-evolution-monitor'
            }

            print("\n📊 Blue-Green Deployment Status:")

            try:
                # Try to get Docker client
                import docker
                client = docker.from_env()

                for env, container_name in containers.items():
                    try:
                        container = client.containers.get(container_name)
                        status = container.status

                        if env == 'blue':
                            print(f"  🔵 Blue (Production): {status}")
                            print(f"     URL: http://localhost:9000")
                        elif env == 'green':
                            print(f"  🟢 Green (Evolution): {status}")
                            print(f"     URL: http://localhost:9001")
                        elif env == 'green-stable':
                            print(f"  🟢 Green (Stable): {status}")
                            print(f"     URL: http://localhost:8088")
                        elif env == 'monitor':
                            print(f"  📊 Monitor: {status}")
                            print(f"     URL: http://localhost:8090")

                    except docker.errors.NotFound:
                        print(f"  ❌ {env.title()}: Container not found")
                    except Exception as e:
                        print(f"  ⚠️  {env.title()}: Error - {e}")

            except Exception as e:
                print(f"❌ Docker client not available: {e}")
                print("💡 Try: docker ps | findstr tars")

        else:
            print(f"⚠️  Blue-Green {action} operation not yet implemented")
            print("💡 Available actions: status")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="TARS Manager CLI - Gordon Integration")
    parser.add_argument("--url", default="http://localhost:8998", help="TARS Manager API URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze TARS infrastructure")
    analyze_parser.add_argument("--type", choices=["health", "performance", "security", "consolidation"], 
                               default="health", help="Analysis type")
    analyze_parser.add_argument("--deep", action="store_true", help="Deep analysis")
    
    # Consolidate command
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate infrastructure")
    consolidate_parser.add_argument("--stage", choices=["database", "application", "web", "all"], 
                                   default="all", help="Consolidation stage")
    consolidate_parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode")
    consolidate_parser.add_argument("--execute", action="store_true", help="Execute consolidation")
    consolidate_parser.add_argument("--force", action="store_true", help="Force consolidation")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor TARS infrastructure")
    monitor_parser.add_argument("--services", default="all", help="Services to monitor")
    monitor_parser.add_argument("--interval", type=int, default=30, help="Monitoring interval")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Get optimization recommendations")
    optimize_parser.add_argument("--target", choices=["performance", "resources", "security", "all"], 
                                default="performance", help="Optimization target")
    
    # Status command
    subparsers.add_parser("status", help="Get TARS status")

    # Blue-Green command
    bg_parser = subparsers.add_parser("blue-green", help="Manage blue-green deployment")
    bg_parser.add_argument("--action", choices=["status", "switch", "promote", "rollback"],
                          default="status", help="Blue-green action")
    bg_parser.add_argument("--target", choices=["blue", "green", "both"],
                          default="both", help="Target environment")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TarsCLI(args.url)
    
    try:
        if args.command == "analyze":
            asyncio.run(cli.analyze(args.type, args.deep))
        elif args.command == "consolidate":
            dry_run = not args.execute
            asyncio.run(cli.consolidate(args.stage, dry_run, args.force))
        elif args.command == "monitor":
            asyncio.run(cli.monitor(args.services, args.interval))
        elif args.command == "optimize":
            asyncio.run(cli.optimize(args.target))
        elif args.command == "status":
            asyncio.run(cli.status())
        elif args.command == "blue-green":
            asyncio.run(cli.blue_green(args.action, args.target))
    except KeyboardInterrupt:
        print("\n👋 TARS CLI interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
