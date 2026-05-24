#!/usr/bin/env python3
"""
TARS MCP Client Implementation
Enables TARS to consume external MCP servers and integrate their capabilities
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import aiohttp
import websockets

class TarsMcpClient:
    def __init__(self):
        self.registered_servers = {}
        self.available_tools = {}
        self.available_resources = {}
        self.server_registry_path = ".tars/mcp_servers.json"
        self.load_server_registry()
    
    def load_server_registry(self):
        """Load registered MCP servers from file"""
        if os.path.exists(self.server_registry_path):
            try:
                with open(self.server_registry_path, 'r') as f:
                    data = json.load(f)
                    self.registered_servers = data.get("servers", {})
                    self.available_tools = data.get("tools", {})
                    self.available_resources = data.get("resources", {})
                print(f"📋 Loaded {len(self.registered_servers)} registered MCP servers")
            except Exception as e:
                print(f"⚠️ Failed to load server registry: {e}")
                self.registered_servers = {}
                self.available_tools = {}
                self.available_resources = {}
        else:
            os.makedirs(os.path.dirname(self.server_registry_path), exist_ok=True)
    
    def save_server_registry(self):
        """Save registered MCP servers to file"""
        try:
            registry_data = {
                "servers": self.registered_servers,
                "tools": self.available_tools,
                "resources": self.available_resources,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.server_registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            print(f"💾 Saved server registry with {len(self.registered_servers)} servers")
        except Exception as e:
            print(f"❌ Failed to save server registry: {e}")
    
    async def discover_servers_on_network(self, network_range="local", timeout=5):
        """Automatically discover MCP servers on the network"""
        discovered_servers = []
        
        print(f"🔍 Discovering MCP servers on {network_range} network...")
        
        # Common MCP server ports to scan
        common_ports = [3000, 3001, 3002, 8000, 8001, 8080]
        
        if network_range == "local":
            # Scan localhost
            for port in common_ports:
                url = f"http://localhost:{port}"
                if await self.test_mcp_server(url, timeout):
                    discovered_servers.append(url)
        
        print(f"🎯 Discovered {len(discovered_servers)} MCP servers")
        return discovered_servers
    
    async def test_mcp_server(self, url, timeout=5):
        """Test if a URL hosts an MCP server"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Try to connect and get server info
                async with session.get(f"{url}/mcp/info") as response:
                    if response.status == 200:
                        server_info = await response.json()
                        if "name" in server_info and "capabilities" in server_info:
                            return True
        except:
            pass
        
        return False
    
    async def register_server(self, server_url, server_name=None, auto_discover=True):
        """Register an external MCP server"""
        
        print(f"📡 Registering MCP server: {server_url}")
        
        try:
            # Connect and get server capabilities
            server_info = await self.get_server_info(server_url)
            
            if not server_name:
                server_name = server_info.get("name", f"server_{len(self.registered_servers)}")
            
            # Register server
            self.registered_servers[server_name] = {
                "url": server_url,
                "info": server_info,
                "tools": server_info.get("tools", []),
                "resources": server_info.get("resources", []),
                "prompts": server_info.get("prompts", []),
                "registered_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Update available capabilities
            self.update_available_capabilities(server_name)
            
            # Save registry
            self.save_server_registry()
            
            print(f"✅ Successfully registered: {server_name}")
            print(f"   Tools: {len(server_info.get('tools', []))}")
            print(f"   Resources: {len(server_info.get('resources', []))}")
            print(f"   Prompts: {len(server_info.get('prompts', []))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to register server {server_url}: {e}")
            return False
    
    async def get_server_info(self, server_url):
        """Get server information and capabilities"""
        # This would implement the actual MCP protocol communication
        # For demo purposes, return mock data
        return {
            "name": f"external-server-{len(self.registered_servers)}",
            "version": "1.0.0",
            "description": "External MCP Server",
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True
            },
            "tools": [
                {
                    "name": "example_tool",
                    "description": "Example external tool",
                    "inputSchema": {"type": "object", "properties": {}}
                }
            ],
            "resources": [
                {
                    "uri": "external://data/",
                    "description": "External data resource",
                    "mimeType": "application/json"
                }
            ],
            "prompts": [
                {
                    "name": "external_prompt",
                    "description": "External prompt template",
                    "arguments": []
                }
            ]
        }
    
    def update_available_capabilities(self, server_name):
        """Update available tools and resources from registered server"""
        server = self.registered_servers[server_name]
        
        # Add tools
        for tool in server.get("tools", []):
            tool_name = tool["name"]
            self.available_tools[tool_name] = {
                "server": server_name,
                "description": tool["description"],
                "inputSchema": tool.get("inputSchema", {})
            }
        
        # Add resources
        for resource in server.get("resources", []):
            resource_uri = resource["uri"]
            self.available_resources[resource_uri] = {
                "server": server_name,
                "description": resource["description"],
                "mimeType": resource.get("mimeType", "text/plain")
            }
    
    async def call_external_tool(self, tool_name, arguments, server_name=None):
        """Call a tool on an external MCP server"""
        
        # Find server that has this tool
        if not server_name:
            if tool_name not in self.available_tools:
                raise ValueError(f"Tool '{tool_name}' not found in any registered server")
            server_name = self.available_tools[tool_name]["server"]
        
        if server_name not in self.registered_servers:
            raise ValueError(f"Server '{server_name}' not registered")
        
        server = self.registered_servers[server_name]
        
        print(f"🔧 Calling tool '{tool_name}' on server '{server_name}'")
        
        # Execute tool call (would implement actual MCP protocol)
        result = await self.execute_remote_tool_call(server["url"], tool_name, arguments)
        
        return result
    
    async def execute_remote_tool_call(self, server_url, tool_name, arguments):
        """Execute tool call on remote MCP server"""
        # This would implement the actual MCP protocol communication
        # For demo purposes, return mock result
        return {
            "success": True,
            "result": f"Mock result from {tool_name} with args: {arguments}",
            "server_url": server_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def access_external_resource(self, resource_uri, server_name=None):
        """Access a resource on an external MCP server"""
        
        # Find server that has this resource
        if not server_name:
            if resource_uri not in self.available_resources:
                raise ValueError(f"Resource '{resource_uri}' not found in any registered server")
            server_name = self.available_resources[resource_uri]["server"]
        
        server = self.registered_servers[server_name]
        
        print(f"📄 Accessing resource '{resource_uri}' on server '{server_name}'")
        
        # Access resource (would implement actual MCP protocol)
        resource_content = await self.fetch_remote_resource(server["url"], resource_uri)
        
        return resource_content
    
    async def fetch_remote_resource(self, server_url, resource_uri):
        """Fetch resource from remote MCP server"""
        # This would implement the actual MCP protocol communication
        # For demo purposes, return mock content
        return {
            "uri": resource_uri,
            "content": f"Mock content from {resource_uri}",
            "mimeType": "application/json",
            "server_url": server_url,
            "timestamp": datetime.now().isoformat()
        }
    
    async def compose_cross_server_workflow(self, workflow_definition):
        """Compose tools from multiple MCP servers into a workflow"""
        
        print(f"🔄 Executing cross-server workflow: {workflow_definition['name']}")
        
        results = []
        workflow_context = {}
        
        for step in workflow_definition["steps"]:
            step_name = step["name"]
            tool_name = step["tool"]
            arguments = step["arguments"]
            server_name = step.get("server")
            
            print(f"  📋 Step: {step_name}")
            
            try:
                # Resolve dynamic arguments from previous steps
                resolved_arguments = self.resolve_workflow_arguments(arguments, workflow_context)
                
                if tool_name.startswith("tars_"):
                    # Local TARS tool
                    result = await self.execute_local_tars_tool(tool_name, resolved_arguments)
                else:
                    # External MCP tool
                    result = await self.call_external_tool(tool_name, resolved_arguments, server_name)
                
                # Store result in workflow context
                workflow_context[step_name] = result
                
                results.append({
                    "step": step_name,
                    "tool": tool_name,
                    "server": server_name or "tars",
                    "success": result.get("success", True),
                    "result": result
                })
                
                print(f"    ✅ Completed successfully")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results.append({
                    "step": step_name,
                    "tool": tool_name,
                    "server": server_name or "tars",
                    "success": False,
                    "error": str(e)
                })
                
                # Handle workflow failure
                if step.get("required", True):
                    break
        
        workflow_result = {
            "workflow": workflow_definition["name"],
            "steps_completed": len(results),
            "total_steps": len(workflow_definition["steps"]),
            "success": all(r["success"] for r in results),
            "results": results,
            "execution_time": datetime.now().isoformat()
        }
        
        print(f"🎯 Workflow completed: {workflow_result['success']}")
        return workflow_result
    
    def resolve_workflow_arguments(self, arguments, context):
        """Resolve dynamic arguments in workflow steps"""
        resolved = {}
        
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("${"):
                # Dynamic reference to previous step result
                ref = value[2:-1]  # Remove ${ and }
                if "." in ref:
                    step_name, field = ref.split(".", 1)
                    if step_name in context:
                        resolved[key] = self.get_nested_value(context[step_name], field)
                    else:
                        resolved[key] = value  # Keep original if not found
                else:
                    resolved[key] = context.get(ref, value)
            else:
                resolved[key] = value
        
        return resolved
    
    def get_nested_value(self, obj, field_path):
        """Get nested value from object using dot notation"""
        fields = field_path.split(".")
        current = obj
        
        for field in fields:
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return None
        
        return current
    
    async def execute_local_tars_tool(self, tool_name, arguments):
        """Execute local TARS tool"""
        # This would integrate with the TARS CLI or internal APIs
        return {
            "success": True,
            "result": f"Mock TARS result from {tool_name}",
            "arguments": arguments,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_integration_metascript(self, server_name, use_case="general"):
        """Generate metascript that integrates external MCP server"""
        
        if server_name not in self.registered_servers:
            raise ValueError(f"Server '{server_name}' not registered")
        
        server = self.registered_servers[server_name]
        
        metascript = f'''# Auto-generated MCP Integration Metascript
# Server: {server_name}
# Use Case: {use_case}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Metascript Metadata
```yaml
name: "{server_name}_integration"
version: "1.0.0"
type: "mcp-integration"
mcp_server: "{server_name}"
server_url: "{server['url']}"
use_case: "{use_case}"
auto_generated: true
```

## MCP Server Integration
let integrate{server_name.replace("-", "").title()} = fun parameters ->
    async {{
        // Connect to MCP server: {server_name}
        let mcpClient = TarsMcpClient()
        let serverName = "{server_name}"
        
        // Available tools from {server_name}
        {self.generate_tool_bindings(server.get("tools", []))}
        
        // Available resources from {server_name}
        {self.generate_resource_bindings(server.get("resources", []))}
        
        // Execute integration workflow
        let workflow = {{
            name = "{use_case}_workflow"
            steps = [
                {self.generate_workflow_steps(server, use_case)}
            ]
        }}
        
        let! result = mcpClient.ComposeCrossServerWorkflow(workflow)
        
        return {{
            IntegrationType = "MCP"
            ServerName = "{server_name}"
            UseCase = "{use_case}"
            Result = result
            Success = result.Success
            Timestamp = DateTime.UtcNow
        }}
    }}

## Autonomous Execution
let autoExecute() =
    async {{
        let! result = integrate{server_name.replace("-", "").title()} defaultParameters
        
        if result.Success then
            printfn "✅ MCP integration successful: {server_name}"
        else
            printfn "❌ MCP integration failed: {server_name}"
        
        return result
    }}
'''
        
        return metascript
    
    def generate_tool_bindings(self, tools):
        """Generate F# bindings for external tools"""
        bindings = []
        for tool in tools:
            tool_name = tool["name"]
            bindings.append(f'        let {tool_name} = mcpClient.CallExternalTool("{tool_name}", parameters)')
        
        return "\n".join(bindings) if bindings else "        // No tools available"
    
    def generate_resource_bindings(self, resources):
        """Generate F# bindings for external resources"""
        bindings = []
        for resource in resources:
            resource_uri = resource["uri"]
            safe_name = resource_uri.replace("://", "_").replace("/", "_")
            bindings.append(f'        let {safe_name} = mcpClient.AccessExternalResource("{resource_uri}")')
        
        return "\n".join(bindings) if bindings else "        // No resources available"
    
    def generate_workflow_steps(self, server, use_case):
        """Generate workflow steps for integration"""
        # This would generate intelligent workflow steps based on server capabilities
        return '''
                {
                    name = "fetch_data"
                    tool = "example_tool"
                    arguments = parameters
                    server = serverName
                }
        '''
    
    def list_registered_servers(self, detailed=False):
        """List all registered MCP servers"""
        if not self.registered_servers:
            print("📭 No MCP servers registered")
            return
        
        print(f"📋 Registered MCP Servers ({len(self.registered_servers)}):")
        print("=" * 50)
        
        for name, server in self.registered_servers.items():
            print(f"🔗 {name}")
            print(f"   URL: {server['url']}")
            print(f"   Status: {server.get('status', 'unknown')}")
            print(f"   Tools: {len(server.get('tools', []))}")
            print(f"   Resources: {len(server.get('resources', []))}")
            
            if detailed:
                print(f"   Description: {server['info'].get('description', 'N/A')}")
                print(f"   Version: {server['info'].get('version', 'N/A')}")
                print(f"   Registered: {server.get('registered_at', 'N/A')}")
            
            print()

def main():
    """Main function for TARS MCP Client CLI"""
    parser = argparse.ArgumentParser(description="TARS MCP Client")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Register server command
    register_parser = subparsers.add_parser("register", help="Register external MCP server")
    register_parser.add_argument("url", help="MCP server URL")
    register_parser.add_argument("--name", help="Server name")
    register_parser.add_argument("--auto-discover", action="store_true", help="Auto-discover capabilities")
    
    # List servers command
    list_parser = subparsers.add_parser("list", help="List registered servers")
    list_parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    
    # Discover servers command
    discover_parser = subparsers.add_parser("discover", help="Discover MCP servers on network")
    discover_parser.add_argument("--network", default="local", help="Network range to scan")
    discover_parser.add_argument("--auto-register", action="store_true", help="Auto-register discovered servers")
    
    # Call tool command
    call_parser = subparsers.add_parser("call", help="Call tool on external server")
    call_parser.add_argument("tool", help="Tool name")
    call_parser.add_argument("--server", help="Server name")
    call_parser.add_argument("--args", help="Tool arguments (JSON)")
    
    # Generate integration command
    integrate_parser = subparsers.add_parser("integrate", help="Generate integration metascript")
    integrate_parser.add_argument("server", help="Server name")
    integrate_parser.add_argument("--use-case", default="general", help="Use case for integration")
    integrate_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = TarsMcpClient()
    
    async def run_command():
        if args.command == "register":
            await client.register_server(args.url, args.name, args.auto_discover)
        
        elif args.command == "list":
            client.list_registered_servers(args.detailed)
        
        elif args.command == "discover":
            servers = await client.discover_servers_on_network(args.network)
            if args.auto_register:
                for server_url in servers:
                    await client.register_server(server_url)
        
        elif args.command == "call":
            arguments = json.loads(args.args) if args.args else {}
            result = await client.call_external_tool(args.tool, arguments, args.server)
            print(json.dumps(result, indent=2))
        
        elif args.command == "integrate":
            metascript = client.generate_integration_metascript(args.server, args.use_case)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(metascript)
                print(f"📄 Integration metascript saved to: {args.output}")
            else:
                print(metascript)
    
    asyncio.run(run_command())

if __name__ == "__main__":
    main()
