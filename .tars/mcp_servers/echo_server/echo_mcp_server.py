#!/usr/bin/env python3
"""
Simple Echo MCP Server for TARS Demo
"""

import asyncio
import json
from datetime import datetime
from aiohttp import web

class EchoMcpServer:
    def __init__(self):
        self.usage_stats = {
            "requests_handled": 0,
            "tools_called": {},
            "start_time": datetime.now().isoformat()
        }
    
    async def handle_tool_call(self, tool_name, arguments):
        """Handle MCP tool calls"""
        
        self.usage_stats["requests_handled"] += 1
        self.usage_stats["tools_called"][tool_name] = self.usage_stats["tools_called"].get(tool_name, 0) + 1
        
        if tool_name == "echo":
            return {
                "content": [{"type": "text", "text": f"Echo: {arguments['text']}"}]
            }
        elif tool_name == "reverse":
            return {
                "content": [{"type": "text", "text": f"Reversed: {arguments['text'][::-1]}"}]
            }
        elif tool_name == "uppercase":
            return {
                "content": [{"type": "text", "text": f"Uppercase: {arguments['text'].upper()}"}]
            }
        elif tool_name == "analyze":
            text = arguments['text']
            analysis = {
                "length": len(text),
                "words": len(text.split()),
                "uppercase_count": sum(1 for c in text if c.isupper()),
                "lowercase_count": sum(1 for c in text if c.islower())
            }
            return {
                "content": [{"type": "text", "text": f"Analysis: {json.dumps(analysis)}"}]
            }
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def get_server_info(self):
        """Get server information"""
        return {
            "name": "echo-mcp-server",
            "version": "1.0.0",
            "description": "Simple Echo MCP Server for TARS Demo",
            "tools": [
                {"name": "echo", "description": "Echo back text"},
                {"name": "reverse", "description": "Reverse text"},
                {"name": "uppercase", "description": "Convert to uppercase"},
                {"name": "analyze", "description": "Analyze text properties"}
            ],
            "resources": [
                {"uri": "echo://status", "description": "Server status"},
                {"uri": "echo://stats", "description": "Usage statistics"}
            ]
        }

async def run_server():
    server = EchoMcpServer()
    
    async def handle_info(request):
        return web.json_response(server.get_server_info())
    
    async def handle_tool_call(request):
        data = await request.json()
        try:
            result = await server.handle_tool_call(data["tool"], data.get("arguments", {}))
            return web.json_response({"success": True, "result": result})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)
    
    async def handle_resource(request):
        uri = request.query.get("uri")
        if uri == "echo://status":
            return web.json_response({"success": True, "result": {"status": "running", "timestamp": datetime.now().isoformat()}})
        elif uri == "echo://stats":
            return web.json_response({"success": True, "result": server.usage_stats})
        else:
            return web.json_response({"success": False, "error": "Unknown resource"}, status=404)
    
    app = web.Application()
    app.router.add_get("/mcp/info", handle_info)
    app.router.add_post("/mcp/tools/call", handle_tool_call)
    app.router.add_get("/mcp/resources/read", handle_resource)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 3001)
    await site.start()
    
    print("✅ Echo MCP Server is running on http://localhost:3001")
    return runner

if __name__ == "__main__":
    async def main():
        runner = await run_server()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await runner.cleanup()
    
    asyncio.run(main())
