# 🔌 TARS MCP Server Setup Guide

## What is MCP?

The **Model Context Protocol (MCP)** allows AI assistants like ChatGPT to connect to external tools and services. TARS implements an MCP server that exposes 124+ tools to any MCP-compatible client.

## Quick Start

### 1. Start TARS MCP Server

```powershell
cd C:\Users\spare\source\repos\tars\v2
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp server
```

The server will start and listen on **stdin/stdout** for JSON-RPC messages.

### 2. Configure ChatGPT Desktop

**For ChatGPT Desktop (Windows):**

1. Locate your ChatGPT config directory:
   - Usually: `%APPDATA%\ChatGPT\` or `%APPDATA%\OpenAI\ChatGPT\`
   
2. Create or edit `mcp_settings.json`:

```json
{
  "mcpServers": {
    "tars": {
      "command": "dotnet",
      "args": [
        "run",
        "--project",
        "C:\\Users\\spare\\source\\repos\\tars\\v2\\src\\Tars.Interface.Cli\\Tars.Interface.Cli.fsproj",
        "--",
        "mcp",
        "server"
      ],
      "env": {
        "GRAPHITI_URL": "http://localhost:8001",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Important:** Note the double backslashes (`\\`) in Windows paths - this is required for JSON!

3. Restart ChatGPT Desktop

### 3. Verify Connection

Once connected, ChatGPT should have access to TARS tools. You can verify by asking ChatGPT:

> "What TARS tools are available?"

## Available Tools (124+)

When connected, ChatGPT can use:

### **Web & Research**
- `fetch_webpage` - Download and parse web pages
- `search_web` - Semantic web search
- `fetch_wikipedia` - Get Wikipedia summaries
- `fetch_arxiv` - Search academic papers

### **Code Analysis**
- `analyze_file_complexity` - Measure code metrics
- `find_code_smells` - Detect anti-patterns
- `ast_analysis` - Parse code structure

### **Knowledge Graph**
- `graph_add_node` - Add entities
- `graph_add_edge` - Create relationships
- `graph_query` - Search the graph
- `graph_find_contradictions` - Logic validation

### **File Operations**
- `read_file` - View file contents
- `write_file` - Create/update files
- `list_directory` - Browse folders
- `search_files` - Find files by pattern

### **Reasoning**
- `run_got` - Graph of Thoughts reasoning
- `run_wot` - Workflow of Thoughts
- `run_tot` - Tree of Thoughts
- `run_cot` - Chain of Thought

### **Memory (Graphiti)**
- `search_memory` - Semantic memory search
- `save_memory` - Store episodic memories

### **And more...**
- Docker operations
- Git operations
- Process management
- Environment variables
- System diagnostics

## Configuration Files

### `mcp_config.json` (TARS Internal)
Located: `C:\Users\spare\source\repos\tars\v2\mcp_config.json`

This configures which **other MCP servers** TARS connects to as a client:

```json
{
  "Servers": [
    {
      "Name": "github",
      "Command": "npx",
      "Arguments": ["-y", "@modelcontextprotocol/server-github"],
      "Environment": {}
    }
  ]
}
```

**✅ FIXED:** Paths now use double backslashes (`\\\\`) for JSON escaping.

### `mcp-config.json` (For ChatGPT)
Located: `C:\Users\spare\source\repos\tars\v2\mcp-config.json`

This is the **template** for ChatGPT to connect to TARS.

## Troubleshooting

### Error: "JSON value could not be converted"

**Cause:** Unescaped backslashes in Windows paths.

**Fix:** Use double backslashes:
- ❌ `"c:\Users\spare\file.py"`
- ✅ `"c:\\\\Users\\\\spare\\\\file.py"`

### Error: "Graphiti not available"

**Cause:** Graphiti knowledge graph service not running.

**Fix:**
```powershell
# Start Graphiti (if you have it configured)
docker-compose up -d graphiti
```

Or set environment variable:
```powershell
$env:GRAPHITI_URL = "http://localhost:8001"
```

### MCP Server Won't Start

**Check:**
1. .NET 10 SDK is installed: `dotnet --version`
2. Project builds: `dotnet build src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj`
3. All dependencies are restored

## Advanced: Running as Background Service

**Option 1: PowerShell Background Job**
```powershell
Start-Job -ScriptBlock {
    cd C:\Users\spare\source\repos\tars\v2
    dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp server
}
```

**Option 2: Windows Service**
Use NSSM (Non-Sucking Service Manager) to install as a Windows service.

## Testing MCP Server Directly

You can test the MCP server manually using stdin/stdout:

```powershell
cd C:\Users\spare\source\repos\tars\v2
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp server
```

Then send JSON-RPC messages:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

## Security Notes

⚠️ **The MCP server has access to:**
- File system operations
- Web requests
- Process execution
- Docker operations

Only connect from trusted clients!

## Next Steps

1. ✅ Fix JSON escaping in `mcp_config.json`
2. ✅ Start TARS MCP server
3. ✅ Configure ChatGPT Desktop
4. 🎉 Use TARS tools from ChatGPT!

## Support

For issues, check:
- TARS logs (stderr)
- ChatGPT Desktop logs
- `mcp_config.json` syntax

---

**Status:** ✅ Ready to use!
**Fixed:** JSON path escaping issue resolved.
