# 🎯 Quick Start: Connect ChatGPT to TARS

## ✅ Issue Fixed!

**Problem:** JSON path escaping error  
**Solution:** Fixed `mcp_config.json` with proper `\\` escaping

## 🚀 How to Connect ChatGPT Desktop to TARS

### Step 1: Copy This Config

Find your ChatGPT Desktop config file (usually in `%APPDATA%\ChatGPT\` or `%APPDATA%\OpenAI\ChatGPT\`).

Create or edit `mcp_settings.json` (or `claude_desktop_config.json` if using Claude):

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
      ]
    }
  }
}
```

⚠️ **Important:** Use double backslashes `\\` in Windows paths!

### Step 2: Restart ChatGPT Desktop

Close and reopen ChatGPT Desktop. It will automatically start the TARS MCP server.

### Step 3: Verify

Ask ChatGPT:
> "What tools are available from TARS?"

You should see 124+ tools including:
- Web scraping (`fetch_webpage`)
- Code analysis (`analyze_file_complexity`)
- Knowledge graph (`graph_query`)
- File operations (`read_file`, `write_file`)
- And many more!

## 🧪 Manual Testing

You can also start the server manually:

```powershell
cd C:\Users\spare\source\repos\tars\v2
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- mcp server
```

You should see:
```
[TARS MCP] TARS MCP Server Interface Started (Agentic AI Mode)
```

## 📋 What's Available?

Once connected, ChatGPT can use TARS to:

✅ **Browse the web** - Fetch and analyze web pages  
✅ **Search knowledge** - Wikipedia, arXiv, semantic search  
✅ **Manage files** - Read, write, list directories  
✅ **Run code analysis** - Complexity, code smells, AST parsing  
✅ **Query knowledge graphs** - Add nodes, create relationships  
✅ **Advanced reasoning** - GoT, WoT, ToT algorithms  
✅ **Manage memory** - Store and search episodic memories  

## 🔧 Configuration Files

### `mcp_config.json` (TARS Internal)
Location: `C:\Users\spare\source\repos\tars\v2\mcp_config.json`

Currently set to:
```json
{
  "Servers": []
}
```

This means TARS only serves its own tools (no external MCP servers).

To add external MCP servers (like GitHub), edit this file:
```json
{
  "Servers": [
    {
      "Name": "github",
      "Command": "npx",
      "Arguments": ["-y", "@modelcontextprotocol/server-github"],
      "Environment": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"
      }
    }
  ]
}
```

##Status

✅ **JSON escaping issue:** FIXED  
✅ **MCP server:** WORKING  
✅ **Configuration:** READY  

---

**Next:** Copy the config above into ChatGPT Desktop and restart!
