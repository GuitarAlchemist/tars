# TARS MCP Server Configuration

This document describes the Model Context Protocol (MCP) server configuration for TARS, replicating common setups used in WebStorm and other development environments.

## Overview

MCP servers provide TARS with access to external tools, APIs, and services. The configuration includes:

- **Development Tools**: File system, Git, GitHub/GitLab integration
- **Database Access**: SQLite, PostgreSQL support
- **Web & API Tools**: HTTP requests, web scraping, browser automation
- **Search & Knowledge**: Web search, Wikipedia access
- **Productivity**: Slack integration, persistent memory
- **System Tools**: Shell commands, Docker management

## Quick Setup

### 1. Install MCP Servers

**Windows (PowerShell):**
```powershell
.\scripts\setup-mcp-servers.ps1
```

**Linux/macOS:**
```bash
./scripts/setup-mcp-servers.sh
```

### 2. Configure Environment

Copy and edit the environment file:
```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

### 3. Test Configuration

```powershell
.\scripts\test-mcp-servers.ps1
```

## Server Descriptions

### Core TARS Servers

| Server | Description | Port |
|--------|-------------|------|
| `tars-default` | Main TARS MCP endpoint | 8999 |
| `augment-local` | Local Augment integration | 9000 |

### Development Tools

| Server | Package | Description |
|--------|---------|-------------|
| `filesystem` | `@modelcontextprotocol/server-filesystem` | File operations in current directory |
| `git` | `mcp-server-git` | Git repository operations |
| `github` | `@modelcontextprotocol/server-github` | GitHub API integration |
| `gitlab` | `@modelcontextprotocol/server-gitlab` | GitLab API integration |
| `shell` | `mcp-server-shell` | Shell command execution |
| `docker` | `mcp-server-docker` | Docker container management |

### Database Servers

| Server | Package | Description |
|--------|---------|-------------|
| `sqlite` | `@modelcontextprotocol/server-sqlite` | SQLite database operations |
| `postgres` | `@modelcontextprotocol/server-postgres` | PostgreSQL database access |

### Web & API Tools

| Server | Package | Description |
|--------|---------|-------------|
| `fetch` | `@modelcontextprotocol/server-fetch` | HTTP requests and web scraping |
| `puppeteer` | `@modelcontextprotocol/server-puppeteer` | Browser automation |

### Search & Knowledge

| Server | Package | Description |
|--------|---------|-------------|
| `brave-search` | `@modelcontextprotocol/server-brave-search` | Web search via Brave API |
| `wikipedia` | `@modelcontextprotocol/server-wikipedia` | Wikipedia article access |

### Productivity Tools

| Server | Package | Description |
|--------|---------|-------------|
| `slack` | `@modelcontextprotocol/server-slack` | Slack workspace integration |
| `memory` | `@modelcontextprotocol/server-memory` | Persistent conversation memory |

## Required API Keys

### GitHub Integration
- **Token**: Personal Access Token from https://github.com/settings/tokens
- **Scopes**: `repo`, `read:user`, `read:org`
- **Environment**: `GITHUB_TOKEN`

### GitLab Integration
- **Token**: Personal Access Token from GitLab profile settings
- **Scopes**: `api`, `read_user`, `read_repository`
- **Environment**: `GITLAB_TOKEN`

### Brave Search
- **API Key**: From https://api.search.brave.com/app/keys
- **Environment**: `BRAVE_API_KEY`

### Slack Integration
- **Bot Token**: From Slack App OAuth & Permissions
- **Format**: `xoxb-...`
- **Environment**: `SLACK_BOT_TOKEN`

### Database Connections
- **PostgreSQL**: Connection string format
- **Example**: `postgresql://user:pass@localhost:5432/db`
- **Environment**: `POSTGRES_URL`

## Usage Examples

### File Operations
```bash
# List files in current directory
tars mcp call filesystem list_files

# Read a specific file
tars mcp call filesystem read_file --path "src/main.fs"
```

### Git Operations
```bash
# Get repository status
tars mcp call git status

# View commit history
tars mcp call git log --limit 10
```

### Web Search
```bash
# Search the web
tars mcp call brave-search search --query "F# functional programming"

# Get Wikipedia article
tars mcp call wikipedia get_article --title "Functional programming"
```

### Database Queries
```bash
# Execute SQLite query
tars mcp call sqlite query --sql "SELECT * FROM users LIMIT 10"

# PostgreSQL query
tars mcp call postgres query --sql "SELECT version()"
```

## Troubleshooting

### Common Issues

1. **Node.js servers not found**
   - Ensure Node.js 18+ is installed
   - Run setup script to pre-install packages

2. **Python servers not working**
   - Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Ensure `uvx` is in PATH

3. **API authentication failures**
   - Check API keys in `config/.env`
   - Verify token permissions and scopes

4. **Database connection errors**
   - Ensure database servers are running
   - Check connection strings in environment

### Testing Individual Servers

```powershell
# Test specific server
npx -y @modelcontextprotocol/server-filesystem . --help

# Test with environment
$env:GITHUB_TOKEN="your_token"
npx -y @modelcontextprotocol/server-github --help
```

### Debugging

Enable debug logging:
```bash
export DEBUG=mcp:*
# or in PowerShell
$env:DEBUG="mcp:*"
```

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Permissions**: Use minimal required scopes for tokens
- **Network**: Consider firewall rules for MCP server ports
- **Shell Access**: Be cautious with shell and Docker servers

## Advanced Configuration

### Custom Server Configuration

Add custom servers to `config/mcp-servers.yaml`:

```yaml
servers:
  - name: custom-api
    command: npx
    args: ["-y", "your-custom-mcp-server"]
    description: "Custom API integration"
    env:
      API_KEY: "${CUSTOM_API_KEY}"
```

### Environment-Specific Configs

Create environment-specific configurations:
- `config/mcp-servers.dev.yaml`
- `config/mcp-servers.prod.yaml`

### Load Balancing

For high-availability setups, configure multiple instances:

```yaml
servers:
  - name: tars-primary
    url: http://localhost:8999/
  - name: tars-secondary
    url: http://localhost:9001/
```

## Integration with TARS CLI

The MCP servers integrate seamlessly with TARS CLI:

```bash
# List available MCP servers
dotnet run --project src/TarsEngine.FSharp.Cli -- mcp list

# Call MCP server function
dotnet run --project src/TarsEngine.FSharp.Cli -- mcp call server_name function_name --args

# Interactive MCP session
dotnet run --project src/TarsEngine.FSharp.Cli -- mcp interactive
```
