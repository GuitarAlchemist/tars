# MCP Server Landscape & Recommendations for TARS V2

**Date:** November 22, 2025
**Status:** Recommendation
**Context:** Selecting the "Standard Library" of MCP servers to bundle or recommend for TARS v2.

---

## Executive Summary

The MCP ecosystem has exploded with high-quality servers. For TARS v2, we should not "reinvent the wheel" by building custom tools for Git or Filesystem access. Instead, we should **bundle** or **auto-install** the official/community MCP servers.

**Top 3 "Must-Haves":**

1. **Filesystem MCP Server**: For reading/writing code.
2. **GitHub MCP Server**: For Issues, PRs, and Repo management.
3. **PostgreSQL MCP Server**: For TARS's own memory/knowledge graph.

---

## Comprehensive MCP Server List

### 1. Core Development (The "TARS Kernel")

These servers are non-negotiable for an Autonomous Software Engineer.

| Server Name | Source | Capabilities | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **Filesystem** | Official | Read/Write files, List dirs, Search | The primary way TARS interacts with the user's codebase. |
| **GitHub** | Official | Issues, PRs, Commits, Search | Managing the "Work" (Tickets) and "Delivery" (PRs). |
| **Git** | Community | Local git operations (checkout, branch, log) | Navigating the local history and managing branches. |
| **Terminal** | Community | Execute shell commands | Running builds (`dotnet build`), tests, and scripts. |

### 2. Knowledge & Research (The "TARS Cortex")

These servers give TARS access to the world.

| Server Name | Source | Capabilities | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **Brave Search** | Official | Web search, Local search | Researching libraries, fixing errors, finding docs. |
| **Microsoft Learn** | Microsoft | Search MS Docs, Azure Docs | Deep knowledge of .NET/Azure ecosystems. |
| **Fetch** | Official | HTTP GET/POST, HTML parsing | Reading documentation pages directly. |
| **Context7** | Community | API Docs for libraries | Bridging the gap for libraries with recent API changes. |
| **Memory (mem0)** | Community | Long-term memory storage | Storing user preferences and project context. |

### 3. Specialized Tools (The "TARS Hands")

These servers extend TARS into specific domains.

| Server Name | Source | Capabilities | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **PostgreSQL** | Official | SQL Query, Schema Inspection | Managing TARS's internal Knowledge Graph (Graphiti). |
| **Playwright** | Community | Browser automation, UI testing | Verifying web apps, taking screenshots of UI changes. |
| **Puppeteer** | Community | Browser automation | Alternative to Playwright for frontend testing. |
| **Azure AI Foundry**| Microsoft | Deploy models, manage resources | Managing cloud infrastructure for the user. |
| **Slack** | Community | Send messages, read channels | Notifying the team of progress or asking questions. |
| **Snyk** | Official | Security scanning | Checking code for vulnerabilities before committing. |
| **Serena** | Community | LSP-like suggestions | Smart code completions and insights. |

---

## Integration Strategy for TARS V2

### 1. The "Bundled" Set

TARS v2 should ship with a `tars-mcp.json` configuration that pre-configures the **Filesystem** and **Git** servers. These should "just work" out of the box.

### 2. The "On-Demand" Set

TARS should have a command (e.g., `/tars install mcp github`) that automates the setup of complex servers like GitHub (handling auth, config, etc.).

### 3. The "Internal" Set

TARS will run its *own* internal MCP server (`Tars.Memory.Server`) to expose its Vector Store and Knowledge Graph to other tools (like the user's IDE).

---

## Resources & Directories

For a constantly updated list of available MCP servers, refer to these community-maintained directories:

* **[Awesome MCP Servers (wong2)](https://github.com/wong2/awesome-mcp-servers)**: The most popular curated list of MCP servers.
* **[Awesome MCP Servers (punkpeye)](https://github.com/punkpeye/awesome-mcp-servers)**: Another extensive collection, including coding agents.
* **[MCPServers.org](https://mcpservers.org)**: A searchable web directory of MCP servers.
* **[Microsoft MCP Blog](https://developer.microsoft.com/blog/10-microsoft-mcp-servers-to-accelerate-your-development-workflow)**: Official Microsoft announcements.
