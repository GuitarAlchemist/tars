# External Tools Landscape for TARS V2

**Date:** November 22, 2025
**Status:** Recommendation
**Context:** Identifying free/freemium tools to augment TARS's capabilities without building everything from scratch.

---

## Executive Summary

To make TARS a true "Super-Agent" while respecting the **"Local First"** philosophy, we prioritize tools that run in Docker or locally. Cloud tools are fallback options only.

**Top 3 Recommendations (Local/Docker):**

1. **Activepieces (Docker)**: Self-hosted automation and tool auth. Replaces Zapier/Arcade for internal workflows.
2. **Obsidian + LocalGPT**: For "Deep Research" on local docs without sending data to Google.
3. **Chroma (Docker/Local)**: The default Vector Store. Runs as a lightweight container or embedded process.

---

## 1. Tool Execution & Authentication (The "Hands")

| Tool | Category | Deployment | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **[Activepieces](https://www.activepieces.com/)** | Automation | **Docker** | **Primary**. Open-source, self-hosted alternative to Zapier. Handles auth for TARS. |
| **[Arcade.dev](https://www.arcade.dev/)** | Tool Auth | Cloud | *Fallback*. Use only if local OAuth handling becomes too complex. |
| **[Pydantic AI Gateway](https://pydantic.dev/)** | API Gateway | **Docker** | Manage LLM API keys and routing locally. |

## 2. Knowledge & Research (The "Brain Extension")

| Tool | Category | Deployment | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **[LocalGPT](https://github.com/PromtEngineer/localGPT)** | Research | **Local** | **Primary**. Chat with your documents locally using Llama 3 / Mistral. |
| **[Obsidian](https://obsidian.md/)** | Knowledge | **Local** | TARS writes research notes here. User uses "Smart Connections" plugin for AI chat. |
| **[NotebookLM](https://notebooklm.google/)** | Research | Cloud | *Fallback*. Use only for public data where privacy is not a concern. |

## 3. Memory & Vector Stores (The "Long-Term Memory")

| Tool | Category | Deployment | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **[Chroma](https://www.trychroma.com/)** | Vector DB | **Docker** | **Primary**. Lightweight, fast, perfect for single-user agents. |
| **[Milvus](https://milvus.io/)** | Vector DB | **Docker** | **Scale**. Use if TARS needs to store >1M code snippets. |
| **[Weaviate](https://weaviate.io/)** | Vector DB | **Docker** | **Hybrid**. Good if we need keyword search + vector search combined. |

## 4. Observability & Debugging (The "Health Monitor")

| Tool | Category | Deployment | TARS Use Case |
| :--- | :--- | :--- | :--- |
| **[Arize Phoenix](https://phoenix.arize.com/)** | Tracing | **Docker** | **Primary**. Self-hosted tracing for LLM apps. Visualizes TARS's thought process. |
| **[LangSmith](https://smith.langchain.com/)** | Tracing | Cloud | *Fallback*. Excellent UI, but requires sending traces to cloud. |

---

## Integration Strategy (Local-First)

1. **The "Docker Sidecar" Pattern**:
    * TARS runs alongside a `docker-compose.yml` stack containing: `chroma`, `activepieces`, and `phoenix`.
    * TARS connects to these via `localhost`.

2. **Local Research Workflow**:
    * Instead of uploading to NotebookLM, TARS indexes docs into **Chroma** and uses a local LLM (via Ollama) to summarize/query them.

3. **Auth Handling**:
    * Try to use **Activepieces** for complex flows.
    * For simple tools (GitHub), just use local Environment Variables (`GITHUB_TOKEN`) stored in `.env`.
