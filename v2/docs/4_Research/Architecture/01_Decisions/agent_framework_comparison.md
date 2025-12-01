# Agent Framework Comparison for TARS V2

**Date:** November 22, 2025
**Status:** Recommendation
**Context:** Re-evaluating the choice of AutoGen for the TARS v2 Agent Runtime.

---

## Executive Summary

After reviewing the 2025 landscape, **AutoGen remains the best primary choice** for TARS v2 due to its unique focus on *conversational code execution*. However, **LangGraph** offers better control for deterministic workflows, and **Agent0** offers a compelling vision for self-evolution.

**Recommendation:**

1. **Core Runtime**: **AutoGen** (Python) for the dynamic "Dev Team" (Coder, Reviewer, User Proxy).
2. **Orchestration**: **F# Micro-Kernel** (Custom) to manage the lifecycle, replacing the need for LangGraph's state management.
3. **Evolution**: Study **Agent0**'s curriculum-based learning for the `Tars.Cortex.Evolution` component.

---

## Detailed Comparison

| Feature | **AutoGen** (Microsoft) | **LangGraph** (LangChain) | **CrewAI** | **Semantic Kernel** (Microsoft) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Paradigm** | **Conversational** (Agents talk to solve tasks) | **Graph-based** (Nodes & Edges) | **Role-based** (Process & Tasks) | **Plugin-based** (Functions & Skills) |
| **Code Execution** | **Native & First-Class** (Docker/Local) | Possible via Tools (Manual setup) | Possible via Tools | Possible via Plugins |
| **Multi-Agent** | Excellent (Group Chat Manager) | Good (Multi-Actor Graph) | Good (Sequential/Hierarchical) | Weak (Orchestrator pattern) |
| **Self-Correction** | **High** (Agents auto-reply to errors) | Medium (Needs explicit edges) | Medium | Low |
| **Complexity** | High (Hard to control flow) | High (Complex graph definition) | Low (Easy to start) | Medium |
| **TARS Fit** | **9/10** (Matches "Dev Team" model) | **7/10** (Good for rigid flows) | **6/10** (Better for content/research) | **5/10** (Better for Copilots) |

### Why AutoGen Wins for TARS

1. **The "Code-Execute-Repair" Loop**: AutoGen is the only framework where "writing code" and "executing code" are deeply integrated into the conversation loop. If a script fails, the UserProxy automatically feeds the error back to the Assistant, who fixes it. This is the *heart* of TARS.
2. **Microsoft Ecosystem**: Aligns with our .NET Core / F# stack (even though AutoGen is Python-first, the .NET version is catching up, or we can use the Python bridge).
3. **Dynamic Group Chat**: TARS needs to spawn agents dynamically (e.g., "I need a Database Expert"). AutoGen's GroupChatManager handles this naturally.

### The "Agent0" Wildcard

* **What is it?**: A framework for agents that learn from zero data using a "Curriculum Agent" vs. "Executor Agent" loop.
* **Relevance**: This is exactly the **Ouroboros Loop**.
* **Strategy**: We should *implement* the Agent0 pattern *using* AutoGen agents, rather than switching frameworks entirely.

### The .NET Ecosystem: LangChain.NET vs. Semantic Kernel

The user specifically asked about **LangChain.NET (tryAGI/LangChain)**. Here is the breakdown for a .NET-native architecture:

| Feature | **LangChain.NET** (Community) | **Semantic Kernel** (Microsoft) | **AutoGen.NET** (Microsoft) |
| :--- | :--- | :--- | :--- |
| **Maturity** | High (Active Community Port) | **Enterprise** (Production Ready) | Emerging (Catching up to Python) |
| **Philosophy** | "Port Python LangChain to C#" | "Native .NET AI Orchestration" | "Multi-Agent Conversation" |
| **Support** | Community-driven | **Microsoft First-Party** | Microsoft Research |
| **Best For** | Developers who *love* LangChain abstractions (Chains, LCEL). | Enterprise apps needing stability, DI, and Azure integration. | Complex multi-agent workflows in .NET. |

**Verdict for TARS:**

* **LangChain.NET** is excellent, but it locks us into the "LangChain Way" (Chains, Memory classes) which can be rigid.
* **Semantic Kernel** is the "Standard Library" for .NET AI. If we need a native .NET kernel, we should use SK for its stability and integration with `Microsoft.Extensions.DependencyInjection`.
* **AutoGen.NET** is the closest fit for our "Agentic" vision, but the Python version is still ahead in features (like Code Execution).

**Decision:** We will use **Semantic Kernel** for the F# Micro-Kernel's internal LLM calls (if needed), but rely on **AutoGen (Python)** via a bridge for the heavy multi-agent lifting. We avoid LangChain.NET to prevent "abstraction leak" from a different ecosystem.

---

## Conclusion

We should stick with **AutoGen** as the engine for the "Consultant" and "Architect" modes. Its ability to handle the messy, iterative loop of coding (write -> run -> error -> fix) is unmatched.

* **LangGraph** is too rigid for creative coding.
* **CrewAI** is too high-level (better for "write a blog post").
* **Semantic Kernel** is too low-level (better for "add AI to my app").
* **LangChain.NET** is a great port, but Semantic Kernel is the native choice for .NET.
