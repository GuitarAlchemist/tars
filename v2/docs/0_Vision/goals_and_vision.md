# TARS V2: Goals and Vision

**"The bridge between human intent and autonomous execution."**

---

## 1. The Core Mission: Recursive Intelligence

TARS v2 is not just a "coding assistant"; it is an **Autonomous Software Engineer** capable of recursive self-improvement.

* **The "Ouroboros" Loop**: TARS must be able to read, understand, modify, build, and test its own source code. Every version of TARS should be capable of creating a better version of itself.
* **Symbiosis**: TARS amplifies human potential by handling the cognitive load of implementation, verification, and maintenance, allowing the human to focus on *Architecture* and *Intent*.

---

## 2. Dual-Mode Operation (The "Prime Directive")

TARS must seamlessly switch between two distinct operational modes without configuration changes:

### Mode A: The "Consultant" (External Repos)

* **Goal**: Work on *any* existing local repository (e.g., `C:\Users\spare\source\repos\ga`).
* **Behavior**: TARS acts as a senior engineer joining the team. It respects existing patterns, reads the `README`, understands the tech stack (C#, Python, Rust, etc.), and delivers value immediately.
* **Constraint**: Zero-config. Point TARS at a folder, and it starts working.

### Mode B: The "Architect" (Self-Evolution)

* **Goal**: Work on *itself* (`C:\Users\spare\source\repos\tars`).
* **Behavior**: TARS takes ownership of its own architecture. It runs the "Ouroboros Loop" to optimize its kernel, expand its skills, and refine its cognitive strategies.
* **Constraint**: Safety. Self-modification must be verified by rigorous automated testing before being committed.

---

## 3. Architectural Pillars

To achieve this, TARS v2 is built on three non-negotiable pillars:

1. **The Micro-Kernel (F#)**:
    * A lightweight, type-safe core that handles the "Lifecycle of Thought."
    * It does not "do" the work; it *orchestrates* the Agents that do the work.

2. **The Polyglot Runtime (MCP + AutoGen)**:
    * **MCP (Model Context Protocol)**: The standard interface for tools (File System, Git, Search, Terminals).
    * **AutoGen**: The multi-agent orchestration layer (likely in Python) that allows specialized agents (Coder, Reviewer, Planner) to collaborate.

3. **The Knowledge Graph (Graphiti)**:
    * TARS does not just "search text"; it builds a *mental model* of the codebase.
    * It understands dependencies, relationships, and architectural patterns, stored in a temporal graph.

---

## 4. The "Zero Friction" User Experience

* **No YAML Hell**: TARS configures itself by analyzing the environment.
* **Natural Language Interface**: You don't learn TARS commands; TARS learns your communication style.
* **Transparent Reasoning**: TARS always explains *why* it is doing something before it does it (unless in "Turbo Mode").

---

## 5. Success Criteria (V2)

We will know V2 is successful when:

1. [ ] TARS can clone a random GitHub repo and fix a GitHub Issue without human intervention.
2. [ ] TARS can write a PR to its own codebase that **improves its own intelligence** (e.g., implements a new cognitive strategy, reduces reasoning errors, or solves a task it previously failed), pass all tests, and merge it.
3. [ ] The user can switch between these two tasks instantly.
