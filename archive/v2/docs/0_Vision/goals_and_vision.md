# TARS V2: Goals and Vision

**"The bridge between human intent and autonomous execution."**

---

## 1. The Core Mission: Recursive Superintelligence

TARS v2 is not just a "coding assistant"; it is a **Topologically Governed Autonomous Engineer** designed for safe, recursive self-improvement.

* **The "Ouroboros" Loop**: TARS must be able to read, understand, modify, build, and test its own source code. Every version of TARS should be capable of creating a better version of itself.
* **Meta-Learning ("Learning to Learn")**: TARS is not static. It maintains a long-term memory of *strategies*, not just code.
  * **Strategy Optimization**: It analyzes its own execution traces to identify which reasoning patterns led to success and which led to dead ends.
  * **Prompt Evolution**: TARS can dynamically rewrite its own system prompts and agent personas to better suit the task at hand.
  * **Tool Synthesis**: If TARS finds itself repeatedly performing a complex sequence of manual steps, it can write a new Tool (in F#) to automate that sequence, effectively "growing" its own API.
* **Topological Safety**: Unlike standard AI agents, TARS operates within a **K-Theoretic framework**. It is mathematically constrained to preserve resources (K0) and prevent infinite causal loops (K1), ensuring that "superintelligence" does not become "super-instability."
* **Symbiosis**: TARS amplifies human potential by handling the cognitive load of implementation, verification, and maintenance, allowing the human to focus on *Architecture* and *Intent*.

---

## 2. Dual-Mode Operation (The "Prime Directive")

TARS must seamlessly switch between two distinct operational modes without configuration changes:

### Mode A: The "Consultant" (External Repos)

* **Goal**: Work on *any* existing local repository.
* **Behavior**: TARS acts as a senior engineer joining the team. It respects existing patterns, reads the `README`, understands the tech stack, and delivers value immediately.
* **Constraint**: Zero-config. Point TARS at a folder, and it starts working.

### Mode B: The "Architect" (Self-Evolution)

* **Goal**: Work on *itself* (`Tars.Core`, `Tars.Cortex`, etc.).
* **Behavior**: TARS takes ownership of its own architecture. It runs the "Ouroboros Loop" to optimize its kernel, expand its skills, and refine its cognitive strategies.
* **Constraint**: **Epistemic Safety**. Self-modification is governed by an Epistemic Governor that validates not just the *code*, but the *reasoning* behind the change.

---

## 3. Architectural Pillars

TARS v2 is built on four non-negotiable pillars:

1. **The Micro-Kernel (F#)**:
    * A lightweight, type-safe core that handles the "Lifecycle of Thought."
    * It does not "do" the work; it *orchestrates* the Agents that do the work.

2. **Cognitive Architecture (The Nervous System)**:
    * **K-Theory Governance**:
        * **K0 (Conservation)**: Every thought and action has a cost. Resources (Tokens, Time, Compute) are conserved quantities.
        * **K1 (Topology)**: The causal graph of agent interactions is monitored for cycles. TARS cannot get stuck in infinite loops because the topology forbids it.
    * **Epistemic Governor**: A meta-agent that monitors the *confidence* and *knowledge* of the system, halting hallucinations before they become code.
    * **Evolutionary Engine**: The component responsible for "Learning to Learn." It uses genetic algorithms or reinforcement learning strategies to optimize the system's "Metascripts" (workflows) over time.

3. **The Knowledge Graph (Graphiti)**:
    * TARS does not just "search text"; it builds a *mental model* of the codebase.
    * It understands dependencies, relationships, and architectural patterns, stored in a temporal graph.

4. **Agentic Interfaces (The Face)**:
    * **Generative UI**: TARS does not just output text. It generates dynamic, ephemeral user interfaces (using React/Next.js) to present complex data, visualize architectural changes, or solicit high-bandwidth human feedback.
    * **The "Holodeck"**: A sandbox where TARS can visualize its own internal state and the state of the project for the user.

---

## 4. The "Zero Friction" User Experience

* **No YAML Hell**: TARS configures itself by analyzing the environment.
* **Natural Language Interface**: You don't learn TARS commands; TARS learns your communication style.
* **Transparent Reasoning**: TARS always explains *why* it is doing something before it does it (unless in "Turbo Mode").

---

## 5. Success Criteria (V2)

We will know V2 is successful when:

1. [ ] **External**: TARS can clone a random GitHub repo and fix a GitHub Issue without human intervention.
2. [ ] **Internal**: TARS can write a PR to its own codebase that **improves its own intelligence** (e.g., implements a new cognitive strategy), pass all tests (including K-Theory invariant checks), and merge it.
3. [ ] **Meta-Learning**: TARS identifies a recurring inefficiency in its own workflow and autonomously **creates a new Metascript** or refactors an existing one to solve it.
4. [ ] **Interface**: TARS can spontaneously generate a UI tool to help the user debug a complex issue.
5. [ ] **Safety**: The system runs for 24 hours in a self-improvement loop without crashing, exhausting resources, or entering a causal loop.
