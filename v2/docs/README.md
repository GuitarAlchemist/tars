# TARS v2 Documentation

Welcome to the TARS v2 documentation. This directory contains all project documentation, organized for easy navigation.

## 📚 Documentation Structure

### 🎯 [0_Vision](./0_Vision/)

High-level vision and design philosophy for TARS v2.

### 🚀 [1_Getting_Started](./1_Getting_Started/)

Tutorials, quick starts, and basic usage examples.

### 📊 [2_Architecture](./2_Architecture/)

System architecture, design decisions, and technical specifications.

- **Core Architecture**: Event bus, kernel, agent lifecycle
- **Memory Systems**: Vector stores, knowledge graphs, episodic memory
- **DSL & Metascripts**: Domain-specific language specifications

### 🗺️ [3_Roadmap](./3_Roadmap/)

Implementation plans, phase summaries, and project milestones.

- **Current Plan**: [`implementation_plan.md`](./3_Roadmap/implementation_plan.md)
- **Phase Summaries**: Completed and in-progress phases
- **Research Integration**: How research insights feed into the roadmap

### 🔬 [4_Research](./4_Research/)

Research papers, analysis, and insights that inform TARS v2 design.

- **Memory Systems**: GAM, Graphiti evaluations
- **Agentic Systems**: Multi-agent patterns, speech acts
- **State Space Analysis**: Observability and controllability

### 🧪 [5_Quality](./5_Quality/)

Testing strategies, metrics, and quality assurance.

- **Testing Guide**: [`Testing_Tips.md`](./5_Quality/Testing_Tips.md)
- **Metrics**: [`Competence_Metrics.md`](./5_Quality/Competence_Metrics.md)
- **Grammar Distillation**: [`Grammar_Distillation.md`](./5_Quality/Grammar_Distillation.md)
- **Cargo Cult Defense**: [`Cargo_Cult_Defense.md`](./QA/Cargo_Cult_Defense.md)
- **Ownership Map**: [`Capability_Ownership_Map.md`](./2_Architecture/Capability_Ownership_Map.md)
- **Output Guard**: [`LLM_Output_Guard.md`](./5_Quality/LLM_Output_Guard.md)

### 🔧 [6_Maintenance](./6_Maintenance/)

Troubleshooting, known issues, and operational guides.

- **Troubleshooting**: Common issues and solutions
- **Cargo Cult Analysis**: [`cargo_cult_analysis.md`](./cargo_cult_analysis.md)

### 📖 [7_Reference](./7_Reference/)

API references, configuration guides, and detailed specifications.

## 🔍 Quick Navigation

### For New Contributors

1. Start with [Vision](./0_Vision/)
2. Follow [Getting Started](./1_Getting_Started/)
3. Review [Architecture](./2_Architecture/)

### For Developers

- **Current Work**: [`3_Roadmap/implementation_plan.md`](./3_Roadmap/implementation_plan.md)
- **Testing**: [`5_Quality/Testing_Tips.md`](./5_Quality/Testing_Tips.md)
- **Architecture**: [`2_Architecture/`](./2_Architecture/)

### For Researchers

- **Research Insights**: [`4_Research/`](./4_Research/)
- **Memory Analysis**: [`4_Research/memory_system_evaluation.md`](./4_Research/memory_system_evaluation.md)

## 📝 Document Types

- **`.md`**: Markdown documentation (primary format)
- **`/assets/`**: Images, diagrams, and supporting files
- **`/conversations/`**: Archived design discussions

## 🔄 Documentation Updates

This documentation is actively maintained and reflects the current state of TARS v2. Last major reorganization: 2025-11-29.

---

**Need help?** Check the [Troubleshooting Guide](./6_Maintenance/Troubleshooting/) or review [Known Issues](./6_Maintenance/).
