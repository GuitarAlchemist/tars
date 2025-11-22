---
description: TARS Product Planning Rules for Agent OS
globs:
alwaysApply: false
version: 4.0
encoding: UTF-8
---

# TARS Product Planning Rules

## Overview

Generate TARS-specific product documentation: mission, tech-stack and roadmap files for autonomous AI agent consumption.

## Step 1: Gather TARS Context

Collect information about the TARS enhancement or feature being planned:

### Required Information
1. **Enhancement objective** - What autonomous capability is being added/improved
2. **TARS components affected** - Which modules, agents, or metascripts will be modified
3. **Performance targets** - Expected improvements (e.g., 184M+ searches/second)
4. **Integration points** - How this fits with existing TARS architecture
5. **Autonomous behavior** - How this enhances TARS self-improvement capabilities

### Data Sources
- Current TARS codebase analysis
- Existing metascripts and autonomous workflows
- Performance benchmarks and targets
- User requirements and memories

## Step 2: Create TARS Documentation Structure

Create the following structure for TARS product planning:

```
.agent-os/
└── product/
    ├── tars-mission.md          # TARS enhancement vision and purpose
    ├── tars-mission-lite.md     # Condensed mission for AI context
    ├── tars-tech-stack.md       # TARS technical architecture
    ├── tars-roadmap.md          # TARS development phases
    └── autonomous-objectives.md  # Autonomous agent goals and metrics
```

## Step 3: Create tars-mission.md

### Template Structure

```markdown
# TARS Enhancement Mission

## Enhancement Pitch
[ENHANCEMENT_NAME] enhances TARS autonomous capabilities by [CAPABILITY_DESCRIPTION] to achieve [PERFORMANCE_TARGET] with [AUTONOMOUS_BEHAVIOR].

## Target Users
- **TARS Autonomous Agents**: Enhanced reasoning and self-improvement capabilities
- **TARS Users**: Improved performance and functionality
- **Development Team**: Better autonomous development workflows

## The Enhancement Objective
### [OBJECTIVE_TITLE]
[OBJECTIVE_DESCRIPTION]. Target performance: [QUANTIFIABLE_TARGET].

**Implementation Approach:** [TECHNICAL_APPROACH]

## Key Differentiators
- **Autonomous Enhancement**: Unlike manual improvements, TARS self-improves through [METHOD]
- **CUDA Acceleration**: Real GPU performance gains with [SPECIFIC_METRICS]
- **Metascript Integration**: Seamless integration with FLUX metascript system

## Core Capabilities
- **[CAPABILITY_NAME]:** [AUTONOMOUS_BENEFIT_DESCRIPTION]
- **Performance Optimization:** [SPEED_IMPROVEMENT_DESCRIPTION]
- **Self-Improvement:** [AUTONOMOUS_LEARNING_DESCRIPTION]
```

## Step 4: Create tars-tech-stack.md

### TARS-Specific Technical Stack

```markdown
# TARS Technical Stack

## Core Architecture
- **Primary Language**: F# 9.0+ (functional logic, DSL, reasoning)
- **Infrastructure**: C# 12+ (.NET 9, CLI, integrations)
- **Metascript Language**: FLUX (multi-modal with Wolfram/Julia support)
- **Agent Framework**: Multi-agent with specialized teams

## Performance Stack
- **GPU Acceleration**: CUDA (WSL compilation required)
- **Vector Operations**: Custom CUDA kernels targeting 184M+ searches/second
- **Memory Systems**: Episodic, semantic, and procedural memory
- **Knowledge Storage**: ChromaDB (Docker) or Custom CUDA Vector Store

## Development Standards
- **Quality**: Zero tolerance for simulations/placeholders
- **Testing**: 80% coverage minimum, TDD preferred
- **Architecture**: Clean Architecture with Dependency Injection
- **Error Handling**: FS0988 warnings as fatal errors
```

## Step 5: Create autonomous-objectives.md

### Autonomous Agent Goals

```markdown
# TARS Autonomous Objectives

## Primary Objectives
1. **Self-Improvement**: Continuous enhancement of reasoning capabilities
2. **Performance Optimization**: Achieve 184M+ searches/second target
3. **Code Quality**: Maintain zero-tolerance for simulations/placeholders
4. **Autonomous Development**: Generate real, functional implementations

## Success Metrics
- **Performance**: Measurable speed improvements with CUDA acceleration
- **Quality**: All tests pass with 80%+ coverage
- **Functionality**: Concrete proof of working implementations
- **Autonomy**: Successful self-directed improvements

## Autonomous Workflows
- **Planning**: Spec-driven development with Agent OS integration
- **Execution**: Real implementations following TARS standards
- **Validation**: Comprehensive testing and performance verification
- **Iteration**: Continuous improvement based on results
```

## Step 6: Create tars-roadmap.md

### TARS Enhancement Roadmap

```markdown
# TARS Enhancement Roadmap

## Phase 1: Agent OS Integration
**Goal:** Integrate Agent OS workflows with TARS autonomous capabilities
**Success Criteria:** TARS agents can follow structured Agent OS workflows

### Features
- [ ] Agent OS installation and configuration `S`
- [ ] TARS-specific standards integration `M`
- [ ] Autonomous workflow templates `M`
- [ ] Testing and validation framework `L`

## Phase 2: Enhanced Autonomous Planning
**Goal:** Improve TARS autonomous planning with Agent OS methodology
**Success Criteria:** Structured specs and task breakdown for autonomous improvements

### Features
- [ ] Spec-driven autonomous development `L`
- [ ] Task breakdown automation `M`
- [ ] Progress tracking integration `S`
- [ ] Performance metrics collection `M`

## Phase 3: Advanced Integration
**Goal:** Full integration of Agent OS with TARS metascript system
**Success Criteria:** Seamless workflow between Agent OS and TARS autonomous agents

### Features
- [ ] FLUX metascript Agent OS templates `L`
- [ ] Autonomous standards enforcement `M`
- [ ] Real-time workflow optimization `XL`
- [ ] Comprehensive testing suite `L`
```

## Effort Scale
- **XS**: 1 day
- **S**: 2-3 days  
- **M**: 1 week
- **L**: 2 weeks
- **XL**: 3+ weeks
