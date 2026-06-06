# Best Practices

## Context

TARS-specific development best practices for Agent OS integration.

## Development Philosophy

### Quality Standards
- **Zero tolerance for simulations/placeholders** - All implementations must be real and functional
- **Quality over quantity** - Polish existing features to production quality rather than rapidly adding half-working capabilities
- **Concrete proof required** - Validate that systems are fully functional before accepting completion claims
- **Real functionality over fake demos** - Demonstrate useful, practical use cases rather than just technical capabilities

### Testing Approach
- **Test-Driven Development (TDD)** preferred
- **80% test coverage minimum** with unit and integration tests
- **Real CUDA testing** - Demonstrate actual GPU execution, not CPU simulation
- **Evidence-based validation** - Provide concrete proof that systems work as claimed

### Architecture Principles
- **Clean Architecture** with Separation of Concerns (SoC)
- **Dependency Inversion** - Use interfaces, inject dependencies
- **Modular design** - Avoid tight coupling
- **Elmish/MVU architecture** for UI components - truly interactive and dynamic functionality

### Code Organization
- **DRY (Don't Repeat Yourself)** - Refactor common patterns
- **KISS (Keep It Simple, Stupid)** - Avoid overengineering
- **Proper indentation** and breaking down large .fs files into smaller, manageable pieces
- **Conventional commits** with clear messages (feat:, refactor:, fix:, etc.)

## TARS-Specific Practices

### Autonomous Development
- **Structured workflows** over ad-hoc autonomous improvements
- **Spec-driven development** - Create detailed specs before autonomous code generation
- **Standards-driven autonomous development** - Ensure autonomous agents follow defined coding standards
- **Multi-agent coordination** with clear hierarchical command structure

### Performance and Optimization
- **Measure impact** (execution time, memory usage) before optimizing
- **CUDA acceleration** must show real GPU performance gains
- **WSL for CUDA compilation** - Never compile CUDA on Windows directly
- **184M+ searches/second target** for vector operations
- **Async/parallel processing** when appropriate

### Metascript Development
- **FLUX metascript language** exclusively - no .fsx F# script files
- **Real F# project files** and FLUX metascript language only
- **Dynamic, configurable solutions** over static, hard-coded implementations
- **Metascript-first approach** with TARS engine injected as API

### Memory and Knowledge Management
- **Local-first approach** - Prefer local resources (Ollama, Docker) before cloud services
- **ChromaDB or custom CUDA vector stores** for knowledge storage
- **BSP partitioning with sedenions** for vector store knowledge clustering
- **Persistent learning capabilities** using supported databases

### Error Handling and Diagnostics
- **FS0988 warnings as fatal errors** for quality assurance
- **Result/AsyncResult monads** over exceptions in F#
- **Graceful degradation** and proper logging
- **Comprehensive diagnostics** with cryptographic certification

### UI and User Experience
- **Elmish over static HTML** for UI implementations
- **Spectre Console** for CLI mini UIs and progress bars
- **Interactive 3D UI** with Three.js for multi-agent visualizations
- **Mouse 3D navigation controls** for interactive camera movement

### Documentation and Communication
- **Document WHY, not just WHAT** in code comments
- **Auto-generate summaries** with meaningful context
- **YAML format over JSON** for structured execution traces
- **Detailed explanations** of what's happening during demonstration processes

## Workflow Practices

### Planning and Execution
- **Iterative improvement** with incremental changes
- **Clear commit messages** following conventional commit format
- **Blue-green deployment strategy** - test changes on Docker blue replica first
- **Feedback loops** - Better feedback loops, improved diagnostics, automation of repetitive tasks

### Team Collaboration
- **Self-improvement first** - Prioritize better feedback loops and diagnostics
- **Agentic patterns** - Apply objective decomposition, chain tasks logically
- **Sub-agent spawning** when needed for complex problem decomposition
- **Game theory-based inter-agent communication**

### Continuous Improvement
- **Meta-scripting awareness** - Recognize .tars metascripts and maintain structure
- **Reasoning blocks updates** where relevant
- **Pattern recognition improvement** and metascript evolution
- **Live infrastructure integration** for real-world testing

## Anti-Patterns to Avoid

### Code Quality Issues
- **Hardcoding values** - Use configs or metascripts instead
- **External dependencies without justification** - Prefer local-first solutions
- **Bypassing feedback or validation systems**
- **Mixing refactor and feature changes** in single commits

### Development Practices
- **Simulated or fake implementations** - Zero tolerance policy
- **Static if/then branching patterns** - Prefer adaptive/dynamic solutions
- **CPU simulation of CUDA operations** - Must use real GPU acceleration
- **Overly granular tasks** - Tasks should represent meaningful units of work (~20 minutes)

### Architecture Violations
- **Tight coupling** between components
- **Violation of separation of concerns**
- **Direct editing of package files** - Use package managers instead
- **Static HTML generation** - Use Elmish architecture instead

## Success Metrics

### Quality Indicators
- **All tests pass** with 80%+ coverage
- **Real GPU acceleration** demonstrated with performance metrics
- **Zero FS0988 warnings** in production code
- **Functional demonstrations** of all claimed capabilities

### Performance Targets
- **184M+ searches/second** for vector operations
- **Sub-second response times** for autonomous reasoning
- **Efficient memory usage** with monitoring and optimization
- **Real-time 3D rendering** for multi-agent visualizations

### User Experience Goals
- **Intuitive CLI interfaces** with progress indicators
- **Clear, detailed explanations** during demonstrations
- **Interactive and dynamic functionality** in all UI components
- **Practical, useful applications** of advanced mathematical concepts
