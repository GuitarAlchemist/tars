# TARS TODOs

This directory contains task lists for various aspects of the TARS project. Each file focuses on a specific area or feature.

## Available TODOs Files

- **TODOs-General.md**: General tasks for the TARS project
- **TODOs-MCP-Swarm.md**: Tasks for implementing and enhancing the MCP Swarm features
- **TODOs-Refactoring.md**: Tasks for refactoring the TarsEngine codebase
- **TODOs-Self-Coding.md**: Tasks for implementing TARS self-coding capabilities
- **TODOs-TARS-Swarm.md**: Tasks for implementing the TARS Swarm features
- **TODOs-TestGenerator-SequentialThinking.md**: Tasks for integrating the Test Generator with Sequential Thinking
- **TODOs-Infrastructure.md**: Shared infrastructure tasks across multiple components
- **TODOs-Automated-Coding-Loop.md**: Tasks for implementing an automated coding loop for TARS self-improvement

## Task Format

Each task should follow this format:

```markdown
- [ ] (P0) [Est: 2d] [Owner: SP] Implement feature X @depends-on:#123 #456
  - Description or additional details
  - Acceptance criteria
  - Links to relevant documentation or code
```

Where:
- `[ ]` or `[x]` indicates completion status
- `(P0)`, `(P1)`, or `(P2)` indicates priority (P0 = highest)
- `[Est: Xd]` provides a time estimate in days
- `[Owner: XYZ]` shows who is responsible (using initials)
- `@depends-on:#123 #456` lists dependencies (GitHub issues, other tasks)

## How to Use

Each TODOs file follows a similar structure:
- Tasks are organized into logical sections
- Tasks include priority, estimates, and ownership
- Dependencies between tasks are explicitly noted
- Implementation phases are defined for larger features
- Documentation tasks are paired with implementation tasks

When working on a task:
1. Mark it as in progress by adding your initials in the Owner field
2. Update the status in weekly team meetings
3. Mark it as complete by checking the box when done
4. Add any relevant notes or links to implementations
5. Ensure documentation is updated alongside code changes

## Sprint Planning

Tasks are organized into quarterly milestones:
- **Q2 2025**: Knowledge Extraction MVP, Test Generator Integration
- **Q3 2025**: Intelligence Measurement MVP, Sequential Thinking Integration
- **Q4 2025**: Full Autonomous Self-Improvement

## Adding New TODOs

When adding new TODOs:
1. Consider whether they fit into an existing file or warrant a new file
2. Follow the established format for consistency, including priority and estimates
3. Organize tasks into logical sections
4. Define implementation phases for larger features
5. Include relevant resources and references
6. Add testing and documentation tasks alongside implementation tasks
7. Identify dependencies between tasks

## Regular Reviews

The TODOs files are reviewed bi-weekly to:
- Prune stale tasks
- Re-estimate pending items
- Celebrate completed milestones
- Adjust priorities based on project needs

## TODOs File Naming Convention

Use the format `TODOs-{Feature}.md` for new TODOs files, where `{Feature}` is a brief description of the feature or area the tasks relate to.
