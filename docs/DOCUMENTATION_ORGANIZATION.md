# TARS Documentation Organization

This document outlines the organization of the TARS documentation to ensure consistency and ease of navigation.

## Directory Structure

The TARS documentation is organized into the following directories:

```
docs/
├── README.md                     # Main documentation index
├── architecture.md               # System architecture overview
├── capabilities.md               # Core capabilities
├── contributing.md               # Contributing guidelines
├── evolution.md                  # Self-evolution milestones
├── faq.md                        # Frequently asked questions
├── getting-started.md            # Getting started guide
├── IMAGES.md                     # Information about images
├── index.md                      # Documentation home page
├── roadmap.md                    # Development roadmap
├── STATUS.md                     # Current status
├── TODOs.md                      # Tasks to be completed
├── vision.md                     # Vision and goals
├── ai-concepts/                  # AI concepts guide
│   ├── index.md                  # AI concepts index
│   ├── large-language-models.md  # LLM guide
│   └── ...
├── api/                          # API reference
│   └── index.md                  # API index
├── AutoCoding/                   # Auto-coding documentation
│   ├── README.md                 # Auto-coding overview
│   ├── TARS-Auto-Coding-Status.md # Auto-coding status
│   ├── Docker/                   # Docker auto-coding
│   │   ├── README.md             # Docker auto-coding overview
│   │   └── ...
│   └── Swarm/                    # Swarm auto-coding
│       ├── README.md             # Swarm auto-coding overview
│       └── ...
├── AutonomousExecution/          # Autonomous execution
│   ├── README.md                 # Autonomous execution overview
│   └── ...
├── demos/                        # Demo documentation
│   ├── A2A-Demo.md               # A2A protocol demo
│   └── ...
├── development/                  # Development guides
│   ├── functional-programming-guide.md # F# guide
│   └── ...
├── DSL/                          # DSL documentation
│   ├── index.md                  # DSL index
│   └── ...
├── Explorations/                 # Explorations
│   ├── Reflections/              # Reflections on explorations
│   └── v1/                       # Version 1 explorations
├── features/                     # Feature documentation
│   ├── auto-improvement.md       # Auto-improvement
│   ├── chat.md                   # Chat bot
│   ├── deep-thinking.md          # Deep thinking
│   ├── Metascripts.md            # Metascripts
│   ├── model-context-protocol.md # MCP
│   ├── self-improvement.md       # Self-improvement
│   ├── speech.md                 # Speech system
│   └── ...
├── FIXES/                        # Fix documentation
│   └── ...
├── history/                      # Historical documentation
│   └── ...
└── post-mortem/                  # Post-mortem analyses
    ├── index.md                  # Post-mortem index
    └── ...
```

## Naming Conventions

To ensure consistency and avoid broken links, the following naming conventions should be followed:

1. **File Names**: Use lowercase with hyphens for spaces (kebab-case)
   - Example: `getting-started.md`, `model-context-protocol.md`

2. **Directory Names**: Use PascalCase for directories
   - Example: `AutoCoding`, `AutonomousExecution`

3. **Special Files**: Use uppercase with underscores for special files
   - Example: `README.md`, `CONTRIBUTING.md`, `TODOs.md`

## Link References

When linking to other documentation files, use the following guidelines:

1. **Relative Paths**: Use relative paths for links within the documentation
   - Example: `[Metascripts](features/Metascripts.md)`

2. **Root-Relative Paths**: For links from the root of the documentation, use paths relative to the docs directory
   - Example: `[Auto-Coding](AutoCoding/README.md)`

3. **External Links**: Use absolute URLs for external links
   - Example: `[Docker Documentation](https://docs.docker.com/)`

4. **GitHub Links**: For links to GitHub files, use the full GitHub URL
   - Example: `[A2A Protocol](https://github.com/GuitarAlchemist/tars/blob/main/docs/A2A-Protocol.md)`

## Documentation Standards

1. **Headers**: Use ATX-style headers (with `#` symbols)
   - Main title: `# Title`
   - Section: `## Section`
   - Subsection: `### Subsection`

2. **Lists**: Use `-` for unordered lists and `1.` for ordered lists

3. **Code Blocks**: Use triple backticks with language identifier
   ```csharp
   public class Example {}
   ```

4. **Images**: Include alt text and use relative paths
   `![Alt Text](../images/example.png)`

5. **Tables**: Use standard Markdown tables
   ```
   | Header 1 | Header 2 |
   |----------|----------|
   | Cell 1   | Cell 2   |
   ```

## Documentation Maintenance

To maintain the documentation:

1. **Regular Reviews**: Review documentation for accuracy and broken links
2. **Update with Features**: Update documentation when new features are added
3. **Fix Broken Links**: Fix broken links as soon as they are discovered
4. **Standardize Format**: Ensure all documentation follows the standards

## Tools

The following tools can help maintain documentation quality:

1. **Markdown Linters**: Use markdown linters to ensure consistent formatting
2. **Link Checkers**: Use link checkers to find broken links
3. **Spell Checkers**: Use spell checkers to find spelling errors
4. **Documentation Generators**: Use documentation generators for API documentation

## Conclusion

Following these guidelines will help maintain a consistent and navigable documentation structure for the TARS project. As the project evolves, the documentation organization may need to be updated to accommodate new features and requirements.
