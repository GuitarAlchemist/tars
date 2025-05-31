# TarsCli Restructuring Plan

Based on analysis of the current codebase, this document outlines a plan to restructure TarsCli into multiple feature-specific projects.

## Project Structure

### TarsCli.Core
- **Purpose**: Core CLI infrastructure, common utilities
- **Components**: Program.cs, CliSupport.cs, base command handling, common utilities
- **Dependencies**: System.CommandLine, NLog, Microsoft.Extensions libraries

### TarsCli.Testing
- **Purpose**: Test generation and execution
- **Components**: TestGeneratorCommand, TestingFrameworkCommand, TestRunnerService
- **Dependencies**: TarsCli.Core, MSTest.TestFramework

### TarsCli.SelfCoding
- **Purpose**: Self-improving code generation
- **Components**: SelfCodingCommand, AutoCodingCommand, SelfCodingWorkflow
- **Dependencies**: TarsCli.Core, TarsCli.Testing, TarsCli.CodeAnalysis

### TarsCli.CodeAnalysis
- **Purpose**: Code analysis and quality assessment
- **Components**: MockCodeComplexityCommand, CodeAnalysis/ services
- **Dependencies**: TarsCli.Core, Microsoft.CodeAnalysis.CSharp

### TarsCli.Docker
- **Purpose**: Docker container management and integration
- **Components**: DockerModelRunnerCommand, DockerAIAgentCommand
- **Dependencies**: TarsCli.Core

### TarsCli.DSL
- **Purpose**: Domain-specific language implementation
- **Components**: DslCommand, DslDebugCommand, MetascriptCommand
- **Dependencies**: TarsCli.Core, TarsEngine.DSL, FSharp libraries

### TarsCli.FSharp
- **Purpose**: F# language integration
- **Components**: FSharpAnalysisCommand, FSharpIntegrationService
- **Dependencies**: TarsCli.Core, TarsCli.CodeAnalysis, FSharp libraries

### TarsCli.Knowledge
- **Purpose**: Knowledge extraction and application
- **Components**: KnowledgeCommands, KnowledgeServices
- **Dependencies**: TarsCli.Core, TarsCli.CodeAnalysis

### TarsCli.Intelligence
- **Purpose**: Intelligence measurement and visualization
- **Components**: IntelligenceCommands, intelligence-related services
- **Dependencies**: TarsCli.Core, TarsEngine.IntelligenceProgression

### TarsCli.Agents
- **Purpose**: Agent-based operations
- **Components**: Services/Agents/ directory, MultiAgentCollaborationService
- **Dependencies**: TarsCli.Core, TarsCli.CodeAnalysis

### TarsCli.Swarm
- **Purpose**: Swarm-based intelligence
- **Components**: SwarmCommands, SwarmServices
- **Dependencies**: TarsCli.Core, TarsCli.Agents, TarsCli.MCP

### TarsCli.MCP
- **Purpose**: Message Control Protocol implementation
- **Components**: McpController, McpServices
- **Dependencies**: TarsCli.Core

### TarsCli.Autonomous
- **Purpose**: Autonomous improvement workflow
- **Components**: AutonomousCommand, RetroactionService
- **Dependencies**: TarsCli.Core, TarsCli.CodeAnalysis, TarsCli.Testing

### TarsCli.LLM
- **Purpose**: LLM integration (Claude, etc.)
- **Components**: ClaudeCommand, ModelProviderFactory
- **Dependencies**: TarsCli.Core, TarsCli.MCP

### TarsCli.SequentialThinking
- **Purpose**: Progressive thinking processes
- **Components**: SequentialThinkingCommand, CognitiveLoopService
- **Dependencies**: TarsCli.Core, TarsCli.Intelligence

### TarsCli.Demo
- **Purpose**: Demonstration workflows
- **Components**: DemoCommand, DemoService
- **Dependencies**: Multiple feature-specific projects

## Implementation Strategy

1. Create new project structure in the solution
2. Extract shared interfaces and models to core project
3. Move feature-specific code to respective projects
4. Update references and namespaces
5. Implement project dependencies
6. Update build scripts and CI/CD pipelines
7. Test extensively at each step

This modular approach will improve maintainability, enable more focused testing, and allow for better separation of concerns.