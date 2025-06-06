namespace TarsEngine.FSharp.Core.Projects.Types

open System
open System.Collections.Generic
open System.Threading.Tasks

/// Project structure definition
type ProjectStructure = {
    ProjectName: string
    ProjectType: string
    Description: string
    Technologies: string list
    Dependencies: string list
    FileStructure: Map<string, string>
    EntryPoint: string option
    BuildCommands: string list
    TestCommands: string list
    CreatedAt: DateTime
}

/// Generated file information
type GeneratedFile = {
    Path: string
    Content: string
    FileType: string
    Language: string option
    Purpose: string
    Dependencies: string list
    GeneratedAt: DateTime
}

/// Project creation result
type ProjectCreationResult = {
    ProjectStructure: ProjectStructure
    GeneratedFiles: int
    TestsGenerated: int
    OutputPath: string
    ValidationResults: bool
    ExecutionTime: TimeSpan
    Errors: string list
    Warnings: string list
    CreatedAt: DateTime
}

/// Project template definition
type ProjectTemplate = {
    Name: string
    Description: string
    Category: string
    Technologies: string list
    FileTemplates: Map<string, string>
    Dependencies: string list
    BuildConfiguration: string
    TestConfiguration: string option
}

/// Autonomous project service interface
type IAutonomousProjectService =
    abstract member CreateProjectFromPromptAsync: prompt: string -> Task<ProjectCreationResult>
    abstract member DemoProjectCreationAsync: unit -> Task<ProjectCreationResult>
    abstract member GenerateProjectStructureAsync: requirements: string -> Task<ProjectStructure>
    abstract member ValidateProjectAsync: projectPath: string -> Task<bool>
    abstract member GetAvailableTemplatesAsync: unit -> Task<ProjectTemplate list>
