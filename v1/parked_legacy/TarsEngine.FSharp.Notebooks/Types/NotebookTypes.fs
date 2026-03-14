namespace TarsEngine.FSharp.Notebooks.Types

open System
open System.Collections.Generic

/// <summary>
/// Core types for Jupyter notebook representation and manipulation
/// </summary>

/// Notebook cell types following nbformat specification
type NotebookCell = 
    | CodeCell of CodeCellData
    | MarkdownCell of MarkdownCellData
    | RawCell of RawCellData

/// Code cell with execution capabilities
and CodeCellData = {
    Source: string list
    Language: string
    Outputs: NotebookOutput list
    ExecutionCount: int option
    Metadata: Map<string, obj>
}

/// Markdown cell for documentation
and MarkdownCellData = {
    Source: string list
    Metadata: Map<string, obj>
}

/// Raw cell for unprocessed content
and RawCellData = {
    Source: string list
    Format: string option
    Metadata: Map<string, obj>
}

/// Notebook output types
and NotebookOutput = 
    | DisplayData of DisplayDataOutput
    | ExecuteResult of ExecuteResultOutput
    | StreamOutput of StreamOutputData
    | ErrorOutput of ErrorOutputData

/// Display data output (images, HTML, etc.)
and DisplayDataOutput = {
    Data: Map<string, obj>
    Metadata: Map<string, obj>
}

/// Execution result with data
and ExecuteResultOutput = {
    Data: Map<string, obj>
    ExecutionCount: int
    Metadata: Map<string, obj>
}

/// Stream output (stdout, stderr)
and StreamOutputData = {
    Name: string // "stdout" or "stderr"
    Text: string list
}

/// Error output with traceback
and ErrorOutputData = {
    Name: string
    Value: string
    Traceback: string list
}

/// Complete Jupyter notebook structure
type JupyterNotebook = {
    Metadata: NotebookMetadata
    Cells: NotebookCell list
    NbFormat: int
    NbFormatMinor: int
}

/// Notebook metadata
and NotebookMetadata = {
    KernelSpec: KernelSpecification option
    LanguageInfo: LanguageInformation option
    Title: string option
    Authors: string list
    Created: DateTime option
    Modified: DateTime option
    Custom: Map<string, obj>
}

/// Kernel specification
and KernelSpecification = {
    Name: string
    DisplayName: string
    Language: string
    InterruptMode: string option
    Env: Map<string, string>
    Argv: string list
}

/// Language information
and LanguageInformation = {
    Name: string
    Version: string
    MimeType: string option
    FileExtension: string option
    PygmentsLexer: string option
    CodeMirrorMode: string option
}

/// Supported kernel types
type SupportedKernel = 
    | Python of PythonKernelConfig
    | FSharp of FSharpKernelConfig  
    | CSharp of CSharpKernelConfig
    | JavaScript of JavaScriptKernelConfig
    | SQL of SQLKernelConfig
    | R of RKernelConfig
    | Custom of CustomKernelConfig

/// Python kernel configuration
and PythonKernelConfig = {
    Version: string
    Packages: string list
    VirtualEnv: string option
}

/// F# kernel configuration
and FSharpKernelConfig = {
    DotNetVersion: string
    Packages: string list
    References: string list
}

/// C# kernel configuration
and CSharpKernelConfig = {
    DotNetVersion: string
    Packages: string list
    References: string list
}

/// JavaScript kernel configuration
and JavaScriptKernelConfig = {
    NodeVersion: string
    Packages: string list
}

/// SQL kernel configuration
and SQLKernelConfig = {
    ConnectionString: string
    DatabaseType: string
    Schema: string option
}

/// R kernel configuration
and RKernelConfig = {
    Version: string
    Packages: string list
    CranMirror: string option
}

/// Custom kernel configuration
and CustomKernelConfig = {
    Name: string
    Command: string
    Args: string list
    Environment: Map<string, string>
}

/// Notebook generation strategy
type NotebookGenerationStrategy = 
    | ExploratoryDataAnalysis
    | MachineLearningPipeline
    | ResearchNotebook
    | TutorialNotebook
    | DocumentationNotebook
    | BusinessReport
    | AcademicPaper

/// Notebook template configuration
type NotebookTemplate = {
    Id: string
    Name: string
    Description: string
    Strategy: NotebookGenerationStrategy
    DefaultKernel: SupportedKernel
    CellTemplates: CellTemplate list
    Variables: Map<string, obj>
    Metadata: Map<string, obj>
}

/// Cell template for generation
and CellTemplate = {
    Type: string // "code", "markdown", "raw"
    Content: string
    Variables: string list
    Order: int
    Conditional: string option
}

/// Notebook quality metrics
type NotebookQualityMetrics = {
    CodeQuality: float
    DocumentationQuality: float
    Reproducibility: float
    Educational: float
    Completeness: float
    Popularity: float
    Recency: float
}

/// Notebook search result
type NotebookSearchResult = {
    Id: string
    Title: string
    Description: string
    Author: string
    Source: string
    Url: string
    Quality: NotebookQualityMetrics
    Tags: string list
    Language: string
    LastModified: DateTime
}

/// Notebook discovery source
type NotebookSource = 
    | GitHub of GitHubConfig
    | Kaggle of KaggleConfig
    | GoogleColab of ColabConfig
    | NBViewer of NBViewerConfig
    | ArXiv of ArXivConfig
    | PapersWithCode of PapersWithCodeConfig
    | Custom of CustomSourceConfig

/// GitHub source configuration
and GitHubConfig = {
    ApiToken: string option
    SearchQuery: string
    MaxResults: int
    IncludePrivate: bool
}

/// Kaggle source configuration
and KaggleConfig = {
    ApiKey: string
    Username: string
    Categories: string list
    MinScore: float option
}

/// Google Colab configuration
and ColabConfig = {
    SearchTerms: string list
    PublicOnly: bool
}

/// NBViewer configuration
and NBViewerConfig = {
    BaseUrl: string
    SearchEndpoint: string
}

/// ArXiv configuration
and ArXivConfig = {
    Categories: string list
    MaxResults: int
    DateRange: (DateTime * DateTime) option
}

/// Papers with Code configuration
and PapersWithCodeConfig = {
    Areas: string list
    MinCitations: int option
}

/// Custom source configuration
and CustomSourceConfig = {
    Name: string
    BaseUrl: string
    SearchEndpoint: string
    Headers: Map<string, string>
    QueryParams: Map<string, string>
}

/// Validation result
type ValidationResult = {
    IsValid: bool
    Errors: ValidationError list
    Warnings: ValidationWarning list
}

/// Validation error
and ValidationError = {
    Code: string
    Message: string
    Location: string option
    Severity: ErrorSeverity
}

/// Validation warning
and ValidationWarning = {
    Code: string
    Message: string
    Location: string option
    Suggestion: string option
}

/// Error severity levels
and ErrorSeverity = 
    | Critical
    | High
    | Medium
    | Low
    | Info

/// Notebook execution result
type NotebookExecutionResult = {
    Notebook: JupyterNotebook
    Success: bool
    ExecutionTime: TimeSpan
    Errors: string list
    Warnings: string list
    OutputSummary: string
}

/// University collaboration types
type ResearchProject = {
    Id: string
    Title: string
    Description: string
    PrincipalInvestigator: Researcher
    TeamMembers: Researcher list
    Notebooks: string list
    Datasets: string list
    Publications: string list
    Status: ProjectStatus
    Timeline: ProjectTimeline
}

/// Researcher information
and Researcher = {
    Id: string
    Name: string
    Affiliation: string
    ORCID: string option
    Expertise: string list
    Role: ProjectRole
}

/// Project status
and ProjectStatus = 
    | Planning
    | Active
    | OnHold
    | Completed
    | Cancelled

/// Project role
and ProjectRole = 
    | PrincipalInvestigator
    | CoInvestigator
    | PostDoc
    | GraduateStudent
    | UndergraduateStudent
    | Collaborator

/// Project timeline
and ProjectTimeline = {
    StartDate: DateTime
    EndDate: DateTime option
    Milestones: Milestone list
}

/// Project milestone
and Milestone = {
    Id: string
    Title: string
    Description: string
    DueDate: DateTime
    Status: MilestoneStatus
    Deliverables: string list
}

/// Milestone status
and MilestoneStatus = 
    | NotStarted
    | InProgress
    | Completed
    | Overdue
