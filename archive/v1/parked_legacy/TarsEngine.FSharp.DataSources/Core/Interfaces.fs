namespace TarsEngine.FSharp.DataSources.Core

open System.Threading.Tasks

/// Interface for data source pattern detection
type IPatternDetector =
    abstract member DetectAsync: source: string -> Task<DetectionResult>
    abstract member GetSupportedPatterns: unit -> string list
    abstract member GetConfidenceThreshold: unit -> float

/// Interface for closure generation
type IClosureGenerator =
    abstract member GenerateAsync: parameters: ClosureParameters -> Task<GeneratedClosure>
    abstract member ValidateAsync: closure: GeneratedClosure -> Task<ValidationResult>
    abstract member CompileAsync: closure: GeneratedClosure -> Task<GeneratedClosure>

/// Interface for template management
type ITemplateEngine =
    abstract member LoadTemplate: templateName: string -> Task<string>
    abstract member FillTemplate: template: string * parameters: Map<string, obj> -> Task<string>
    abstract member ValidateTemplate: template: string -> Task<ValidationResult>
