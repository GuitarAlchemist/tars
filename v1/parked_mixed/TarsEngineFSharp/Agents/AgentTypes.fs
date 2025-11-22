namespace TarsEngineFSharp

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.CodeAnalysis
open Microsoft.CodeAnalysis.CSharp
open Microsoft.CodeAnalysis.CSharp.Syntax

/// <summary>
/// Represents the severity of a validation issue
/// </summary>
type ValidationSeverity =
    | Info = 0
    | Warning = 1
    | Error = 2

/// <summary>
/// Represents a validation issue found in code
/// </summary>
type ValidationIssue(message: string, severity: ValidationSeverity, location: string) =
    member val Message = message with get, set
    member val Severity = severity with get, set
    member val Location = location with get, set

/// <summary>
/// Represents the result of a validation operation
/// </summary>
type ValidationResult(isValid: bool, issues: IEnumerable<ValidationIssue>) =
    member val IsValid = isValid with get, set
    member val Issues = issues with get, set

/// <summary>
/// Represents a code issue found during analysis
/// </summary>
type CodeIssue(message: string, description: string, location: string) =
    member val Message = message with get, set
    member val Description = description with get, set
    member val Location = location with get, set

/// <summary>
/// Represents a code fix that can be applied to address an issue
/// </summary>
type CodeFix(original: string, replacement: string, description: string) =
    member val Original = original with get, set
    member val Replacement = replacement with get, set
    member val Description = description with get, set

/// <summary>
/// Represents the result of a code analysis operation
/// </summary>
type AnalysisResult(issues: IEnumerable<CodeIssue>, fixes: IEnumerable<CodeFix>) =
    member val Issues = issues with get, set
    member val Fixes = fixes with get, set

/// <summary>
/// Interface for an agent that analyzes code
/// </summary>
type IAnalysisAgent =
    abstract member AnalyzeAsync : string -> Task<AnalysisResult>

/// <summary>
/// Interface for an agent that transforms code
/// </summary>
type ITransformationAgent =
    abstract member TransformAsync : string -> Task<string>

/// <summary>
/// Interface for an agent that validates code
/// </summary>
type IValidationAgent =
    abstract member ValidateAsync : string * string -> Task<ValidationResult>

/// <summary>
/// Interface for an agent that learns from code transformations
/// </summary>
type ILearningAgent =
    abstract member LearnAsync : string * string * bool -> Task<unit>
