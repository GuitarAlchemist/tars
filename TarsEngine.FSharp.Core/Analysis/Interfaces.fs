namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Interface for language-specific code analyzers.
/// </summary>
type ILanguageAnalyzer =
    /// <summary>
    /// Gets the language supported by this analyzer.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Analyzes code content.
    /// </summary>
    /// <param name="content">The code content to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    abstract member AnalyzeAsync : content:string * ?options:Map<string, string> -> Task<CodeAnalysisResult>
    
    /// <summary>
    /// Analyzes a file.
    /// </summary>
    /// <param name="filePath">The path to the file to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    abstract member AnalyzeFileAsync : filePath:string * ?options:Map<string, string> -> Task<CodeAnalysisResult>
    
    /// <summary>
    /// Gets the supported analysis options.
    /// </summary>
    /// <returns>A dictionary of option names to descriptions.</returns>
    abstract member GetSupportedOptions : unit -> IDictionary<string, string>

/// <summary>
/// Interface for extracting code structures from source code.
/// </summary>
type ICodeStructureExtractor =
    /// <summary>
    /// Gets the language supported by this extractor.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Extracts code structures from the provided content.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of extracted code structures.</returns>
    abstract member ExtractStructures : content:string -> CodeStructure list
    
    /// <summary>
    /// Extracts code structures from a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of extracted code structures.</returns>
    abstract member ExtractStructuresFromFile : filePath:string -> CodeStructure list
    
    /// <summary>
    /// Gets a structure by name.
    /// </summary>
    /// <param name="structures">The list of structures to search.</param>
    /// <param name="name">The name of the structure to find.</param>
    /// <returns>The found structure, if any.</returns>
    abstract member GetStructureByName : structures:CodeStructure list * name:string -> CodeStructure option
    
    /// <summary>
    /// Gets structures by type.
    /// </summary>
    /// <param name="structures">The list of structures to search.</param>
    /// <param name="structureType">The type of structures to find.</param>
    /// <returns>The list of found structures.</returns>
    abstract member GetStructuresByType : structures:CodeStructure list * structureType:string -> CodeStructure list

/// <summary>
/// Interface for detecting code issues.
/// </summary>
type IIssueDetector =
    /// <summary>
    /// Gets the language supported by this detector.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Detects issues in the provided content.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectIssues : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects issues in a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectIssuesInFile : filePath:string -> CodeIssue list
    
    /// <summary>
    /// Gets the supported issue types.
    /// </summary>
    /// <returns>A list of supported issue types.</returns>
    abstract member GetSupportedIssueTypes : unit -> CodeIssueType list

/// <summary>
/// Interface for detecting security issues in code.
/// </summary>
type ISecurityIssueDetector =
    inherit IIssueDetector
    
    /// <summary>
    /// Detects SQL injection vulnerabilities.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectSqlInjection : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects cross-site scripting vulnerabilities.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectXss : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects insecure cryptography.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectInsecureCryptography : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects hardcoded credentials.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectHardcodedCredentials : content:string -> CodeIssue list

/// <summary>
/// Interface for detecting performance issues in code.
/// </summary>
type IPerformanceIssueDetector =
    inherit IIssueDetector
    
    /// <summary>
    /// Detects inefficient loops.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectInefficientLoops : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects large object creation.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectLargeObjectCreation : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects excessive memory usage.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectExcessiveMemoryUsage : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects inefficient string operations.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectInefficientStringOperations : content:string -> CodeIssue list

/// <summary>
/// Interface for detecting complexity issues in code.
/// </summary>
type IComplexityIssueDetector =
    inherit IIssueDetector
    
    /// <summary>
    /// Detects methods with high cyclomatic complexity.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <param name="structures">The extracted code structures.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectHighCyclomaticComplexity : content:string * structures:CodeStructure list -> CodeIssue list
    
    /// <summary>
    /// Detects methods with too many parameters.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectTooManyParameters : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects deeply nested code.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectDeepNesting : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects long methods.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <param name="structures">The extracted code structures.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectLongMethods : content:string * structures:CodeStructure list -> CodeIssue list

/// <summary>
/// Interface for detecting style issues in code.
/// </summary>
type IStyleIssueDetector =
    inherit IIssueDetector
    
    /// <summary>
    /// Detects naming convention violations.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectNamingConventionViolations : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects formatting issues.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectFormattingIssues : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects code style inconsistencies.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectCodeStyleInconsistencies : content:string -> CodeIssue list
    
    /// <summary>
    /// Detects comment style issues.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    abstract member DetectCommentStyleIssues : content:string -> CodeIssue list

/// <summary>
/// Interface for the code analyzer service.
/// </summary>
type ICodeAnalyzerService =
    /// <summary>
    /// Analyzes a file.
    /// </summary>
    /// <param name="filePath">The path to the file to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    abstract member AnalyzeFileAsync : filePath:string * ?options:Map<string, string> -> Task<CodeAnalysisResult>
    
    /// <summary>
    /// Analyzes a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory to analyze.</param>
    /// <param name="recursive">Whether to analyze subdirectories.</param>
    /// <param name="filePattern">The pattern to match files to analyze.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis results.</returns>
    abstract member AnalyzeDirectoryAsync : directoryPath:string * ?recursive:bool * ?filePattern:string * ?options:Map<string, string> -> Task<CodeAnalysisResult list>
    
    /// <summary>
    /// Analyzes code content.
    /// </summary>
    /// <param name="content">The code content to analyze.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="options">Optional analysis options.</param>
    /// <returns>The analysis result.</returns>
    abstract member AnalyzeContentAsync : content:string * language:string * ?options:Map<string, string> -> Task<CodeAnalysisResult>
    
    /// <summary>
    /// Gets the supported languages for analysis.
    /// </summary>
    /// <returns>The list of supported languages.</returns>
    abstract member GetSupportedLanguagesAsync : unit -> Task<string list>
    
    /// <summary>
    /// Gets the issues for a specific file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <param name="issueTypes">The types of issues to get.</param>
    /// <param name="minSeverity">The minimum severity of issues to get.</param>
    /// <param name="options">Optional filtering options.</param>
    /// <returns>The list of issues.</returns>
    abstract member GetIssuesForFileAsync : filePath:string * ?issueTypes:CodeIssueType list * ?minSeverity:IssueSeverity * ?options:Map<string, string> -> Task<CodeIssue list>

/// <summary>
/// Interface for the pattern matcher service.
/// </summary>
type IPatternMatcherService =
    /// <summary>
    /// Finds patterns in the provided content.
    /// </summary>
    /// <param name="content">The code content to analyze.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="options">Optional matching options.</param>
    /// <returns>The list of pattern matches.</returns>
    abstract member FindPatternsAsync : content:string * language:string * ?options:Map<string, string> -> Task<PatternMatch list>
    
    /// <summary>
    /// Finds patterns in a file.
    /// </summary>
    /// <param name="filePath">The path to the file to analyze.</param>
    /// <param name="options">Optional matching options.</param>
    /// <returns>The list of pattern matches.</returns>
    abstract member FindPatternsInFileAsync : filePath:string * ?options:Map<string, string> -> Task<PatternMatch list>
    
    /// <summary>
    /// Finds patterns in a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory to analyze.</param>
    /// <param name="recursive">Whether to analyze subdirectories.</param>
    /// <param name="filePattern">The pattern to match files to analyze.</param>
    /// <param name="options">Optional matching options.</param>
    /// <returns>The list of pattern matches grouped by file.</returns>
    abstract member FindPatternsInDirectoryAsync : directoryPath:string * ?recursive:bool * ?filePattern:string * ?options:Map<string, string> -> Task<Map<string, PatternMatch list>>
    
    /// <summary>
    /// Calculates the similarity between two code snippets.
    /// </summary>
    /// <param name="source">The source code snippet.</param>
    /// <param name="target">The target code snippet.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <returns>The similarity score (0.0 to 1.0).</returns>
    abstract member CalculateSimilarityAsync : source:string * target:string * language:string -> Task<float>
    
    /// <summary>
    /// Finds similar patterns to the provided code.
    /// </summary>
    /// <param name="content">The code content to find similar patterns for.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="minSimilarity">The minimum similarity score (0.0 to 1.0).</param>
    /// <param name="maxResults">The maximum number of results to return.</param>
    /// <returns>The list of similar patterns with their similarity scores.</returns>
    abstract member FindSimilarPatternsAsync : content:string * language:string * ?minSimilarity:float * ?maxResults:int -> Task<(CodePattern * float) list>

/// <summary>
/// Interface for code complexity analyzer.
/// </summary>
type ICodeComplexityAnalyzer =
    /// <summary>
    /// Analyzes cyclomatic complexity of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Complexity metrics for the file.</returns>
    abstract member AnalyzeCyclomaticComplexityAsync : filePath:string * language:string -> Task<ComplexityMetric list>
    
    /// <summary>
    /// Analyzes cognitive complexity of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Complexity metrics for the file.</returns>
    abstract member AnalyzeCognitiveComplexityAsync : filePath:string * language:string -> Task<ComplexityMetric list>
    
    /// <summary>
    /// Analyzes maintainability index of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Maintainability metrics for the file.</returns>
    abstract member AnalyzeMaintainabilityIndexAsync : filePath:string * language:string -> Task<MaintainabilityMetric list>
    
    /// <summary>
    /// Analyzes Halstead complexity of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Halstead complexity metrics for the file.</returns>
    abstract member AnalyzeHalsteadComplexityAsync : filePath:string * language:string -> Task<HalsteadMetric list>
    
    /// <summary>
    /// Analyzes readability of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <param name="readabilityType">Type of readability to analyze.</param>
    /// <returns>Readability metrics for the file.</returns>
    abstract member AnalyzeReadabilityAsync : filePath:string * language:string * readabilityType:ReadabilityType -> Task<ReadabilityMetric list>
    
    /// <summary>
    /// Analyzes all complexity metrics of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>All complexity metrics for the file.</returns>
    abstract member AnalyzeAllComplexityMetricsAsync : filePath:string * language:string -> Task<ComplexityMetric list * HalsteadMetric list * MaintainabilityMetric list * ReadabilityMetric list>
    
    /// <summary>
    /// Analyzes complexity metrics of a project.
    /// </summary>
    /// <param name="projectPath">Path to the project.</param>
    /// <returns>All complexity metrics for the project.</returns>
    abstract member AnalyzeProjectComplexityAsync : projectPath:string -> Task<ComplexityMetric list * HalsteadMetric list * MaintainabilityMetric list * ReadabilityMetric list>

/// <summary>
/// Interface for readability analyzer.
/// </summary>
type IReadabilityAnalyzer =
    /// <summary>
    /// Analyzes naming conventions of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Readability metrics for the file.</returns>
    abstract member AnalyzeNamingConventionsAsync : filePath:string * language:string -> Task<ReadabilityMetric list>
    
    /// <summary>
    /// Analyzes comment quality of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Readability metrics for the file.</returns>
    abstract member AnalyzeCommentQualityAsync : filePath:string * language:string -> Task<ReadabilityMetric list>
    
    /// <summary>
    /// Analyzes code structure of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Readability metrics for the file.</returns>
    abstract member AnalyzeCodeStructureAsync : filePath:string * language:string -> Task<ReadabilityMetric list>
    
    /// <summary>
    /// Analyzes overall readability of a file.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>Readability metrics for the file.</returns>
    abstract member AnalyzeOverallReadabilityAsync : filePath:string * language:string -> Task<ReadabilityMetric list>

/// <summary>
/// Interface for progress reporter.
/// </summary>
type IProgressReporter =
    /// <summary>
    /// Reports progress.
    /// </summary>
    /// <param name="message">The progress message.</param>
    /// <param name="percentComplete">The percentage complete (0-100).</param>
    abstract member ReportProgress : message:string * percentComplete:int -> unit
    
    /// <summary>
    /// Reports a warning.
    /// </summary>
    /// <param name="message">The warning message.</param>
    /// <param name="exception">The exception, if any.</param>
    abstract member ReportWarning : message:string * ?exception:Exception -> unit
    
    /// <summary>
    /// Reports an error.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="exception">The exception, if any.</param>
    abstract member ReportError : message:string * ?exception:Exception -> unit
    
    /// <summary>
    /// Reports information.
    /// </summary>
    /// <param name="message">The information message.</param>
    abstract member ReportInfo : message:string -> unit
    
    /// <summary>
    /// Reports a success.
    /// </summary>
    /// <param name="message">The success message.</param>
    abstract member ReportSuccess : message:string -> unit
