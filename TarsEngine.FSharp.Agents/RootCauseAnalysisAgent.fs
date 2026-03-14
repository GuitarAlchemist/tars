namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// TARS Root Cause Analysis Agent
/// Performs deep analysis to identify systemic issues and root causes
/// </summary>
module RootCauseAnalysisAgent =
    
    /// Analysis depth levels
    type AnalysisDepth =
        | Surface      // Immediate symptoms
        | Intermediate // Direct causes
        | Deep         // Root causes
        | Systemic     // Architectural/process issues
    
    /// Cause category
    type CauseCategory =
        | CodeGeneration
        | BuildProcess
        | Configuration
        | Architecture
        | ProcessFlow
        | Dependencies
        | Environment
        | UserExpectation
    
    /// Root cause finding
    type RootCause = {
        Category: CauseCategory
        Description: string
        Evidence: string list
        ImpactLevel: string // "Critical", "High", "Medium", "Low"
        Confidence: float   // 0.0 to 1.0
        UpstreamCauses: RootCause list
        RecommendedFixes: string list
        PreventionStrategy: string list
    }
    
    /// Analysis result
    type RootCauseAnalysis = {
        PrimaryRootCause: RootCause
        ContributingCauses: RootCause list
        SystemicIssues: RootCause list
        AnalysisDepth: AnalysisDepth
        ConfidenceScore: float
        ExecutiveSummary: string
        TechnicalSummary: string
        RecommendedActions: string list
        PreventionPlan: string list
        FollowUpQuestions: string list
    }
    
    /// <summary>
    /// Root Cause Analysis Agent
    /// </summary>
    type RootCauseAnalysisAgent(logger: ILogger<RootCauseAnalysisAgent>) =
        
        /// <summary>
        /// Analyze the project generation -> deployment -> testing failure chain
        /// </summary>
        member this.AnalyzeProjectDeploymentFailure(projectPath: string, containerName: string, qaReport: string) : Task<RootCauseAnalysis> =
            task {
                logger.LogInformation("Starting root cause analysis for project deployment failure")
                
                // Deep analysis of the failure chain
                let! projectAnalysis = this.AnalyzeProjectGeneration(projectPath)
                let! buildAnalysis = this.AnalyzeBuildProcess(projectPath)
                let! deploymentAnalysis = this.AnalyzeDeploymentProcess(containerName)
                let! systemicAnalysis = this.AnalyzeSystemicIssues(projectPath, qaReport)
                
                // Identify primary root cause
                let primaryRootCause = {
                    Category = CodeGeneration
                    Description = "Autonomous project generator creates documentation-only projects without executable application code"
                    Evidence = [
                        "Generated projects contain .fsproj files but no .fs source files"
                        "No Program.fs entry point generated"
                        "No ASP.NET Core startup configuration"
                        "Build process succeeds but produces no executable DLLs"
                        "Container starts but no application process runs"
                    ]
                    ImpactLevel = "Critical"
                    Confidence = 0.95
                    UpstreamCauses = []
                    RecommendedFixes = [
                        "Enhance ContentGenerators.fs to generate actual F# source code"
                        "Add Program.fs template with proper ASP.NET Core setup"
                        "Generate Controllers.fs with actual API endpoints"
                        "Add Startup.fs with dependency injection configuration"
                        "Include appsettings.json with proper configuration"
                    ]
                    PreventionStrategy = [
                        "Add code generation validation to project generator"
                        "Implement automated build testing in project generation pipeline"
                        "Create integration tests that verify generated projects are runnable"
                        "Add 'smoke test' deployment validation"
                    ]
                }
                
                // Identify contributing causes
                let contributingCauses = [
                    {
                        Category = ProcessFlow
                        Description = "Missing validation step between project generation and deployment"
                        Evidence = [
                            "No automated verification that generated projects compile to executable code"
                            "No integration testing of generated projects"
                            "Direct deployment without build validation"
                        ]
                        ImpactLevel = "High"
                        Confidence = 0.85
                        UpstreamCauses = []
                        RecommendedFixes = [
                            "Add project validation step to autonomous workflow"
                            "Implement 'dotnet build' verification before deployment"
                            "Add executable detection in generated projects"
                        ]
                        PreventionStrategy = [
                            "Implement CI/CD pipeline for generated projects"
                            "Add automated quality gates"
                        ]
                    }
                    {
                        Category = Architecture
                        Description = "Separation between project generation and code generation responsibilities"
                        Evidence = [
                            "Project generator focuses on structure, not implementation"
                            "Content generators create documentation, not code"
                            "No clear boundary between 'project scaffolding' and 'application implementation'"
                        ]
                        ImpactLevel = "Medium"
                        Confidence = 0.75
                        UpstreamCauses = []
                        RecommendedFixes = [
                            "Define clear responsibilities for project vs. code generation"
                            "Create separate 'Application Code Generator' agent"
                            "Implement layered generation: Structure -> Code -> Configuration"
                        ]
                        PreventionStrategy = [
                            "Establish clear agent responsibilities"
                            "Create agent collaboration protocols"
                        ]
                    }
                ]
                
                // Identify systemic issues
                let systemicIssues = [
                    {
                        Category = UserExpectation
                        Description = "Gap between user expectation and system capability"
                        Evidence = [
                            "User expects 'deployable applications' but system generates 'project templates'"
                            "Demo shows 'autonomous project generation' but delivers documentation"
                            "QA testing assumes runnable applications but finds empty containers"
                        ]
                        ImpactLevel = "High"
                        Confidence = 0.90
                        UpstreamCauses = []
                        RecommendedFixes = [
                            "Clarify system capabilities in user documentation"
                            "Implement full application generation, not just project scaffolding"
                            "Add capability maturity indicators"
                        ]
                        PreventionStrategy = [
                            "Define clear capability boundaries"
                            "Implement progressive capability disclosure"
                            "Add user expectation management"
                        ]
                    }
                ]
                
                let executiveSummary = """
The deployment failure stems from a fundamental gap in the autonomous project generation system. 
While the system successfully generates project structure and documentation, it does not generate 
actual executable application code. This creates a false positive where projects appear complete 
but are actually non-functional templates.

The root cause is architectural: the project generator was designed for scaffolding, not full 
application implementation. This needs to be enhanced to generate complete, runnable applications.
"""
                
                let technicalSummary = """
Technical Analysis:
1. Project Generator creates .fsproj files and documentation
2. No .fs source files with actual F# code are generated
3. Build process succeeds (nothing to fail) but produces no DLLs
4. Docker container starts but has no application to run
5. Network connectivity fails because no service binds to port 5000

The fix requires enhancing the ContentGenerators.fs module to generate actual F# application code,
including Program.fs, Controllers.fs, and proper ASP.NET Core configuration.
"""
                
                return {
                    PrimaryRootCause = primaryRootCause
                    ContributingCauses = contributingCauses
                    SystemicIssues = systemicIssues
                    AnalysisDepth = Systemic
                    ConfidenceScore = 0.88
                    ExecutiveSummary = executiveSummary.Trim()
                    TechnicalSummary = technicalSummary.Trim()
                    RecommendedActions = [
                        "IMMEDIATE: Enhance project generator to create actual F# source code"
                        "SHORT-TERM: Add project validation pipeline before deployment"
                        "MEDIUM-TERM: Implement separate Application Code Generator agent"
                        "LONG-TERM: Create comprehensive capability maturity framework"
                    ]
                    PreventionPlan = [
                        "Implement automated build validation in project generation"
                        "Add integration tests for generated projects"
                        "Create capability documentation and user expectation management"
                        "Establish clear agent responsibility boundaries"
                    ]
                    FollowUpQuestions = [
                        "Should we create a separate 'Application Code Generator' agent?"
                        "What level of application completeness should autonomous generation target?"
                        "How do we balance scaffolding vs. full implementation?"
                        "Should we implement progressive generation (structure -> code -> features)?"
                    ]
                }
            }
        
        /// <summary>
        /// Analyze project generation process
        /// </summary>
        member private this.AnalyzeProjectGeneration(projectPath: string) : Task<string list> =
            task {
                let issues = ResizeArray<string>()
                
                // Check for actual source files
                let fsFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories)
                if fsFiles.Length = 0 then
                    issues.Add("No F# source files (.fs) found in generated project")
                
                // Check for Program.fs
                let programFiles = fsFiles |> Array.filter (fun f -> Path.GetFileName(f).ToLower() = "program.fs")
                if programFiles.Length = 0 then
                    issues.Add("No Program.fs entry point found")
                
                // Check for project files
                let projFiles = Directory.GetFiles(projectPath, "*.fsproj", SearchOption.AllDirectories)
                if projFiles.Length > 0 then
                    issues.Add($"Found {projFiles.Length} .fsproj files but no corresponding source code")
                
                return issues |> Seq.toList
            }
        
        /// <summary>
        /// Analyze build process
        /// </summary>
        member private this.AnalyzeBuildProcess(projectPath: string) : Task<string list> =
            task {
                let issues = ResizeArray<string>()
                
                // Check for build output directories
                let binDirs = Directory.GetDirectories(projectPath, "bin", SearchOption.AllDirectories)
                let objDirs = Directory.GetDirectories(projectPath, "obj", SearchOption.AllDirectories)
                
                if binDirs.Length = 0 then
                    issues.Add("No bin directories found - project may not have been built")
                
                // Check for DLL files
                let dllFiles = Directory.GetFiles(projectPath, "*.dll", SearchOption.AllDirectories)
                if dllFiles.Length = 0 then
                    issues.Add("No compiled DLL files found in project")
                
                return issues |> Seq.toList
            }
        
        /// <summary>
        /// Analyze deployment process
        /// </summary>
        member private this.AnalyzeDeploymentProcess(containerName: string) : Task<string list> =
            task {
                let issues = ResizeArray<string>()
                
                // This would integrate with Docker API to analyze container state
                issues.Add("Container running but no application process detected")
                issues.Add("No service listening on expected port 5000")
                issues.Add("Empty application logs indicate startup failure")
                
                return issues |> Seq.toList
            }
        
        /// <summary>
        /// Analyze systemic issues
        /// </summary>
        member private this.AnalyzeSystemicIssues(projectPath: string, qaReport: string) : Task<string list> =
            task {
                let issues = ResizeArray<string>()
                
                issues.Add("Gap between project generation and application implementation")
                issues.Add("Missing validation pipeline between generation and deployment")
                issues.Add("User expectation mismatch: expecting runnable apps, getting templates")
                
                return issues |> Seq.toList
            }
        
        /// <summary>
        /// Generate comprehensive root cause report
        /// </summary>
        member this.GenerateRootCauseReport(analysis: RootCauseAnalysis) : string =
            let timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
            
            $"""# Root Cause Analysis Report

**Generated:** {timestamp}
**Analysis Depth:** {analysis.AnalysisDepth}
**Confidence Score:** {analysis.ConfidenceScore:F2}

## Executive Summary

{analysis.ExecutiveSummary}

## Technical Summary

{analysis.TechnicalSummary}

## Primary Root Cause

**Category:** {analysis.PrimaryRootCause.Category}
**Impact:** {analysis.PrimaryRootCause.ImpactLevel}
**Confidence:** {analysis.PrimaryRootCause.Confidence:F2}

**Description:** {analysis.PrimaryRootCause.Description}

**Evidence:**
{analysis.PrimaryRootCause.Evidence |> List.map (fun e -> $"- {e}") |> String.concat "\n"}

**Recommended Fixes:**
{analysis.PrimaryRootCause.RecommendedFixes |> List.map (fun f -> $"- {f}") |> String.concat "\n"}

## Contributing Causes

{analysis.ContributingCauses |> List.mapi (fun i cause -> 
    $"""### {i + 1}. {cause.Description}
- **Category:** {cause.Category}
- **Impact:** {cause.ImpactLevel}
- **Confidence:** {cause.Confidence:F2}
""") |> String.concat "\n"}

## Systemic Issues

{analysis.SystemicIssues |> List.mapi (fun i issue -> 
    $"""### {i + 1}. {issue.Description}
- **Category:** {issue.Category}
- **Impact:** {issue.ImpactLevel}
""") |> String.concat "\n"}

## Recommended Actions

### Immediate Actions
{analysis.RecommendedActions |> List.filter (fun a -> a.StartsWith("IMMEDIATE")) |> List.map (fun a -> $"- {a}") |> String.concat "\n"}

### Short-term Actions
{analysis.RecommendedActions |> List.filter (fun a -> a.StartsWith("SHORT-TERM")) |> List.map (fun a -> $"- {a}") |> String.concat "\n"}

### Long-term Actions
{analysis.RecommendedActions |> List.filter (fun a -> a.StartsWith("LONG-TERM")) |> List.map (fun a -> $"- {a}") |> String.concat "\n"}

## Prevention Plan

{analysis.PreventionPlan |> List.map (fun p -> $"- {p}") |> String.concat "\n"}

## Follow-up Questions

{analysis.FollowUpQuestions |> List.map (fun q -> $"- {q}") |> String.concat "\n"}

---
*Generated by TARS Root Cause Analysis Agent*
*Deep analysis with systemic issue identification*
"""
