#!/usr/bin/env dotnet fsi

// DETAILED EXPLORATION ANALYSIS SYSTEM
// Comprehensive analysis and path planning before code generation

open System
open System.IO
open System.Text.Json

type ExplorationComplexity = Simple | Medium | Complex | VeryComplex
type ExplorationCategory = WebApp | DesktopApp | API | Library | Tool | Game | AI_ML | DataProcessing

type ExplorationAnalysis = {
    Id: string
    Description: string
    Category: ExplorationCategory
    Complexity: ExplorationComplexity
    EstimatedLinesOfCode: int
    RequiredFeatures: string list
    TechnicalChallenges: string list
    Dependencies: string list
    TargetFramework: string
    ExpectedDuration: string
    BusinessValue: string
    RiskLevel: string
    FullProjectPath: string
    SummaryPath: string
    MetadataPath: string
}

type ExplorationSummary = {
    Analysis: ExplorationAnalysis
    ArchitecturalDecisions: string list
    ImplementationPlan: string list
    QualityGates: string list
    PromotionCriteria: string list
    GeneratedAt: DateTime
}

module ExplorationAnalyzer =
    let analyzeExploration (description: string) : ExplorationAnalysis =
        printfn "🔍 [ANALYSIS] Analyzing exploration: %s" description
        printfn "=============================================================="
        
        let id = Guid.NewGuid().ToString("N")[..7]
        let timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
        
        // Determine category based on keywords
        let category = 
            let desc = description.ToLower()
            if desc.Contains("web") || desc.Contains("api") || desc.Contains("rest") then API
            elif desc.Contains("chat") || desc.Contains("messaging") || desc.Contains("real-time") then WebApp
            elif desc.Contains("desktop") || desc.Contains("gui") || desc.Contains("ui") then DesktopApp
            elif desc.Contains("ml") || desc.Contains("ai") || desc.Contains("machine learning") then AI_ML
            elif desc.Contains("data") || desc.Contains("processing") || desc.Contains("analytics") then DataProcessing
            elif desc.Contains("game") || desc.Contains("gaming") then Game
            elif desc.Contains("library") || desc.Contains("framework") then Library
            else Tool
        
        // Determine complexity based on description length and keywords
        let complexity =
            let desc = description.ToLower()
            let complexKeywords = ["distributed"; "real-time"; "machine learning"; "ai"; "blockchain"; "microservice"]
            let mediumKeywords = ["database"; "authentication"; "api"; "websocket"; "sync"]
            let complexCount = complexKeywords |> List.filter desc.Contains |> List.length
            let mediumCount = mediumKeywords |> List.filter desc.Contains |> List.length

            if complexCount > 0 || description.Length > 100 then
                VeryComplex
            elif mediumCount > 0 || description.Length > 60 then
                Complex
            elif description.Length > 30 then
                Medium
            else
                Simple
        
        // Estimate lines of code
        let estimatedLOC =
            match complexity with
            | Simple -> 50 + Random().Next(50)
            | Medium -> 100 + Random().Next(100)
            | Complex -> 200 + Random().Next(200)
            | VeryComplex -> 400 + Random().Next(400)
        
        // Extract features from description
        let extractFeatures (desc: string) =
            let features = ResizeArray<string>()
            let desc = desc.ToLower()
            
            if desc.Contains("real-time") then features.Add("Real-time processing")
            if desc.Contains("chat") || desc.Contains("messaging") then features.Add("Messaging system")
            if desc.Contains("user") then features.Add("User management")
            if desc.Contains("database") || desc.Contains("storage") then features.Add("Data persistence")
            if desc.Contains("api") then features.Add("API endpoints")
            if desc.Contains("web") then features.Add("Web interface")
            if desc.Contains("sync") then features.Add("Synchronization")
            if desc.Contains("file") then features.Add("File operations")
            if desc.Contains("ml") || desc.Contains("ai") then features.Add("Machine learning")
            if desc.Contains("visualization") then features.Add("Data visualization")
            
            features |> Seq.toList
        
        let requiredFeatures = extractFeatures description
        
        // Determine technical challenges
        let technicalChallenges = 
            match complexity with
            | Simple -> ["Basic F# syntax"; "Simple data structures"]
            | Medium -> ["Error handling"; "Data validation"; "Basic concurrency"]
            | Complex -> ["Advanced concurrency"; "Performance optimization"; "Complex data models"]
            | VeryComplex -> ["Distributed systems"; "Advanced algorithms"; "Scalability"; "Real-time constraints"]
        
        // Determine dependencies
        let dependencies = 
            let deps = ResizeArray<string>()
            if requiredFeatures |> List.exists (fun f -> f.Contains("Web")) then deps.Add("ASP.NET Core")
            if requiredFeatures |> List.exists (fun f -> f.Contains("Data")) then deps.Add("Entity Framework")
            if requiredFeatures |> List.exists (fun f -> f.Contains("Real-time")) then deps.Add("SignalR")
            if requiredFeatures |> List.exists (fun f -> f.Contains("Machine learning")) then deps.Add("ML.NET")
            deps.Add("FSharp.Core")
            deps |> Seq.toList
        
        // Create full project structure paths
        let projectName = sprintf "Exploration_%s_%s" id timestamp
        let baseProjectPath = Path.Combine(".tars", "explorations", "detailed", projectName)
        let summaryPath = Path.Combine(baseProjectPath, "analysis", "exploration-summary.md")
        let metadataPath = Path.Combine(baseProjectPath, "analysis", "metadata.json")
        
        let analysis = {
            Id = id
            Description = description
            Category = category
            Complexity = complexity
            EstimatedLinesOfCode = estimatedLOC
            RequiredFeatures = requiredFeatures
            TechnicalChallenges = technicalChallenges
            Dependencies = dependencies
            TargetFramework = "net8.0"
            ExpectedDuration = match complexity with
                | Simple -> "1-2 hours"
                | Medium -> "4-6 hours"
                | Complex -> "1-2 days"
                | VeryComplex -> "3-5 days"
            BusinessValue = match complexity with
                | Simple -> "Low-Medium"
                | Medium -> "Medium"
                | Complex -> "Medium-High"
                | VeryComplex -> "High"
            RiskLevel = match complexity with
                | Simple -> "Low"
                | Medium -> "Low-Medium"
                | Complex -> "Medium"
                | VeryComplex -> "Medium-High"
            FullProjectPath = baseProjectPath
            SummaryPath = summaryPath
            MetadataPath = metadataPath
        }
        
        printfn "📊 [ANALYSIS] Exploration Analysis Complete"
        printfn "  ID: %s" analysis.Id
        printfn "  Category: %A" analysis.Category
        printfn "  Complexity: %A" analysis.Complexity
        printfn "  Estimated LOC: %d" analysis.EstimatedLinesOfCode
        printfn "  Expected Duration: %s" analysis.ExpectedDuration
        printfn "  Business Value: %s" analysis.BusinessValue
        printfn "  Risk Level: %s" analysis.RiskLevel
        printfn ""
        
        analysis
    
    let createDetailedSummary (analysis: ExplorationAnalysis) : ExplorationSummary =
        printfn "📋 [SUMMARY] Creating detailed exploration summary..."
        
        let architecturalDecisions = [
            sprintf "Target Framework: %s" analysis.TargetFramework
            sprintf "Primary Language: F#"
            sprintf "Architecture Pattern: %s" (match analysis.Complexity with
                | Simple -> "Simple functional design"
                | Medium -> "Modular functional architecture"
                | Complex -> "Layered functional architecture"
                | VeryComplex -> "Domain-driven functional design")
            sprintf "Deployment Model: Blue node → QA validation → Green promotion"
        ]
        
        let implementationPlan = [
            "Phase 1: Core domain model definition"
            "Phase 2: Business logic implementation"
            "Phase 3: Infrastructure and I/O operations"
            "Phase 4: User interface and interaction layer"
            "Phase 5: Testing and validation"
            "Phase 6: Documentation and deployment preparation"
        ]
        
        let qualityGates = [
            "✅ Code compiles without errors or warnings"
            "✅ All unit tests pass"
            "✅ Code coverage > 80%"
            "✅ Performance benchmarks met"
            "✅ Security scan passes"
            "✅ Documentation complete"
        ]
        
        let promotionCriteria = [
            sprintf "Overall QA score ≥ 0.8 (targeting complexity: %A)" analysis.Complexity
            "Compilation success rate: 100%"
            "Runtime stability: No crashes in 10 test runs"
            "Feature completeness: All required features implemented"
            "Code quality: Follows F# best practices"
            "Performance: Meets expected benchmarks"
        ]
        
        {
            Analysis = analysis
            ArchitecturalDecisions = architecturalDecisions
            ImplementationPlan = implementationPlan
            QualityGates = qualityGates
            PromotionCriteria = promotionCriteria
            GeneratedAt = DateTime.UtcNow
        }
    
    let createProjectStructure (summary: ExplorationSummary) =
        let analysis = summary.Analysis
        printfn "📁 [STRUCTURE] Creating project structure at: %s" analysis.FullProjectPath
        printfn "=================================================================="
        
        // Create directory structure
        Directory.CreateDirectory(analysis.FullProjectPath) |> ignore
        Directory.CreateDirectory(Path.Combine(analysis.FullProjectPath, "src")) |> ignore
        Directory.CreateDirectory(Path.Combine(analysis.FullProjectPath, "tests")) |> ignore
        Directory.CreateDirectory(Path.Combine(analysis.FullProjectPath, "docs")) |> ignore
        Directory.CreateDirectory(Path.Combine(analysis.FullProjectPath, "analysis")) |> ignore
        Directory.CreateDirectory(Path.Combine(analysis.FullProjectPath, "scripts")) |> ignore
        
        printfn "📁 Created directories:"
        printfn "  📂 %s/src/ (source code)" analysis.FullProjectPath
        printfn "  📂 %s/tests/ (unit tests)" analysis.FullProjectPath
        printfn "  📂 %s/docs/ (documentation)" analysis.FullProjectPath
        printfn "  📂 %s/analysis/ (exploration analysis)" analysis.FullProjectPath
        printfn "  📂 %s/scripts/ (build and deployment)" analysis.FullProjectPath
        
        // Create detailed summary markdown
        let summaryMarkdown = sprintf """# Exploration Analysis: %s

**Generated:** %s  
**Exploration ID:** %s  
**Full Project Path:** `%s`

## 📊 Analysis Overview

| Attribute | Value |
|-----------|-------|
| **Category** | %A |
| **Complexity** | %A |
| **Estimated LOC** | %d lines |
| **Expected Duration** | %s |
| **Business Value** | %s |
| **Risk Level** | %s |
| **Target Framework** | %s |

## 🎯 Required Features

%s

## 🚧 Technical Challenges

%s

## 📦 Dependencies

%s

## 🏗️ Architectural Decisions

%s

## 📋 Implementation Plan

%s

## ✅ Quality Gates

%s

## 🎯 Promotion Criteria

%s

## 📁 Project Structure

```
%s/
├── src/                    # Source code
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── analysis/              # Exploration analysis
│   ├── exploration-summary.md
│   └── metadata.json
└── scripts/               # Build and deployment scripts
```

## 🔄 Development Workflow

1. **Blue Node Creation** - Generate experimental implementation
2. **QA Validation** - Run comprehensive quality assurance
3. **Green Promotion** - Promote to stable green node if passing
4. **Continuous Monitoring** - Track performance and stability

---
*Generated by TARS Autonomous Exploration Analysis System*
""" 
            analysis.Description
            (summary.GeneratedAt.ToString("yyyy-MM-dd HH:mm:ss"))
            analysis.Id
            analysis.FullProjectPath
            analysis.Category
            analysis.Complexity
            analysis.EstimatedLinesOfCode
            analysis.ExpectedDuration
            analysis.BusinessValue
            analysis.RiskLevel
            analysis.TargetFramework
            (analysis.RequiredFeatures |> List.map (sprintf "- %s") |> String.concat "\n")
            (analysis.TechnicalChallenges |> List.map (sprintf "- %s") |> String.concat "\n")
            (analysis.Dependencies |> List.map (sprintf "- %s") |> String.concat "\n")
            (summary.ArchitecturalDecisions |> List.map (sprintf "- %s") |> String.concat "\n")
            (summary.ImplementationPlan |> List.mapi (fun i plan -> sprintf "%d. %s" (i+1) plan) |> String.concat "\n")
            (summary.QualityGates |> List.map (sprintf "- %s") |> String.concat "\n")
            (summary.PromotionCriteria |> List.map (sprintf "- %s") |> String.concat "\n")
            analysis.FullProjectPath
        
        File.WriteAllText(analysis.SummaryPath, summaryMarkdown)
        
        // Create metadata JSON
        let metadataJson = JsonSerializer.Serialize(summary, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(analysis.MetadataPath, metadataJson)
        
        printfn "📄 Created analysis files:"
        printfn "  📋 %s" analysis.SummaryPath
        printfn "  📊 %s" analysis.MetadataPath
        
        summary

// Demo: Detailed Exploration Analysis
printfn "🔍📊 DETAILED EXPLORATION ANALYSIS SYSTEM"
printfn "=========================================="
printfn ""

let explorations = [
    "Create a real-time collaborative document editor with WebSocket support"
    "Build a distributed microservices architecture with API gateway and service discovery"
    "Develop a machine learning pipeline for sentiment analysis with data visualization"
    "Design a blockchain-based cryptocurrency wallet with transaction history"
]

printfn "🎯 Analyzing %d explorations with full detail..." explorations.Length
printfn ""

let analysisResults = 
    explorations 
    |> List.mapi (fun i exploration ->
        printfn "🔍 [%d/%d] EXPLORATION ANALYSIS" (i+1) explorations.Length
        printfn "================================"
        
        let analysis = ExplorationAnalyzer.analyzeExploration exploration
        let summary = ExplorationAnalyzer.createDetailedSummary analysis
        let finalSummary = ExplorationAnalyzer.createProjectStructure summary
        
        printfn "✅ [COMPLETE] Analysis complete for exploration %d" (i+1)
        printfn "📁 Full path: %s" analysis.FullProjectPath
        printfn "📋 Summary: %s" analysis.SummaryPath
        printfn ""
        
        finalSummary)

printfn "📊 [FINAL SUMMARY] Detailed Analysis Complete"
printfn "============================================="
printfn "🔍 Explorations Analyzed: %d" analysisResults.Length

analysisResults |> List.iteri (fun i result ->
    let analysis = result.Analysis
    printfn ""
    printfn "%d. %s" (i+1) analysis.Description
    printfn "   📁 Path: %s" analysis.FullProjectPath
    printfn "   📊 Complexity: %A | LOC: %d | Duration: %s" analysis.Complexity analysis.EstimatedLinesOfCode analysis.ExpectedDuration
    printfn "   💼 Business Value: %s | Risk: %s" analysis.BusinessValue analysis.RiskLevel
    printfn "   📋 Summary: %s" analysis.SummaryPath)

printfn ""
printfn "✅ All explorations analyzed with full detail!"
printfn "📁 Project structures created with comprehensive documentation!"
printfn "🎯 Ready for detailed code generation with complete context!"
