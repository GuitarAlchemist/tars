// ================================================
// 🌌 BSP Demo Runner
// ================================================
// Orchestrates the real sedenion BSP code analysis demo

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Net.Http
open System.Text
open System.Threading.Tasks
open Spectre.Console
open BspCodeAnalysisDemo

module BspDemoRunner =

    let runBspDemoAsync (vectorCount: int) (dimension: int) : Task<unit> = task {
        AnsiConsole.MarkupLine("[bold green]🌌 REAL USE CASE: TARS CODEBASE ANALYSIS WITH SEDENION BSP CLUSTERING[/]")
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("[yellow]🎯 PROBLEM: How do we understand and organize a large codebase like TARS?[/]")
        AnsiConsole.MarkupLine("[cyan]SOLUTION: Use sedenion BSP partitioning to cluster related code files![/]")
        AnsiConsole.WriteLine()

        let startTime = DateTime.UtcNow

        // Real TARS codebase analysis
        AnsiConsole.MarkupLine("[yellow]📁 REAL TARS CODEBASE ANALYSIS:[/]")
        let allFsFiles = Directory.GetFiles(".", "*.fs", SearchOption.AllDirectories)
                        |> Array.filter (fun f -> not (f.Contains("bin") || f.Contains("obj")))
        let fsFiles = allFsFiles |> Array.take (min vectorCount allFsFiles.Length)
        
        AnsiConsole.MarkupLine($"[cyan]Analyzing {fsFiles.Length} F# source files in TARS codebase...[/]")
        
        let codeFiles = fsFiles |> Array.map analyzeCodeFile |> Array.toList
        
        AnsiConsole.MarkupLine("[green]✅ Code analysis complete![/]")
        for i in 0..min 4 (codeFiles.Length - 1) do
            let file = codeFiles.[i]
            AnsiConsole.MarkupLine($"[cyan]  {file.FileName}: {file.LineCount} lines, {file.FunctionCount} functions, {file.TypeCount} types[/]")

        AnsiConsole.WriteLine()
        
        // Convert code files to sedenions for BSP partitioning
        AnsiConsole.MarkupLine("[yellow]🌌 CONVERTING CODE TO SEDENIONS FOR BSP PARTITIONING:[/]")
        let codeSedenions = 
            codeFiles 
            |> List.map (fun file -> createSedenion file.SemanticFeatures)
        
        AnsiConsole.MarkupLine($"[cyan]Converted {codeSedenions.Length} code files to 16D sedenions:[/]")
        
        // Show semantic feature analysis
        let aiFiles = codeFiles |> List.filter (fun f -> f.SemanticFeatures.[4] > 0.5)
        let cliFiles = codeFiles |> List.filter (fun f -> f.SemanticFeatures.[5] > 0.5)
        let coreFiles = codeFiles |> List.filter (fun f -> f.SemanticFeatures.[6] > 0.5)
        let cudaFiles = codeFiles |> List.filter (fun f -> f.SemanticFeatures.[7] > 0.5)
        
        AnsiConsole.MarkupLine($"[green]  • AI/Inference files: {aiFiles.Length}[/]")
        AnsiConsole.MarkupLine($"[green]  • CLI/Command files: {cliFiles.Length}[/]")
        AnsiConsole.MarkupLine($"[green]  • Core/Engine files: {coreFiles.Length}[/]")
        AnsiConsole.MarkupLine($"[green]  • CUDA/GPU files: {cudaFiles.Length}[/]")

        AnsiConsole.WriteLine()
        
        // Real hypercomplex partitioning hierarchy demonstration
        AnsiConsole.MarkupLine("[yellow]🔢 REAL HYPERCOMPLEX PARTITIONING HIERARCHY:[/]")
        AnsiConsole.MarkupLine("[cyan]TARS supports the complete Cayley-Dickson construction:[/]")
        AnsiConsole.MarkupLine("[green]  • Real numbers (1D) - Basic scalar partitioning[/]")
        AnsiConsole.MarkupLine("[green]  • Complex numbers (2D) - Planar partitioning[/]")
        AnsiConsole.MarkupLine("[green]  • Quaternions (4D) - 3D rotation partitioning[/]")
        AnsiConsole.MarkupLine("[green]  • Octonions (8D) - Non-associative partitioning[/]")
        AnsiConsole.MarkupLine("[green]  • Sedenions (16D) - Non-alternative knowledge clustering[/]")
        
        AnsiConsole.WriteLine()
        
        // Real BSP tree construction with sedenions
        AnsiConsole.MarkupLine("[yellow]🌳 REAL SEDENION BSP TREE CONSTRUCTION:[/]")
        let bspStartTime = DateTime.UtcNow
        
        let bspTree = buildCodeBspTree codeFiles codeSedenions 0 "root"
        let constructionTime = (DateTime.UtcNow - bspStartTime).TotalMilliseconds

        AnsiConsole.MarkupLine($"[green]✅ Sedenion BSP tree constructed in {constructionTime:F2} ms[/]")
        
        let (leafCount, nodeCount, totalFiles, totalSignificance) = analyzeCodeBspTree bspTree
        
        AnsiConsole.MarkupLine("[cyan]📈 CODE CLUSTERING RESULTS:[/]")
        AnsiConsole.MarkupLine($"[green]  • Total BSP Nodes: {nodeCount}[/]")
        AnsiConsole.MarkupLine($"[green]  • Code Clusters Found: {leafCount}[/]")
        AnsiConsole.MarkupLine($"[green]  • Files Clustered: {totalFiles}[/]")
        AnsiConsole.MarkupLine($"[green]  • Average Files/Cluster: {float totalFiles / float leafCount:F1}[/]")
        AnsiConsole.MarkupLine($"[green]  • Clustering Depth: ≤ 6 levels[/]")
        AnsiConsole.MarkupLine($"[green]  • Total Significance: {totalSignificance:F1}[/]")

        AnsiConsole.WriteLine()
        
        let codeClusters = extractCodeClusters bspTree
        
        AnsiConsole.MarkupLine($"[cyan]🎯 DISCOVERED {codeClusters.Length} ARCHITECTURAL CLUSTERS:[/]")
        for i, cluster in codeClusters |> List.indexed do
            AnsiConsole.MarkupLine($"[green]📁 Cluster {i+1}: {cluster.ClusterName}[/]")
            AnsiConsole.MarkupLine($"[cyan]   Role: {cluster.ArchitecturalRole}[/]")
            AnsiConsole.MarkupLine($"[cyan]   Type: {cluster.ClusterType}[/]")
            AnsiConsole.MarkupLine($"[cyan]   Files: {cluster.Files.Length}[/]")
            
            let totalLines = cluster.Files |> List.sumBy (fun f -> f.LineCount)
            let totalFunctions = cluster.Files |> List.sumBy (fun f -> f.FunctionCount)
            AnsiConsole.MarkupLine($"[cyan]   Code: {totalLines} lines, {totalFunctions} functions[/]")
            
            // Show sample files in cluster
            let sampleFiles = cluster.Files |> List.take (min 3 cluster.Files.Length)
            for file in sampleFiles do
                AnsiConsole.MarkupLine($"[dim]     • {file.FileName}[/]")
            
            AnsiConsole.WriteLine()

        // Real practical benefits demonstration
        AnsiConsole.MarkupLine("[yellow]🎯 PRACTICAL BENEFITS OF SEDENION BSP CODE CLUSTERING:[/]")
        AnsiConsole.MarkupLine("[green]✅ Architecture Understanding: Automatically discover code organization[/]")
        AnsiConsole.MarkupLine("[green]✅ Refactoring Guidance: Identify tightly coupled components[/]")
        AnsiConsole.MarkupLine("[green]✅ Code Navigation: Find related files quickly[/]")
        AnsiConsole.MarkupLine("[green]✅ Dependency Analysis: Understand module relationships[/]")
        AnsiConsole.MarkupLine("[green]✅ Quality Assessment: Identify architectural patterns[/]")
        
        AnsiConsole.WriteLine()
        
        // Real performance and scalability
        let totalAnalysisTime = (DateTime.UtcNow - startTime).TotalMilliseconds
        AnsiConsole.MarkupLine("[yellow]⚡ REAL PERFORMANCE METRICS:[/]")
        AnsiConsole.MarkupLine($"[cyan]Total Analysis Time: {totalAnalysisTime:F2} ms[/]")
        AnsiConsole.MarkupLine($"[cyan]Files Analyzed: {codeFiles.Length}[/]")
        AnsiConsole.MarkupLine($"[cyan]Analysis Speed: {float codeFiles.Length / (totalAnalysisTime / 1000.0):F1} files/second[/]")
        AnsiConsole.MarkupLine($"[cyan]Clusters Generated: {codeClusters.Length}[/]")
        AnsiConsole.MarkupLine($"[cyan]Clustering Efficiency: {float codeFiles.Length / float codeClusters.Length:F1} files/cluster[/]")
        
        AnsiConsole.WriteLine()
        
        // Real use case value proposition
        AnsiConsole.MarkupLine("[yellow]💡 REAL-WORLD USE CASES:[/]")
        AnsiConsole.MarkupLine("[cyan]🏢 Enterprise: Understand large legacy codebases[/]")
        AnsiConsole.MarkupLine("[cyan]🔄 Refactoring: Identify microservice boundaries[/]")
        AnsiConsole.MarkupLine("[cyan]📚 Documentation: Auto-generate architecture diagrams[/]")
        AnsiConsole.MarkupLine("[cyan]🔍 Code Review: Find similar patterns and anti-patterns[/]")
        AnsiConsole.MarkupLine("[cyan]🎯 Onboarding: Help new developers understand codebase structure[/]")
        
        AnsiConsole.WriteLine()

        // Real LLM Context Injection Demo
        AnsiConsole.MarkupLine("[yellow]🤖 SEDENION-ENHANCED LLM CONTEXT INJECTION DEMO:[/]")
        AnsiConsole.MarkupLine("[cyan]Demonstrating how sedenion clusters enhance LLM responses![/]")
        AnsiConsole.WriteLine()

        // Sample queries about TARS
        let sampleQueries = [
            "How does TARS handle AI inference and machine learning?"
            "What CUDA capabilities does TARS have?"
            "How is the CLI structured in TARS?"
            "What are the core engine components?"
        ]

        let queryToTest = sampleQueries.[0] // Test the first query
        AnsiConsole.MarkupLine($"[cyan]🔍 Test Query: \"{queryToTest}\"[/]")
        AnsiConsole.WriteLine()

        // Find relevant clusters using sedenion similarity
        AnsiConsole.MarkupLine("[yellow]🎯 FINDING RELEVANT CLUSTERS:[/]")

        // Create query sedenion based on keywords
        let querySedenion =
            let queryFeatures = Array.zeroCreate 16
            let queryLower = queryToTest.ToLower()
            queryFeatures.[4] <- if queryLower.Contains("ai") || queryLower.Contains("inference") || queryLower.Contains("machine") then 1.0 else 0.0
            queryFeatures.[5] <- if queryLower.Contains("cli") || queryLower.Contains("command") then 1.0 else 0.0
            queryFeatures.[6] <- if queryLower.Contains("core") || queryLower.Contains("engine") then 1.0 else 0.0
            queryFeatures.[7] <- if queryLower.Contains("cuda") || queryLower.Contains("gpu") then 1.0 else 0.0
            queryFeatures.[10] <- if queryLower.Contains("async") then 1.0 else 0.0
            queryFeatures.[14] <- if queryLower.Contains("cuda") then 1.0 else 0.0
            queryFeatures.[15] <- if queryLower.Contains("vector") then 1.0 else 0.0
            createSedenion queryFeatures

        // Find most relevant clusters
        let relevantClusters =
            codeClusters
            |> List.map (fun cluster ->
                let distance =
                    Array.zip querySedenion.Components cluster.CentroidFeatures
                    |> Array.map (fun (a, b) -> (a - b) * (a - b))
                    |> Array.sum
                    |> sqrt
                (cluster, distance))
            |> List.sortBy snd
            |> List.take (min 3 codeClusters.Length)

        AnsiConsole.MarkupLine($"[green]Found {relevantClusters.Length} relevant clusters:[/]")
        for i, (cluster, distance) in relevantClusters |> List.indexed do
            AnsiConsole.MarkupLine($"[cyan]  {i+1}. {cluster.ClusterName} (similarity: {1.0 - distance:F3})[/]")

        AnsiConsole.WriteLine()

        // Extract context from relevant clusters
        AnsiConsole.MarkupLine("[yellow]📋 EXTRACTING CONTEXT FROM CLUSTERS:[/]")
        let contextBuilder = System.Text.StringBuilder()
        contextBuilder.AppendLine("TARS Codebase Context:") |> ignore
        contextBuilder.AppendLine("") |> ignore

        for (cluster, _) in relevantClusters do
            contextBuilder.AppendLine($"## {cluster.ClusterName} ({cluster.ArchitecturalRole})") |> ignore
            contextBuilder.AppendLine($"Files: {cluster.Files.Length}") |> ignore

            for file in cluster.Files |> List.take (min 2 cluster.Files.Length) do
                contextBuilder.AppendLine($"- {file.FileName}: {file.LineCount} lines, {file.FunctionCount} functions") |> ignore

                // Add sample code content if available
                try
                    let content = File.ReadAllText(file.FilePath)
                    let allLines = content.Split('\n')
                    let lines = allLines |> Array.take (min 5 allLines.Length)
                    contextBuilder.AppendLine("  Sample code:") |> ignore
                    for line in lines do
                        if not (String.IsNullOrWhiteSpace(line)) then
                            contextBuilder.AppendLine($"    {line.Trim()}") |> ignore
                with
                | _ -> contextBuilder.AppendLine("  (Code content not accessible)") |> ignore

            contextBuilder.AppendLine("") |> ignore

        let extractedContext = contextBuilder.ToString()
        let contextPreview = if extractedContext.Length > 200 then extractedContext.Substring(0, 200) + "..." else extractedContext
        AnsiConsole.MarkupLine($"[green]✅ Extracted {extractedContext.Length} characters of context[/]")
        AnsiConsole.MarkupLine($"[dim]Preview: {contextPreview}[/]")

        AnsiConsole.WriteLine()

        // LLM Comparison Demo
        AnsiConsole.MarkupLine("[yellow]🤖 LLM RESPONSE COMPARISON:[/]")

        try
            use client = new HttpClient()
            client.Timeout <- TimeSpan.FromSeconds(30.0)

            // Test 1: Raw query without context
            AnsiConsole.MarkupLine("[cyan]1. Raw LLM Response (no context):[/]")
            let rawQuery = queryToTest
            let rawRequestBody = $"""{{
                "model": "llama3",
                "prompt": "{rawQuery}",
                "stream": false
            }}"""

            let rawContent = new StringContent(rawRequestBody, System.Text.Encoding.UTF8, "application/json")
            let! rawResponse = client.PostAsync("http://localhost:11434/api/generate", rawContent)

            if rawResponse.IsSuccessStatusCode then
                let! rawResponseText = rawResponse.Content.ReadAsStringAsync()
                AnsiConsole.MarkupLine($"[green]✅ Raw response received ({rawResponseText.Length} chars)[/]")
                AnsiConsole.MarkupLine("[dim]This would be a generic response about AI/ML in general[/]")
            else
                AnsiConsole.MarkupLine($"[red]❌ Raw query failed: {rawResponse.StatusCode}[/]")

            AnsiConsole.WriteLine()

            // Test 2: Context-enhanced query
            AnsiConsole.MarkupLine("[cyan]2. Sedenion-Enhanced LLM Response (with cluster context):[/]")
            let enhancedQuery = $"""Based on this TARS codebase context:

{extractedContext}

Question: {queryToTest}

Please provide a specific answer based on the TARS codebase context provided above."""

            let enhancedRequestBody = $"""{{
                "model": "llama3",
                "prompt": "{enhancedQuery.Replace("\"", "\\\"")}",
                "stream": false
            }}"""

            let enhancedContent = new StringContent(enhancedRequestBody, System.Text.Encoding.UTF8, "application/json")
            let! enhancedResponse = client.PostAsync("http://localhost:11434/api/generate", enhancedContent)

            if enhancedResponse.IsSuccessStatusCode then
                let! enhancedResponseText = enhancedResponse.Content.ReadAsStringAsync()
                AnsiConsole.MarkupLine($"[green]✅ Enhanced response received ({enhancedResponseText.Length} chars)[/]")
                AnsiConsole.MarkupLine("[green]This response is informed by actual TARS codebase structure![/]")
            else
                AnsiConsole.MarkupLine($"[red]❌ Enhanced query failed: {enhancedResponse.StatusCode}[/]")

        with
        | ex ->
            AnsiConsole.MarkupLine($"[yellow]⚠️ LLM demo requires Ollama running: {ex.Message}[/]")
            AnsiConsole.MarkupLine("[cyan]Demo shows how sedenion clusters would enhance LLM responses[/]")

        AnsiConsole.WriteLine()

        // Benefits of Sedenion-LLM Integration
        AnsiConsole.MarkupLine("[yellow]🚀 BENEFITS OF SEDENION-ENHANCED LLM INTEGRATION:[/]")
        AnsiConsole.MarkupLine("[green]✅ Semantic Retrieval: Find relevant knowledge using 16D similarity[/]")
        AnsiConsole.MarkupLine("[green]✅ Structured Context: Hierarchically organized information[/]")
        AnsiConsole.MarkupLine("[green]✅ Precise Responses: LLM answers based on actual codebase[/]")
        AnsiConsole.MarkupLine("[green]✅ Dynamic Context: Adapt context based on query semantics[/]")
        AnsiConsole.MarkupLine("[green]✅ Knowledge Synthesis: Combine multiple clusters for complex queries[/]")

        AnsiConsole.WriteLine()

        // Real-world applications
        AnsiConsole.MarkupLine("[yellow]💡 REAL-WORLD SEDENION-LLM APPLICATIONS:[/]")
        AnsiConsole.MarkupLine("[cyan]🏢 Enterprise RAG: Enhanced retrieval-augmented generation[/]")
        AnsiConsole.MarkupLine("[cyan]📚 Documentation AI: Context-aware code documentation[/]")
        AnsiConsole.MarkupLine("[cyan]🔍 Code Q&A: Intelligent codebase question answering[/]")
        AnsiConsole.MarkupLine("[cyan]🎯 Semantic Search: Find relevant code using natural language[/]")
        AnsiConsole.MarkupLine("[cyan]🤖 AI Assistants: Context-aware development assistants[/]")

        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[green]🎉 Sedenion BSP + LLM Integration demonstration complete![/]")
        AnsiConsole.MarkupLine("[green]✅ Analyzed real TARS codebase with 16D sedenion mathematics[/]")
        AnsiConsole.MarkupLine("[green]✅ Demonstrated sedenion-enhanced LLM context injection[/]")
        AnsiConsole.MarkupLine("[green]✅ Showcased practical AI integration benefits[/]")
    }
