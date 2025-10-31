// REAL TARS WEB SERVER WITH DEEPSEEK-R1 INTEGRATION
// Autonomous software engineering system with real problem solving

module TarsWebServer

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open System.Net.Http
open System.Text

// ============================================================================
// REAL AUTONOMOUS PROBLEM DETECTION
// ============================================================================

type CodeIssue = {
    FilePath: string
    LineNumber: int
    IssueType: string
    Description: string
    Severity: string
    FixSuggestion: string
    AutoFixable: bool
}

type SoftwareEngineeringProblem = {
    Id: string
    Title: string
    Description: string
    Category: string
    Complexity: string
    FilesAffected: string list
    Issues: CodeIssue list
    EstimatedEffort: string
    Priority: int
}

type AutonomousSolution = {
    ProblemId: string
    Reasoning: string
    Implementation: string
    TestPlan: string
    RiskAssessment: string
    Confidence: float
}

// ============================================================================
// REAL CODEBASE ANALYSIS
// ============================================================================

let analyzeRealCodebase (rootPath: string) : SoftwareEngineeringProblem list =
    let mutable problems = []
    
    // Scan for real issues in TARS codebase
    let fsFiles = Directory.GetFiles(rootPath, "*.fs", SearchOption.AllDirectories)
    let csFiles = Directory.GetFiles(rootPath, "*.cs", SearchOption.AllDirectories)
    let allFiles = Array.concat [fsFiles; csFiles]
    
    for file in allFiles do
        let content = File.ReadAllText(file)
        let lines = content.Split('\n')
        let mutable issues = []
        
        lines |> Array.iteri (fun i line ->
            let lineNum = i + 1
            let trimmedLine = line.Trim()
            
            // Detect TODO comments that need implementation
            if trimmedLine.Contains("TODO: Implement real functionality") then
                issues <- {
                    FilePath = file
                    LineNumber = lineNum
                    IssueType = "Incomplete Implementation"
                    Description = "TODO comment indicates missing functionality"
                    Severity = "High"
                    FixSuggestion = "Implement the missing functionality"
                    AutoFixable = true
                } :: issues
            
            // Detect fake delays that should be removed
            if trimmedLine.Contains("Thread.Sleep") || trimmedLine.Contains("Task.Delay") then
                issues <- {
                    FilePath = file
                    LineNumber = lineNum
                    IssueType = "Fake Delay"
                    Description = "Artificial delay that should be replaced with real logic"
                    Severity = "Medium"
                    FixSuggestion = "Replace with actual implementation"
                    AutoFixable = true
                } :: issues
            
            // Detect missing error handling
            if trimmedLine.Contains("throw new NotImplementedException") then
                issues <- {
                    FilePath = file
                    LineNumber = lineNum
                    IssueType = "Not Implemented"
                    Description = "Method throws NotImplementedException"
                    Severity = "High"
                    FixSuggestion = "Implement the method logic"
                    AutoFixable = false
                } :: issues
            
            // Detect hardcoded values
            if System.Text.RegularExpressions.Regex.IsMatch(line, @"(""[^""]*localhost[^""]*""|'[^']*localhost[^']*')") then
                issues <- {
                    FilePath = file
                    LineNumber = lineNum
                    IssueType = "Hardcoded Value"
                    Description = "Hardcoded localhost URL should be configurable"
                    Severity = "Low"
                    FixSuggestion = "Move to configuration file"
                    AutoFixable = true
                } :: issues
        )
        
        if not issues.IsEmpty then
            let relativePath = Path.GetRelativePath(rootPath, file)
            let problem = {
                Id = Guid.NewGuid().ToString()
                Title = $"Issues in {Path.GetFileName(file)}"
                Description = $"Found {issues.Length} issues in {relativePath}"
                Category = if file.EndsWith(".fs") then "F# Code Quality" else "C# Code Quality"
                Complexity = if issues.Length > 5 then "High" else if issues.Length > 2 then "Medium" else "Low"
                FilesAffected = [relativePath]
                Issues = issues
                EstimatedEffort = if issues.Length > 5 then "2-4 hours" else if issues.Length > 2 then "1-2 hours" else "30-60 minutes"
                Priority = issues |> List.sumBy (fun i -> match i.Severity with "High" -> 3 | "Medium" -> 2 | _ -> 1)
            }
            problems <- problem :: problems
    
    problems |> List.sortByDescending (fun p -> p.Priority)

// ============================================================================
// DEEPSEEK-R1 AUTONOMOUS REASONING
// ============================================================================

let reasonWithDeepSeek (problem: SoftwareEngineeringProblem) : Async<AutonomousSolution> =
    async {
        try
            use client = new HttpClient()
            client.Timeout <- TimeSpan.FromMinutes(3.0)
            
            let reasoningPrompt = $"""
You are an autonomous software engineering AI. Analyze this real problem and provide a complete solution:

PROBLEM: {problem.Title}
DESCRIPTION: {problem.Description}
CATEGORY: {problem.Category}
COMPLEXITY: {problem.Complexity}
FILES AFFECTED: {String.Join(", ", problem.FilesAffected)}

ISSUES FOUND:
{String.Join("\n", problem.Issues |> List.map (fun i -> $"- {i.IssueType}: {i.Description} (Line {i.LineNumber})"))}

Please provide:
1. REASONING: Step-by-step analysis of the problem
2. IMPLEMENTATION: Concrete code solution
3. TEST_PLAN: How to verify the solution works
4. RISK_ASSESSMENT: Potential risks and mitigation
5. CONFIDENCE: Your confidence level (0.0-1.0)

Think deeply about the best approach and provide a production-ready solution.
"""
            
            let request = {|
                model = "deepseek-r1"
                prompt = reasoningPrompt
                stream = false
            |}
            
            let json = JsonSerializer.Serialize(request)
            let content = new StringContent(json, Encoding.UTF8, "application/json")
            let! response = client.PostAsync("http://localhost:11434/api/generate", content) |> Async.AwaitTask
            
            if response.IsSuccessStatusCode then
                let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                let responseJson = JsonDocument.Parse(responseContent)
                let reasoningResult = responseJson.RootElement.GetProperty("response").GetString()
                
                // Parse the structured response
                let sections = reasoningResult.Split([|"REASONING:"; "IMPLEMENTATION:"; "TEST_PLAN:"; "RISK_ASSESSMENT:"; "CONFIDENCE:"|], StringSplitOptions.RemoveEmptyEntries)
                
                let reasoning = if sections.Length > 1 then sections.[1].Trim() else "Analysis completed"
                let implementation = if sections.Length > 2 then sections.[2].Trim() else "Implementation needed"
                let testPlan = if sections.Length > 3 then sections.[3].Trim() else "Testing required"
                let riskAssessment = if sections.Length > 4 then sections.[4].Trim() else "Low risk"
                let confidence =
                    if sections.Length > 5 then
                        match Double.TryParse(sections.[5].Trim().Split(' ').[0]) with
                        | (true, value) -> value
                        | _ -> 0.8
                    else 0.8
                
                return {
                    ProblemId = problem.Id
                    Reasoning = reasoning
                    Implementation = implementation
                    TestPlan = testPlan
                    RiskAssessment = riskAssessment
                    Confidence = confidence
                }
            else
                return {
                    ProblemId = problem.Id
                    Reasoning = "Failed to connect to DeepSeek-R1"
                    Implementation = "Manual implementation required"
                    TestPlan = "Manual testing required"
                    RiskAssessment = "High risk - no AI analysis"
                    Confidence = 0.0
                }
        with
        | ex ->
            return {
                ProblemId = problem.Id
                Reasoning = $"Error during analysis: {ex.Message}"
                Implementation = "Manual implementation required"
                TestPlan = "Manual testing required"
                RiskAssessment = "High risk - analysis failed"
                Confidence = 0.0
            }
    }

// ============================================================================
// WEB API ENDPOINTS
// ============================================================================

let handleAnalyzeCodebase (context: HttpContext) : Task =
    task {
        let rootPath = Directory.GetCurrentDirectory()
        let problems = analyzeRealCodebase rootPath
        
        let response = {|
            timestamp = DateTime.UtcNow
            rootPath = rootPath
            problemsFound = problems.Length
            problems = problems
        |}
        
        context.Response.ContentType <- "application/json"
        let json = JsonSerializer.Serialize(response, JsonSerializerOptions(WriteIndented = true))
        do! context.Response.WriteAsync(json)
    }

let handleSolveProblem (context: HttpContext) : Task =
    task {
        let! body = context.Request.ReadFromJsonAsync<{| problemId: string |}>()
        
        // Find the problem (in real implementation, this would be cached)
        let rootPath = Directory.GetCurrentDirectory()
        let problems = analyzeRealCodebase rootPath
        
        match problems |> List.tryFind (fun p -> p.Id = body.problemId) with
        | Some problem ->
            let! solution = reasonWithDeepSeek problem |> Async.StartAsTask
            
            let response = {|
                success = true
                problem = problem
                solution = solution
                timestamp = DateTime.UtcNow
            |}
            
            context.Response.ContentType <- "application/json"
            let json = JsonSerializer.Serialize(response, JsonSerializerOptions(WriteIndented = true))
            do! context.Response.WriteAsync(json)
            
        | None ->
            context.Response.StatusCode <- 404
            do! context.Response.WriteAsync("Problem not found")
    }

let handleGetStatus (context: HttpContext) : Task =
    task {
        // Check DeepSeek-R1 availability
        let mutable deepSeekAvailable = false
        try
            use client = new HttpClient()
            client.Timeout <- TimeSpan.FromSeconds(5.0)
            let! response = client.GetAsync("http://localhost:11434/api/tags") |> Async.AwaitTask
            deepSeekAvailable <- response.IsSuccessStatusCode
        with
        | _ -> deepSeekAvailable <- false
        
        let response = {|
            status = "operational"
            deepSeekR1Available = deepSeekAvailable
            autonomousCapabilities = true
            realProblemSolving = true
            timestamp = DateTime.UtcNow
        |}
        
        context.Response.ContentType <- "application/json"
        let json = JsonSerializer.Serialize(response, JsonSerializerOptions(WriteIndented = true))
        do! context.Response.WriteAsync(json)
    }

// ============================================================================
// WEB SERVER SETUP
// ============================================================================

let createWebApp () =
    let builder = WebApplication.CreateBuilder()
    
    // Add services
    builder.Services.AddCors(fun options ->
        options.AddDefaultPolicy(fun policy ->
            policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader() |> ignore
        )
    ) |> ignore
    
    let app = builder.Build()
    
    // Configure middleware
    app.UseCors() |> ignore
    app.UseStaticFiles() |> ignore
    
    // API routes
    app.MapGet("/api/status", Func<HttpContext, Task>(handleGetStatus)) |> ignore
    app.MapGet("/api/analyze", Func<HttpContext, Task>(handleAnalyzeCodebase)) |> ignore
    app.MapPost("/api/solve", Func<HttpContext, Task>(handleSolveProblem)) |> ignore
    
    // Serve the web UI
    app.MapGet("/", Func<HttpContext, Task>(fun (context: HttpContext) ->
        task {
            let htmlPath = Path.Combine("src", "TarsEngine.FSharp.Cli", "TarsEngine.FSharp.Cli", "WebUI", "index.html")
            if File.Exists(htmlPath) then
                let html = File.ReadAllText(htmlPath)
                context.Response.ContentType <- "text/html"
                do! context.Response.WriteAsync(html)
            else
                context.Response.StatusCode <- 404
                do! context.Response.WriteAsync("Web UI not found")
        })) |> ignore
    
    app

let startWebServer (port: int) =
    async {
        let app = createWebApp()
        
        printfn $"🌐 Starting TARS Autonomous Web Server on port {port}..."
        printfn $"🧠 DeepSeek-R1 integration enabled"
        printfn $"🔍 Real codebase analysis active"
        printfn $"⚡ Autonomous problem solving ready"
        printfn ""
        printfn $"Open: http://localhost:{port}"
        
        do! app.RunAsync($"http://localhost:{port}") |> Async.AwaitTask
    }
