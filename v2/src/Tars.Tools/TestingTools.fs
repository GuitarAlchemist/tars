namespace Tars.Tools.Standard

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Tars.Tools

module TestingTools =

    /// Helper to run dotnet commands
    let private runDotnet (workDir: string) (args: string) =
        task {
            try
                let psi = ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- args
                psi.WorkingDirectory <- workDir
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                use proc = Process.Start(psi)
                let! stdout = proc.StandardOutput.ReadToEndAsync()
                let! stderr = proc.StandardError.ReadToEndAsync()

                // Wait with timeout
                let completed = proc.WaitForExit(30000)

                if not completed then
                    proc.Kill()
                    return Error "Test execution timed out (30s)"
                else if proc.ExitCode = 0 then
                    return Ok(stdout.Trim())
                else
                    return Error($"Exit {proc.ExitCode}: {stderr.Trim()}")
            with ex ->
                return Error(ex.Message)
        }

    [<TarsToolAttribute("run_tests",
                        "Runs .NET tests in the project. Input JSON: { \"project\": \"path/to/test.fsproj\", \"filter\": \"optional filter\" }")>]
    let runTests (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let mutable projProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let project =
                    if root.TryGetProperty("project", &projProp) then
                        projProp.GetString()
                    else
                        "tests/Tars.Tests/Tars.Tests.fsproj"

                let mutable filterProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let filter =
                    if root.TryGetProperty("filter", &filterProp) then
                        $" --filter \"{filterProp.GetString()}\""
                    else
                        ""

                printfn $"🧪 RUNNING TESTS: {project}{filter}"
                let workDir = Directory.GetCurrentDirectory()
                let! result = runDotnet workDir $"test {project}{filter} --no-build --verbosity quiet"

                match result with
                | Ok output ->
                    printfn "✅ Tests passed!"
                    return $"Test Results:\n{output}"
                | Error e ->
                    printfn "❌ Tests failed!"
                    return $"Test Failure:\n{e}"
            with ex ->
                return $"run_tests error: {ex.Message}"
        }

    [<TarsToolAttribute("generate_test",
                        "Generates a unit test template. Input JSON: { \"function\": \"functionName\", \"module\": \"ModuleName\", \"file\": \"path/to/file.fs\" }")>]
    let generateTest (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let funcName = root.GetProperty("function").GetString()

                let moduleName =
                    let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>

                    if root.TryGetProperty("module", &prop) then
                        prop.GetString()
                    else
                        "Module"

                printfn $"📝 GENERATING TEST for {moduleName}.{funcName}"

                let testTemplate =
                    $"""
[<Fact>]
let ``{funcName} should work correctly`` () =
    // Arrange
    let input = () // TODO: Set up test input
    
    // Act
    let result = {moduleName}.{funcName} input
    
    // Assert
    Assert.NotNull(result)
    // TODO: Add specific assertions

[<Fact>]
let ``{funcName} handles edge cases`` () =
    // Test edge cases like empty input, null, etc.
    Assert.True(true) // TODO: Implement
"""

                return
                    $"Generated test template for {funcName}:\n```fsharp{testTemplate}```\n\nUse write_code to save this to a test file."
            with ex ->
                return $"generate_test error: {ex.Message}"
        }

    [<TarsToolAttribute("analyze_code",
                        "Analyzes code for common issues and metrics. Input JSON: { \"path\": \"file.fs\" }")>]
    let analyzeCode (args: string) =
        task {
            try
                let path =
                    try
                        let doc = System.Text.Json.JsonDocument.Parse(args)
                        doc.RootElement.GetProperty("path").GetString()
                    with _ ->
                        args

                let fullPath = Path.GetFullPath(path)

                if not (File.Exists fullPath) then
                    return $"File not found: {fullPath}"
                else
                    let content = File.ReadAllText(fullPath)
                    let lines = content.Split('\n')

                    // Basic metrics
                    let totalLines = lines.Length

                    let codeLines =
                        lines
                        |> Array.filter (fun l -> not (String.IsNullOrWhiteSpace(l)) && not (l.Trim().StartsWith("//")))
                        |> Array.length

                    let commentLines =
                        lines |> Array.filter (fun l -> l.Trim().StartsWith("//")) |> Array.length

                    let functionCount =
                        lines
                        |> Array.filter (fun l -> l.Contains("let ") && l.Contains("="))
                        |> Array.length

                    // Find potential issues
                    let longLines = lines |> Array.filter (fun l -> l.Length > 120) |> Array.length

                    let todoCount =
                        lines |> Array.filter (fun l -> l.ToUpper().Contains("TODO")) |> Array.length

                    let mutableCount =
                        lines |> Array.filter (fun l -> l.Contains("mutable ")) |> Array.length

                    printfn $"📊 ANALYZING: {path}"

                    return
                        $"""Code Analysis for {Path.GetFileName(path)}:

📏 Metrics:
  - Total lines: {totalLines}
  - Code lines: {codeLines}
  - Comment lines: {commentLines}
  - Functions/bindings: {functionCount}

⚠️ Potential Issues:
  - Long lines (>120 char): {longLines}
  - TODO comments: {todoCount}
  - Mutable bindings: {mutableCount}

💡 Suggestions:
{if longLines > 5 then
     "  - Consider breaking long lines for readability"
 else
     ""}
{if todoCount > 0 then
     $"  - {todoCount} TODO items need attention"
 else
     ""}
{if mutableCount > 3 then
     "  - Consider reducing mutable state for better functional design"
 else
     ""}
"""
            with ex ->
                return $"analyze_code error: {ex.Message}"
        }

    [<TarsToolAttribute("build_project",
                        "Builds the .NET project. Input JSON: { \"project\": \"path/to/project.fsproj\" } or empty for solution")>]
    let buildProject (args: string) =
        task {
            try
                let project =
                    if String.IsNullOrWhiteSpace(args) || args = "{}" then
                        "Tars.sln"
                    else
                        try
                            let doc = System.Text.Json.JsonDocument.Parse(args)
                            let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>

                            if doc.RootElement.TryGetProperty("project", &prop) then
                                prop.GetString()
                            else
                                "Tars.sln"
                        with _ ->
                            args

                printfn $"🔨 BUILDING: {project}"
                let workDir = Directory.GetCurrentDirectory()
                let! result = runDotnet workDir $"build {project} --verbosity quiet"

                match result with
                | Ok output ->
                    printfn "✅ Build succeeded!"
                    return $"Build succeeded:\n{output}"
                | Error e ->
                    printfn "❌ Build failed!"
                    return $"Build failed:\n{e}"
            with ex ->
                return $"build_project error: {ex.Message}"
        }
