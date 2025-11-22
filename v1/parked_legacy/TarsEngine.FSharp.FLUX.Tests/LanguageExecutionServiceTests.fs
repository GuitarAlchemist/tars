namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.Diagnostics
open System.ComponentModel
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.Execution.FluxRuntime

/// Focused tests for the language execution service
module LanguageExecutionServiceTests =

    let private isExecutableAvailable (candidates: string list) =
        candidates
        |> List.exists (fun executable ->
            try
                let psi = ProcessStartInfo()
                psi.FileName <- executable
                psi.Arguments <- "--version"
                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.CreateNoWindow <- true

                use proc = new Process()
                proc.StartInfo <- psi
                proc.Start() |> ignore

                if not (proc.WaitForExit(5000)) then
                    try proc.Kill(true) with | _ -> ()
                true
            with
            | :? Win32Exception as ex when ex.NativeErrorCode = 2 || ex.NativeErrorCode = 3 -> false
            | :? Win32Exception -> false
            | _ -> false)

    let private isWindows = OperatingSystem.IsWindows()
    let private pythonCandidates = if isWindows then [ "python.exe"; "python"; "py.exe"; "py" ] else [ "python3"; "python" ]
    let private nodeCandidates = if isWindows then [ "node.exe"; "node" ] else [ "node"; "nodejs" ]
    let private dotnetCandidates = [ "dotnet" ]
    let private juliaCandidates = if isWindows then [ "julia.exe"; "julia" ] else [ "julia" ]
    let private wolframCandidates = if isWindows then [ "wolframscript.exe"; "wolframscript" ] else [ "wolframscript" ]

    let private pythonAvailable = lazy (isExecutableAvailable pythonCandidates)
    let private nodeAvailable = lazy (isExecutableAvailable nodeCandidates)
    let private dotnetAvailable = lazy (isExecutableAvailable dotnetCandidates)
    let private juliaAvailable = lazy (isExecutableAvailable juliaCandidates)
    let private wolframAvailable = lazy (isExecutableAvailable wolframCandidates)

    [<Fact>]
    let ``Python execution injects variables and produces output`` () =
        Assert.True(pythonAvailable.Value, "Python interpreter not available on PATH.")

        let service = LanguageExecutionService()
        let variables =
            Map.ofList [
                ("name", StringValue "Codex")
                ("value", NumberValue 41.0)
            ]
        let code = "print(f\"Greetings, {name}!\")\nprint(value + 1)"

        let result = service.ExecutePython(code, variables).GetAwaiter().GetResult()

        result.Success |> should equal true
        result.ExitCode |> should equal 0
        result.Output.Contains("Greetings, Codex!") |> should equal true
        result.Output.Contains("42") |> should equal true

    [<Fact>]
    let ``JavaScript execution injects globals via Node`` () =
        Assert.True(nodeAvailable.Value, "Node.js interpreter not available on PATH.")

        let service = LanguageExecutionService()
        let variables =
            Map.ofList [
                ("name", StringValue "Codex")
                ("value", NumberValue 41.0)
            ]
        let code = "console.log(`Greetings, ${name}!`);\nconsole.log(value + 1);"

        let result = service.ExecuteJavaScript(code, variables).GetAwaiter().GetResult()

        result.Success |> should equal true
        result.ExitCode |> should equal 0
        result.Output.Contains("Greetings, Codex!") |> should equal true
        result.Output.Contains("42") |> should equal true

    [<Fact>]
    let ``CSharp execution wires bootstrap variables`` () =
        Assert.True(dotnetAvailable.Value, ".NET SDK not available on PATH.")

        let service = LanguageExecutionService()
        let variables =
            Map.ofList [
                ("name", StringValue "Codex")
                ("number", NumberValue 41.0)
            ]
        let code = """
Console.WriteLine($"Greetings, {name}!");
Console.WriteLine(number + 1);
"""
        let result = service.ExecuteCSharp(code, variables).GetAwaiter().GetResult()

        result.Success |> should equal true
        result.ExitCode |> should equal 0
        result.Output.Contains("Greetings, Codex!") |> should equal true
        result.Output.Contains("42") |> should equal true

    [<Fact>]
    let ``CSharp execution surfaces compiler diagnostics on failure`` () =
        Assert.True(dotnetAvailable.Value, ".NET SDK not available on PATH.")

        let service = LanguageExecutionService()
        let code = "Console.WriteLine(\"Missing semicolon\")"

        let result = service.ExecuteCSharp(code, Map.empty).GetAwaiter().GetResult()

        result.Success |> should equal false
        result.ExitCode |> should not' (equal 0)
        let error = result.Error |> Option.defaultValue ""
        error.Equals("", StringComparison.Ordinal) |> should equal false
        error.Contains("error", StringComparison.OrdinalIgnoreCase) |> should equal true

    [<Fact>]
    let ``Julia execution reports interpreter availability`` () =
        let service = LanguageExecutionService()
        let code = "println(\"FLUX Julia integration test\")"

        let result = service.ExecuteJulia(code, Map.empty).GetAwaiter().GetResult()

        if juliaAvailable.Value then
            result.Success |> should equal true
            result.ExitCode |> should equal 0
            result.Output.Contains("FLUX Julia integration test") |> should equal true
        else
            result.Success |> should equal false
            result.ExitCode |> should equal -1
            let error = result.Error |> Option.defaultValue ""
            error.Contains("Interpreter not available") |> should equal true

    [<Fact(Skip = "WolframScript not available in this environment")>]
    let ``Wolfram execution reports interpreter availability`` () =
        let service = LanguageExecutionService()
        let code = "Print[\"FLUX Wolfram integration test\"];"

        let result = service.ExecuteWolfram(code, Map.empty).GetAwaiter().GetResult()

        if wolframAvailable.Value then
            result.Success |> should equal true
            result.ExitCode |> should equal 0
            result.Output.Contains("FLUX Wolfram integration test") |> should equal true
        else
            result.Success |> should equal false
            result.ExitCode |> should equal -1
            let error = result.Error |> Option.defaultValue ""
            error.Contains("Interpreter not available") |> should equal true
