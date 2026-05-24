namespace TarsEngineFSharp

open System
open System.Collections.Generic
open System.Threading.Tasks
open System.Linq
open Microsoft.CodeAnalysis
open Microsoft.CodeAnalysis.CSharp
open Microsoft.CodeAnalysis.CSharp.Syntax
open TarsEngineFSharp

/// <summary>
/// Agent that analyzes code for potential improvements
/// </summary>
type CSharpAnalysisAgent() =

    interface IAnalysisAgent with
        member _.AnalyzeAsync(code: string) =
            task {
                // Parse the code
                let tree = CSharpSyntaxTree.ParseText(code)
                let root = tree.GetRoot() :?> CompilationUnitSyntax

                // Analyze the code using RetroactionAnalysis
                let retroResult = RetroactionAnalysis.analyzeCode code

                // Convert to the expected AnalysisResult type
                let issues = retroResult.Issues |> Seq.map (fun issue ->
                    CodeIssue(issue.Description, issue.Description, issue.Location))
                let fixes = retroResult.Fixes |> Seq.map (fun fix ->
                    CodeFix(fix.Original, fix.Replacement, fix.Description))

                return AnalysisResult(issues, fixes)
            }
