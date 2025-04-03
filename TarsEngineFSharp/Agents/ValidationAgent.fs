namespace TarsEngineFSharp

open System
open System.Collections.Generic
open System.Threading.Tasks
open System.Linq
open Microsoft.CodeAnalysis
open Microsoft.CodeAnalysis.CSharp

/// <summary>
/// Agent that validates code transformations
/// </summary>
type CSharpValidationAgent() =

    interface IValidationAgent with
        member _.ValidateAsync(originalCode: string, transformedCode: string) =
            task {
                let issues = new List<ValidationIssue>()

                // Check if the transformed code is valid C#
                try
                    let tree = CSharpSyntaxTree.ParseText(transformedCode)
                    let diagnostics = tree.GetDiagnostics()

                    if diagnostics.Any() then
                        for diagnostic in diagnostics do
                            issues.Add(
                                ValidationIssue(diagnostic.GetMessage(),
                                    (if diagnostic.Severity = DiagnosticSeverity.Error then
                                        ValidationSeverity.Error
                                     elif diagnostic.Severity = DiagnosticSeverity.Warning then
                                        ValidationSeverity.Warning
                                     else
                                        ValidationSeverity.Info),
                                    sprintf "Line %d" (diagnostic.Location.GetLineSpan().StartLinePosition.Line + 1)
                                )
                            )
                with ex ->
                    issues.Add(
                        ValidationIssue(
                            sprintf "Error parsing transformed code: %s" ex.Message,
                            ValidationSeverity.Error,
                            ""
                        )
                    )

                return ValidationResult(
                    not (issues |> Seq.exists (fun i -> i.Severity = ValidationSeverity.Error)),
                    issues
                )
            }
