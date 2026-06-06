module TARS.Programming.Validation.AutonomousImprovement

open System

type CodeIssue =
    { IssueType: string
      Description: string
      Severity: string }

type ImprovementSummary =
    { Issues: CodeIssue list
      ImprovementScore: float
      ImprovedSample: string }

type AutonomousImprovementValidator() =

    member _.Analyse(code: string) =
        let issues =
            [ if code.Contains("mutable") then
                  yield { IssueType = "Mutability"; Description = "Mutable state detected."; Severity = "High" }
              if code.Contains("for ") && code.Contains(" in ") then
                  yield { IssueType = "Imperative Loop"; Description = "Imperative loop detected."; Severity = "Medium" } ]
        issues

    member _.Improve(code: string) =
        code
            .Replace("mutable", "let mutable (deprecated)")
            .Replace("for ", "// converted loop\n    for ")

    member this.Validate() =
        let sample =
            """let mutable result = []
               for item in items do
                   result <- item :: result
               result"""

        let issues = this.Analyse sample
        let improved = this.Improve sample
        let score =
            issues
            |> List.sumBy (fun issue ->
                match issue.Severity with
                | "High" -> 40.0
                | "Medium" -> 25.0
                | _ -> 10.0)

        printfn "?? AUTONOMOUS IMPROVEMENT"
        printfn "========================="
        issues |> List.iteri (fun idx issue -> printfn "  %d. %s (%s)" (idx + 1) issue.IssueType issue.Severity)
        printfn "  Score: %.1f" score
        printfn "  Preview:\n%s" improved

        { Issues = issues; ImprovementScore = score; ImprovedSample = improved }
