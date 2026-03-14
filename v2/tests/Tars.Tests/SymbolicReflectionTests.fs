namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Connectors

module SymbolicReflectionTests =

    let mkStep id content = 
        { RunId = Guid.Empty
          StepId = id
          NodeType = "Goal"
          Content = content
          Timestamp = DateTime.UtcNow }

    [<Fact>]
    let ``AnalyzeSteps detects tool usage patterns`` () =
        let steps = [
            mkStep "1" "Thought: Call tool\nToolCall(code_search, 'args')"
            mkStep "2" "Thought: Call tool again\nToolCall(code_search, 'args')"
            mkStep "3" "Thought: Call tool\nToolCall(code_search, 'args')"
        ]
        
        let report = SymbolicReflector.AnalyzeSteps(Guid.NewGuid(), steps)
        
        Assert.Contains(report.Observations, fun obs -> 
            match obs with
            | PatternObserved(desc, count, _) -> desc.Contains("FrequentToolUsage: code_search") && count = 3
            | _ -> false)

    [<Fact>]
    let ``AnalyzeSteps detects error keywords`` () =
        let steps = [
            mkStep "1" "Everything is fine"
            mkStep "2" "Oh no, an Exception occurred"
            mkStep "3" "Another Fail"
        ]
        
        let report = SymbolicReflector.AnalyzeSteps(Guid.NewGuid(), steps)
        
        Assert.Contains(report.Observations, fun obs -> 
            match obs with
            | AnomalyObserved(desc, severity) -> desc.Contains("ErrorKeywordsCaptured: 2") && severity = AnomalySeverity.Low
            | _ -> false)

    [<Fact>]
    let ``AnalyzeSteps detects repeated steps`` () =
         let steps = [
            mkStep "Rep" "Content"
            mkStep "Rep" "Content Retry"
         ]

         let report = SymbolicReflector.AnalyzeSteps(Guid.NewGuid(), steps)
         
         Assert.Contains(report.Observations, fun obs -> 
            match obs with
            | PatternObserved(desc, count, _) -> desc.Contains("StepRepetition:Rep") && count = 2
            | _ -> false)
