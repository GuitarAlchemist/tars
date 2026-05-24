namespace Tars.Tests

open System
open Xunit
open Tars.Core

module ReasoningAuditTests =

    [<Fact>]
    let ``summary counts kept and pruned decisions`` () =
        let audit = ReasoningAudit.create()

        let build status =
            { BranchDecision.NodeId = Guid.NewGuid()
              Content = "Sample thought"
              NodeType = "Generate"
              Action = "test"
              Status = status
              Score = Some 0.5
              Confidence = Some 0.85
              Reasons = [ "reason" ]
              Risks = [ "risk" ]
              Timestamp = DateTime.UtcNow }

        ReasoningAudit.record audit (build "kept")
        ReasoningAudit.record audit (build "pruned")

        Assert.Equal(
            "decisions=2;kept=1;pruned=1;fallbacks=0;parseFailures=0;score=0.50-0.50;conf=0.85-0.85",
            ReasoningAudit.summary audit
        )
