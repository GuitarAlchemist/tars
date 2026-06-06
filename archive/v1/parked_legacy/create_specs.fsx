open System
open System.IO

let baseDir = Directory.GetCurrentDirectory()
let specs =
    [ "superintelligence-consensus", "Superintelligence Consensus Framework", "P1", "multi-agent consensus validation", "critic escalation safeguards"
      "superintelligence-critic", "Superintelligence Critic Training", "P1", "meta-critic feedback loop", "knowledge distillation"
      "superintelligence-recursive", "Superintelligence Recursive Goals", "P1", "auto-goal synthesis", "long-horizon planning" ]

let writeLines path lines =
    Directory.CreateDirectory(Path.GetDirectoryName(path)) |> ignore
    File.WriteAllLines(path, lines)

for (id, title, priority, task1, task2) in specs do
    let specDir = Path.Combine(baseDir, ".specify", "specs", id)
    Directory.CreateDirectory(specDir) |> ignore
    let specLines =
        [ sprintf "# Feature Specification: %s" title
          ""
          sprintf "**Feature Branch**: " id
          "**Created**: 2025-10-20"
          "**Status**: Draft"
          ""
          sprintf "### User Story 1 - %s (Priority: %s)" task1 priority
          ""
          "**Acceptance Scenarios**:"
          sprintf "1. Execute metascript-driven workflow for %s." title
          "2. Capture validation signals aligned with superintelligence objectives."
          ""
          "### Edge Cases"
          "- Harness validation requires additional tools."
          "- Consensus agents disagree on outcomes."
          ""
          ""
          ""
          "" ]
    writeLines (Path.Combine(specDir, "spec.md")) specLines

    let tasksLines =
        [ "## Phase 1"
          ""
          sprintf "- [ ] T001 [%s] Ensure %s" priority task1
          sprintf "- [ ] T002 [P2] Plan %s" task2 ]
    writeLines (Path.Combine(specDir, "tasks.md")) tasksLines
