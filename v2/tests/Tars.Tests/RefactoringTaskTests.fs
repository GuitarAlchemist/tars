namespace Tars.Tests

open System
open System.IO
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Core.HybridBrain

/// Alias to avoid System.Action collision
type HybridAction = Tars.Core.HybridBrain.Action

module RefactoringTaskTests =

    /// Helper to create execution config with custom executor
    let createConfig (executor: HybridAction -> Task<Result<obj option, string>>) : ExecutionConfig =
        { VerboseLogging = true
          DryRun = false
          MaxRetries = 1
          DefaultTimeout = TimeSpan.FromSeconds(30.0)
          Drives =
            { Accuracy = 0.9
              Speed = 0.5
              Creativity = 0.5
              Safety = 0.9 }
          ActionExecutor = executor }

    [<Fact>]
    let ``Canonical Task - ExtractFunction action executes via HybridBrain`` () =
        task {
            // 1. Setup: Create a messy file
            let tempFile = Path.Combine(Path.GetTempPath(), $"messy_{Guid.NewGuid()}.fs")

            let initialContent =
                """module MessyCode

let doSomething x =
    let a = x + 1
    let b = a * 2
    let y = x + 1
    let z = y * 2
    b + z
"""

            File.WriteAllText(tempFile, initialContent)

            try
                // 2. Define the ActionExecutor
                let actionExecutor (action: HybridAction) : Task<Result<obj option, string>> =
                    task {
                        match action with
                        | HybridAction.ExtractFunction(name, startLine, endLine) ->
                            // Mock refactoring - replace file content
                            let refactoredContent =
                                """module MessyCode

let extracted x =
    let a = x + 1
    a * 2

let doSomething x =
    let b = extracted x
    let z = extracted x
    b + z
"""

                            File.WriteAllText(tempFile, refactoredContent)
                            return FSharp.Core.Ok(Some(box $"Extracted function '{name}'"))

                        | HybridAction.NoOp -> return FSharp.Core.Ok None

                        | _ -> return FSharp.Core.Error $"Action not supported: {action}"
                    }

                // 3. Create a Draft Plan with refactoring step
                let basePlan =
                    StateTransitions.createDraft "Refactor MessyCode.fs" "Extract duplicate logic into helper function"

                let refactorStep: Step =
                    { Id = 1
                      Name = "Extract Function"
                      Description = "Extract duplicate math logic"
                      Action = HybridAction.ExtractFunction("extracted", 6, 9)
                      Preconditions = []
                      Postconditions = []
                      EvidenceRequired = false
                      Timeout = None
                      RetryCount = 0 }

                let planWithSteps: Plan<Draft> =
                    { basePlan with
                        Steps = [ refactorStep ] }

                // 4. Run the Pipeline
                let config = createConfig actionExecutor
                let! result = HybridBrain.processAndExecute ValidationContext.Empty planWithSteps config

                // 5. Assertions
                match result with
                | FSharp.Core.Ok executionResult ->
                    Assert.Equal(RunOutcome.Success, executionResult.Outcome)
                    Assert.Equal(1, executionResult.StepsExecuted)

                    // Verify file content changed
                    let finalContent = File.ReadAllText(tempFile)
                    Assert.Contains("let extracted x", finalContent)
                    Assert.Contains("let z = extracted x", finalContent)

                | FSharp.Core.Error critique ->
                    let formatted = CritiqueFormatter.formatForLlm critique
                    Assert.Fail($"Pipeline failed:\n{formatted}")

            finally
                // Cleanup
                if File.Exists(tempFile) then
                    File.Delete(tempFile)
        }

    [<Fact>]
    let ``Draft plan with no steps executes with zero steps`` () =
        task {
            let emptyPlan = StateTransitions.createDraft "Empty" "No steps"
            let config = HybridBrain.defaultConfig

            let! result = HybridBrain.processAndExecute ValidationContext.Empty emptyPlan config

            match result with
            | FSharp.Core.Ok executionResult -> Assert.Equal(0, executionResult.StepsExecuted)
            | FSharp.Core.Error _ -> () // Validation failure is also acceptable
        }

    [<Fact>]
    let ``ActionExecutor error propagates as step failure`` () =
        task {
            let failingExecutor (action: HybridAction) : Task<Result<obj option, string>> =
                Task.FromResult(FSharp.Core.Error "Intentional failure")

            let basePlan = StateTransitions.createDraft "Test" "Test failure handling"

            let step: Step =
                { Id = 1
                  Name = "Failing Step"
                  Description = "This should fail"
                  Action = HybridAction.NoOp
                  Preconditions = []
                  Postconditions = []
                  EvidenceRequired = false
                  Timeout = None
                  RetryCount = 0 }

            let plan: Plan<Draft> = { basePlan with Steps = [ step ] }
            let config = createConfig failingExecutor

            let! result = HybridBrain.processAndExecute ValidationContext.Empty plan config

            match result with
            | FSharp.Core.Ok executionResult ->
                // Should be Partial or Failure due to step failure
                match executionResult.Outcome with
                | RunOutcome.Success -> Assert.Fail("Expected failure but got success")
                | RunOutcome.Partial _ -> () // Expected
                | RunOutcome.Failure _ -> () // Also acceptable
            | FSharp.Core.Error _ -> () // Critique is also acceptable
        }
