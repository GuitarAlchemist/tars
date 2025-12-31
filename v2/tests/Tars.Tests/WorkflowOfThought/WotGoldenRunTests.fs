namespace Tars.Tests.WorkflowOfThought

open Xunit
open Tars.Core.WorkflowOfThought
open System.Collections.Generic

type FakeToolInvoker(responder: string -> Map<string,string> -> Result<obj,string>) =
    let calls = ResizeArray<string * Map<string,string>>()
    member _.Calls = Seq.toList calls

    interface IToolInvoker with
        member _.Invoke(toolName, args) = async {
            calls.Add((toolName, args))
            return responder toolName args
        }

module WotGoldenRunTests =

    [<Fact>]
    let ``executePlanV0 - golden run resolves vars, invokes tool, verifies, traces`` () =
        // Arrange
        let inputs = Map.ofList [ "target_msg", "Hello from Inputs" ]

        let steps : Step list =
          [ // Work: ToolCall echoes target_msg into res1
            { Id = "tool1"
              Inputs = []
              Outputs = ["res1"]
              Action =
                 StepAction.Work (
                  WorkOperation.ToolCall(
                    "mock_echo",
                    Map.ofList [ "msg", box "${target_msg}" ] 
                  )
                ) }

            // Verify: NonEmpty + Contains
            { Id = "verify1"
              Inputs = ["res1"]
              Outputs = ["verification"]
              Action =
                StepAction.Work (
                  WorkOperation.Verify [
                    WotCheck.NonEmpty "res1"
                    WotCheck.Contains ("res1", "Hello")
                  ]
                ) }
          ]

        let policy : WotExecutor.ExecutionPolicy =
          { AllowedTools = set ["mock_echo"]
            MaxToolCalls = 1 }

        let invoker =
          FakeToolInvoker(fun tool args ->
            match tool with
            | "mock_echo" ->
                let msg = args.["msg"]
                Ok (box msg)
            | _ -> Error "unknown tool")

        let reasoner = { new IReasoner with member _.Reason(_,_,_,_) = async { return Ok "stub" } }

        // Act
        let result =
          WotExecutor.executePlanV0 (invoker :> IToolInvoker) reasoner ReasonStepMode.Stub policy inputs steps
          |> Async.RunSynchronously

        // Assert
        match result with
        | Error (err, trace) ->
            failwith $"Expected Ok, got Error: {err}\nTrace count: {trace.Length}"
        | Ok (ctx, lastVerify, trace) ->
            // tool output stored
            Assert.True(ctx.Vars.ContainsKey "res1")
            Assert.Equal("Hello from Inputs", ctx.Vars.["res1"] :?> string)

            // verify passed
            match lastVerify with
            | None -> failwith "Expected a verify result"
            | Some vr -> Assert.True(vr.Passed, sprintf "Verification failed: %A" vr.Errors)

            // trace shape
            Assert.Equal(2, trace.Length)
            Assert.True(trace |> List.forall (fun e -> e.Status = StepStatus.Ok))
            
            // Allowlist check
            Assert.Equal(1, invoker.Calls.Length)

    [<Fact>]
    let ``executePlanV0 - reason step invokes IReasoner when mode is Llm`` () =
        let inputs : Map<string,string> = Map.empty
        let steps = 
             [{ Id = "reason1"
                Inputs = []
                Outputs = ["thought"]
                Action = StepAction.Reason (ReasonOperation.Plan "Think deeply") }]

        let policy : WotExecutor.ExecutionPolicy = { AllowedTools = Set.empty; MaxToolCalls = 0 }
        
        // Mock Tool Invoker (unused)
        let invoker = FakeToolInvoker(fun (t:string) (a:Map<string,string>) -> Error "should not be called")
        
        // Mock Reasoner
        let mutable reasonerCalled = false
        let reasoner = 
             { new IReasoner with 
                 member _.Reason(stepId, ctx, goal, _) = async { 
                     reasonerCalled <- true
                     Assert.True((goal = Some "Think deeply"), $"Goal mismatch. Got {goal}")
                     return Ok "Cogito ergo sum" } }

        let result =
          WotExecutor.executePlanV0 (invoker :> IToolInvoker) reasoner ReasonStepMode.Llm policy inputs steps
          |> Async.RunSynchronously

        match result with
        | Ok (ctx, _, trace) ->
             Assert.True(reasonerCalled, "Reasoner should have been called")
             Assert.Equal("Cogito ergo sum", ctx.Vars.["thought"] :?> string)
             Assert.Equal(1, trace.Length)
             let ev = trace.[0]
             Assert.Equal(StepStatus.Ok, ev.Status)
             Assert.Equal("reason", ev.Kind)
        | Error (e, _) -> failwith $"Execution failed: {e}"

    [<Fact>]
    let ``executePlanV0 - is deterministic (golden snapshot equality)`` () =
        // Arrange
        let inputs = Map.empty
        let steps = 
             [{ Id = "step1"; Inputs=[]; Outputs=["o1"]; Action=StepAction.Reason(ReasonOperation.Explain "Something") }]
        let policy : WotExecutor.ExecutionPolicy = { AllowedTools = Set.empty; MaxToolCalls = 0 }
        
        let invoker = FakeToolInvoker(fun _ _ -> Ok (box "dummy"))
        let reasoner = { new IReasoner with member _.Reason(_,_,_,_) = async { return Ok "Output" } }
        
        // Act - Run 1
        let res1 = 
            WotExecutor.executePlanV0 (invoker :> IToolInvoker) reasoner ReasonStepMode.Stub policy inputs steps
            |> Async.RunSynchronously

        // Act - Run 2 (simulate delay if needed, but not needed for logic determinism)
        let res2 = 
            WotExecutor.executePlanV0 (invoker :> IToolInvoker) reasoner ReasonStepMode.Stub policy inputs steps
            |> Async.RunSynchronously

        // Assert
        match res1, res2 with
        | Ok (_, _, trace1), Ok (_, _, trace2) ->
            let golden1 = trace1 |> List.map TraceEvent.toCanonical
            let golden2 = trace2 |> List.map TraceEvent.toCanonical
            
            // Raw traces differ by timestamp/duration
            Assert.NotEqual<TraceEvent list>(trace1, trace2) 
            
            // Golden traces must be identical
            // F# list equality is structural
            Assert.Equal<CanonicalTraceEvent list>(golden1, golden2)

        | _ -> failwith "One of the runs failed"
