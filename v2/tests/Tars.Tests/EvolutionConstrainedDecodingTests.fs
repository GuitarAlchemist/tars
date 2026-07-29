namespace Tars.Tests

open System
open System.Collections.Generic
open System.Text.Json
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Evolution
open Tars.Llm

/// Item 4 slice A — the Tars.Evolution call sites must issue *Constrained* requests,
/// not bare JsonMode. These assert positively on the captured `ResponseFormat`:
/// grepping for the absence of `ResponseFormat.Json` is not equivalent, because
/// SymbolicReflector historically set `JsonMode = true` with `ResponseFormat = None`
/// and would satisfy an absence check while staying entirely unconstrained.
module EvolutionConstrainedDecodingTests =

    /// Records every request it is handed, then replies with canned text.
    type CapturingLlm(responseText: string) =
        let captured = ResizeArray<LlmRequest>()
        member _.Captured = List.ofSeq captured

        member _.Single =
            Assert.True(captured.Count >= 1, "call site issued no LLM request")
            captured.[0]

        interface ILlmService with
            member _.CompleteAsync(req) =
                captured.Add req
                task { return { Text = responseText; FinishReason = Some "stop"; Usage = None; Raw = None } }

            member _.CompleteStreamAsync(req, _handler) =
                captured.Add req
                task { return { Text = responseText; FinishReason = Some "stop"; Usage = None; Raw = None } }

            member _.EmbedAsync(_text) = Task.FromResult([| 0.0f |])

            member _.RouteAsync(_) =
                task { return { Backend = Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None } }

    type StubGraph() =
        interface IGraphService with
            member _.AddNodeAsync(_e) = Task.FromResult "node"
            member _.AddFactAsync(_f) = Task.FromResult(Guid.NewGuid())
            member _.AddEpisodeAsync(_ep) = Task.FromResult "episode"
            member _.QueryAsync(_q) = Task.FromResult List.empty<TarsFact>
            member _.PersistAsync() = Task.FromResult(())

    /// Assert the captured request is Constrained by exactly `expectedSchema`, and
    /// that JsonMode is still set — the degradation belt for backends that cannot
    /// enforce a schema (Ollama honours it; Anthropic/Gemini fall back to prompting).
    let private assertConstrainedBy (expectedSchema: string) (site: string) (req: LlmRequest) =
        match req.ResponseFormat with
        | Some(ResponseFormat.Constrained(Grammar.JsonSchema schema)) ->
            Assert.Equal(expectedSchema, schema)
        | other -> failwithf "%s sent %A — expected Constrained (JsonSchema ...)" site other

        Assert.True(req.JsonMode, $"{site} dropped JsonMode — the degradation belt must stay on")

    // ── Sites driven end-to-end through their public entry points ──────────────

    [<Fact>]
    let ``Evaluation site sends Constrained evaluationSchema`` () =
        task {
            let llm =
                CapturingLlm("""{"passed":true,"confidence":0.9,"summary":"ok","issues":[],"suggested_fixes":[]}""")

            let evaluator = SemanticEvaluation(llm :> ILlmService, minConfidence = 0.6) :> IEvaluationStrategy

            let t =
                { Id = Guid.NewGuid()
                  DifficultyLevel = 1
                  Goal = "Return the sum of two integers."
                  Constraints = []
                  ValidationCriteria = "Sum is correct"
                  Timeout = TimeSpan.FromSeconds 1.0
                  Score = 1.0 }

            let r =
                { TaskId = t.Id
                  TaskGoal = t.Goal
                  ExecutorId = AgentId(Guid.NewGuid())
                  Success = true
                  Output = "let add a b = a + b"
                  ExecutionTrace = []
                  Duration = TimeSpan.FromSeconds 1.0
                  Evaluation = None }

            let! _ = evaluator.Evaluate(t, r)
            assertConstrainedBy EvolutionSchemas.evaluationSchema "Evaluation" llm.Single
        }

    [<Fact>]
    let ``Reflection site sends Constrained reflectionSchema`` () =
        task {
            let llm =
                CapturingLlm("""{"type":"Success","score":0.9,"comment":"ok","suggestion":"none"}""")

            let agent = Reflection.LlmReflectionAgent(llm :> ILlmService) :> Reflection.IReflectionAgent

            let trace: Reflection.TraceItem list =
                [ { Step = "s1"; Input = "in"; Output = "out"; DurationMs = 1L } ]

            let! _ = agent.ReflectAsync("goal", "output", trace)
            assertConstrainedBy EvolutionSchemas.reflectionSchema "Reflection" llm.Single
        }

    [<Fact>]
    let ``SymbolicReflector site sends Constrained symbolicReflectionSchema`` () =
        task {
            let llm =
                CapturingLlm(
                    """{"trigger_type":"TaskCompleted","trigger_details":"d","observations":[],"summary":"s"}"""
                )

            let reflector =
                SymbolicReflector(llm :> ILlmService, StubGraph() :> IGraphService, AgentId(Guid.NewGuid()))

            let! _ = reflector.ReflectOnTrace(Guid.NewGuid(), [])
            assertConstrainedBy EvolutionSchemas.symbolicReflectionSchema "SymbolicReflector" llm.Single
        }

    // ── Schema well-formedness: these guard the two Engine sites, whose call
    //    sites sit in private functions behind the full evolution loop, and they
    //    guard every site against the strict-mode 400 that OpenAI returns for a
    //    schema missing additionalProperties:false or an incomplete `required`.

    let private allSchemas =
        [ "contradictionSchema", EvolutionSchemas.contradictionSchema
          "taskGenerationSchema", EvolutionSchemas.taskGenerationSchema
          "evaluationSchema", EvolutionSchemas.evaluationSchema
          "reflectionSchema", EvolutionSchemas.reflectionSchema
          "optimizerSchema", EvolutionSchemas.optimizerSchema
          "symbolicReflectionSchema", EvolutionSchemas.symbolicReflectionSchema ]

    [<Fact>]
    let ``every evolution schema is well-formed JSON`` () =
        for name, schema in allSchemas do
            let ex = Record.Exception(fun () -> JsonDocument.Parse(schema) |> ignore)
            Assert.True(isNull ex, $"{name} is not parseable JSON: {ex}")

    /// OpenAI's `json_schema` with strict:true — which OpenAiCompatibleClient always
    /// sets — rejects any object that omits additionalProperties:false or leaves a
    /// declared property out of `required`. A schema that violates this fails at
    /// request time against OpenAI, so assert it here instead.
    [<Fact>]
    let ``every evolution schema is strict-mode compatible`` () =
        let rec check (path: string) (el: JsonElement) =
            if el.ValueKind = JsonValueKind.Object then
                let mutable typeProp = JsonElement()
                let isObjectSchema =
                    el.TryGetProperty("type", &typeProp)
                    && typeProp.ValueKind = JsonValueKind.String
                    && typeProp.GetString() = "object"

                let mutable props = JsonElement()
                let hasProps = el.TryGetProperty("properties", &props)

                if isObjectSchema && hasProps then
                    let mutable addl = JsonElement()

                    Assert.True(
                        el.TryGetProperty("additionalProperties", &addl)
                        && addl.ValueKind = JsonValueKind.False,
                        $"{path}: object schema must set additionalProperties:false for strict mode"
                    )

                    let declared =
                        props.EnumerateObject() |> Seq.map (fun p -> p.Name) |> Set.ofSeq

                    let mutable req = JsonElement()
                    Assert.True(el.TryGetProperty("required", &req), $"{path}: object schema must list required")

                    let required =
                        req.EnumerateArray() |> Seq.map (fun e -> e.GetString()) |> Set.ofSeq

                    let missing = Set.difference declared required

                    Assert.True(
                        Set.isEmpty missing,
                        $"""{path}: strict mode requires every property in `required`; missing: {String.Join(", ", missing)}"""
                    )

                for p in el.EnumerateObject() do
                    check $"{path}.{p.Name}" p.Value
            elif el.ValueKind = JsonValueKind.Array then
                let mutable i = 0

                for item in el.EnumerateArray() do
                    check $"{path}[{i}]" item
                    i <- i + 1

        for name, schema in allSchemas do
            use doc = JsonDocument.Parse(schema)
            check name doc.RootElement
