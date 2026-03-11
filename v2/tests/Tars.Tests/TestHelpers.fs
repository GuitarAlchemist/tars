namespace Tars.Tests

open System.Threading.Tasks
open Xunit
open Tars.Core

/// Mock implementation of IOutputGuardAnalyzer for testing
type NoopOutputGuardAnalyzer() =
    interface IOutputGuardAnalyzer with
        member _.Analyze(_input: GuardInput) : Async<GuardResult option> = async { return None }

type EpistemicStatus =
    | Hypothesis
    | VerifiedFact
    | Fallacy
    | UniversalPrinciple

/// Delegate-based analyzer for tests
type DelegateOutputGuardAnalyzer(analyze: GuardInput -> Async<GuardResult option>) =
    interface IOutputGuardAnalyzer with
        member _.Analyze input = analyze input

module TestHelpers =
    open System
    open System.IO
    open System.Net.Http
    open System.Text.Json

    /// Checks if Tars.Tools.dll can be loaded (not blocked by WDAC/Smart App Control).
    /// Returns true if tools are available, false if blocked.
    let toolsAvailable =
        lazy (
            try
                // Force-reference a type from Tars.Tools to trigger assembly load
                let _ = typeof<Tars.Tools.ToolRegistry>
                true
            with
            | :? FileLoadException -> false
            | :? TypeInitializationException as ex when
                ex.InnerException <> null && (ex.InnerException :? FileLoadException) ->
                false)

    /// Guard for tests that depend on Tars.Tools.
    /// If WDAC blocks the DLL, the test should return early (pass vacuously).
    /// Usage: if not (TestHelpers.requireTools ()) then () else ...
    let requireTools () = toolsAvailable.Value

    /// Create a modern Belief record for testing
    let createTestBelief (id: BeliefId) (subject: string) (predicate: RelationType) (object: string) =
        { Id = id
          Subject = EntityId subject
          Predicate = predicate
          Object = EntityId object
          Provenance = Provenance.FromUser()
          Confidence = 1.0
          ValidFrom = DateTime.UtcNow
          InvalidAt = None
          Version = 1
          Tags = [] }

    /// Dynamic helper to find an available local model for integration tests
    let resolveTestModel () =
        task {
            // Priority: Env Vars -> Localhost probe -> Default fallback
            let envModel =
                [ "TARS_TEST_MODEL"; "OLLAMA_MODEL"; "TARS_LLM_MODEL" ]
                |> List.tryPick (fun name ->
                    let value = Environment.GetEnvironmentVariable(name)
                    if String.IsNullOrWhiteSpace value then None else Some value)

            match envModel with
            | Some model -> return model
            | None ->
                try
                    use http = new HttpClient(Timeout = TimeSpan.FromSeconds(2.0))
                    use! resp = http.GetAsync("http://localhost:11434/api/tags")

                    if not resp.IsSuccessStatusCode then
                        return "deepseek-r1:8b"
                    else
                        let! raw = resp.Content.ReadAsStringAsync()
                        use doc = JsonDocument.Parse(raw)

                        let models =
                            doc.RootElement.GetProperty("models").EnumerateArray()
                            |> Seq.choose (fun m ->
                                let mutable nameProp = Unchecked.defaultof<JsonElement>

                                if m.TryGetProperty("name", &nameProp) then
                                    Option.ofObj (nameProp.GetString())
                                else
                                    None)
                            |> Seq.toList

                        // Prefer fast/capable models if available
                        let preferred =
                            [ "llama3.2:3b"
                              "qwen2.5:3b"
                              "gemma3:1b"
                              "mistral:7b"
                              "llama3:8b"
                              "deepseek-r1:8b" ]

                        return
                            preferred
                            |> List.tryFind (fun name -> models |> List.contains name)
                            |> Option.orElse (models |> List.tryHead)
                            |> Option.defaultValue "deepseek-r1:8b"
                with _ ->
                    return "deepseek-r1:8b"
        }
