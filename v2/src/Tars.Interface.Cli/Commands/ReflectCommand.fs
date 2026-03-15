namespace Tars.Interface.Cli.Commands

open System
open Serilog
open Tars.Core
open Tars.Knowledge
open Spectre.Console
open Tars.Llm
open Tars.Interface.Cli

/// Reflection command - runs the Symbolic Reflection System
module ReflectCommand =

    /// Run legacy reflection on the knowledge ledger
    let runScan (confidenceThreshold: float option) =
        async {
            Logging.init LoggingConfig.Development
            let log = Logging.withCategory "Reflect"

            log.Info "╔═══════════════════════════════════════════════════════════╗"
            log.Info "║              TARS Symbolic Reflection (Phase 15)          ║"
            log.Info "╚═══════════════════════════════════════════════════════════╝"
            log.Info ""

            // Load configuration to get Postgres connection
            let config = Tars.Interface.Cli.ConfigurationLoader.load ()

            try
                // Create ledger (use Postgres if available, otherwise in-memory)
                let ledger =
                    match config.Memory.PostgresConnectionString with
                    | None ->
                        log.Warn "No Postgres connection configured. Using in-memory ledger."
                        KnowledgeLedger.createInMemory ()
                    | Some connString ->
                        let connParts = connString.Split([| ';' |])
                        log.Info $"Connecting to Postgres: {connParts.[0]}..."
                        let storage = PostgresLedgerStorage(connString)
                        KnowledgeLedger(storage)

                // Initialize ledger
                do! ledger.Initialize() |> Async.AwaitTask

                let stats = ledger.Stats()
                AnsiConsole.MarkupLine($"[blue]📊 Ledger Stats:[/]")
                AnsiConsole.MarkupLine($"   - Valid Beliefs: [green]{stats.ValidBeliefs}[/]")
                AnsiConsole.MarkupLine($"   - Contradictions: [red]{stats.Contradictions}[/]")

                // Create reflection agent
                let reflectionAgent = ReflectionAgent(ledger)

                // Run reflection
                AnsiConsole.MarkupLine "[yellow]🔍 Running symbolic reflection scan...[/]"
                do! reflectionAgent.ReflectAsync()

                // Get updated stats
                let statsAfter = ledger.Stats()
                let newContradictions = statsAfter.Contradictions - stats.Contradictions

                AnsiConsole.MarkupLine ""
                AnsiConsole.MarkupLine $"[green]✅ Reflection complete![/]"

                if newContradictions > 0 then
                    AnsiConsole.MarkupLine $"   - New contradictions found: [red]{newContradictions}[/]"
                else
                    AnsiConsole.MarkupLine $"   - No new contradictions found."

                // Optionally cleanup low-confidence beliefs
                match confidenceThreshold with
                | Some threshold ->
                    AnsiConsole.MarkupLine ""
                    AnsiConsole.MarkupLine $"[yellow]🧹 Cleaning up beliefs below {threshold:P0} confidence...[/]"
                    let! retracted = reflectionAgent.CleanupAsync(threshold)
                    AnsiConsole.MarkupLine $"   - Retracted [red]{retracted}[/] low-confidence beliefs"
                | None ->
                    AnsiConsole.MarkupLine ""

                    AnsiConsole.MarkupLine
                        "[dim]💡 Tip: Use --cleanup <threshold> to auto-retract low-confidence beliefs[/]"

                return 0

            with ex ->
                AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]")
                return 1
        }

    /// Run evidence chain trace demo
    let runTrace (beliefId: string) =
        async {
            try
                AnsiConsole.MarkupLine $"[blue]🔍 Tracing evidence chain for belief: {beliefId}[/]"

                // DEMO: Create a sample chain since we don't have persistence yet
                let evidence1 =
                    ReflectionEvidence.Create(
                        ReflectionEvidenceSource.ExternalSource(
                            "https://arxiv.org/abs/2305.10601",
                            DateTimeOffset.UtcNow.AddDays(-1.0)
                        ),
                        "Tree of Thoughts improves reasoning significantly",
                        0.95
                    )

                let evidence2 =
                    ReflectionEvidence.Create(
                        ReflectionEvidenceSource.TestResult("benchmark_tot", true),
                        "Benchmark passed with 85% accuracy",
                        0.90
                    )

                let belief =
                    { ReflectionBelief.Create("Tree of Thoughts is effective", 0.92) with
                        Id =
                            match Guid.TryParse(beliefId) with
                            | true, g -> g
                            | _ -> Guid.NewGuid()
                        Evidence = [ evidence1; evidence2 ] }

                let chain = EvidenceChains.buildChain belief Map.empty

                // Visualize
                let viz = EvidenceChains.visualize chain
                let escapedViz = viz.Replace("[", "[[").Replace("]", "]]")
                AnsiConsole.Write(Panel(escapedViz).Header("Evidence Chain"))

                // Check weakest link
                match EvidenceChains.findWeakestLink chain with
                | Some(e, conf) ->
                    AnsiConsole.Markup $"[yellow]⚠️ Weakest Link ({conf:P0}):[/] "
                    AnsiConsole.WriteLine(e.Content)
                | None -> AnsiConsole.MarkupLine "[green]✅ Strong chain - no weak links detected[/]"

                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[red]Error in Trace: {ex.Message}[/]")
                AnsiConsole.MarkupLine($"[dim]{ex.StackTrace}[/]")
                return 1
        }

    /// Run proof verification demo
    let runVerify (proofId: string) =
        async {
            try
                AnsiConsole.MarkupLine $"[blue]🛡️ Verifying proof: {proofId}[/]"

                // DEMO: Create sample proofs
                let validProof =
                    ReflectionProof.LogicalInference(
                        [ "All agents require memory"; "TARS is an agent" ],
                        "TARS requires memory",
                        ReflectionInferenceRule.Syllogism
                    )

                let invalidProof =
                    ReflectionProof.LogicalInference(
                        [], // Missing premises
                        "TARS is conscious",
                        ReflectionInferenceRule.ModusPonens
                    )

                let proof = if proofId = "valid" then validProof else invalidProof

                AnsiConsole.Markup("Structure: ")
                AnsiConsole.WriteLine(ProofSystem.describe proof)

                match ProofSystem.verifyProof proof with
                | FSharp.Core.Ok result ->
                    AnsiConsole.MarkupLine $"[green]✅ Proof Validated[/]"

                    AnsiConsole.MarkupLine
                        $"Strength: {result.Strength:P1} ({ProofSystem.categoryDescription result.StrengthCategory})"
                | FSharp.Core.Error errors ->
                    AnsiConsole.MarkupLine $"[red]❌ Proof Invalid[/]"

                    for err in errors do
                        AnsiConsole.Markup("  - ")
                        AnsiConsole.WriteLine(err.ToString())

                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[red]Error in Verify: {ex.Message}[/]")
                AnsiConsole.MarkupLine($"[dim]{ex.StackTrace}[/]")
                return 1
        }

    /// Run architectural reflection (The "Architect's Eye")
    let runArchitect () =
        async {
            try
                AnsiConsole.MarkupLine "[blue]🏗️  Initialising TARS Architect...[/]"
                
                // 1. Load Config & LLM
                let config = ConfigurationLoader.load ()
                let llmService = LlmFactory.create Log.Logger
                
                // 2. Load Ledger
                AnsiConsole.MarkupLine "[dim]Connecting to Knowledge Ledger...[/]"
                let ledger =
                    match config.Memory.PostgresConnectionString with
                    | None -> KnowledgeLedger.createInMemory ()
                    | Some conn -> 
                        let storage = PostgresLedgerStorage(conn)
                        KnowledgeLedger(storage)
                
                do! ledger.Initialize() |> Async.AwaitTask
                
                // 3. Create Agent
                let agent = ReflectionAgent(ledger, None, Some llmService)
                
                // 4. Run Reflection
                AnsiConsole.MarkupLine "[yellow]🧠 Reflecting on System Architecture...[/]"
                do! agent.ReflectOnArchitectureAsync()
                
                // 5. Check results (by querying what was just added)
                // We look for TARS_Architecture beliefs added in the last minute
                let recentArchitecture = 
                     ledger.Query()
                     |> Seq.filter (fun b -> b.Subject.ToString() = "TARS_Architecture" && b.ValidFrom > DateTime.UtcNow.AddMinutes(-1.0))
                     |> Seq.toList
                     
                if recentArchitecture.IsEmpty then
                    AnsiConsole.MarkupLine "[green]✅ No critical issues found. Architecture is stable.[/]"
                else
                    AnsiConsole.MarkupLine "[bold red]🚨 Architect Directives Generated:[/]"
                    for b in recentArchitecture do
                        AnsiConsole.MarkupLine $"   - [yellow]{b.Object}[/]"
                        
                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]")
                return 1
        }

    /// Parse command line args for scan command
    let parseScanArgs (args: string array) =
        let mutable threshold = None
        let mutable i = 0

        while i < args.Length do
            match args.[i].ToLowerInvariant() with
            | "--cleanup" when i + 1 < args.Length ->
                match Double.TryParse(args.[i + 1]) with
                | true, value ->
                    threshold <- Some value
                    i <- i + 2
                | _ -> i <- args.Length
            | _ -> i <- i + 1

        threshold

    /// Main entry point for reflect command
    let run (args: string array) =
        async {
            match args with
            | [||] ->
                // Default to help
                AnsiConsole.MarkupLine "Usage: tars reflect <command>"
                AnsiConsole.MarkupLine "Commands:"
                AnsiConsole.MarkupLine "  scan [--cleanup 0.X]   Run full reflection scan"
                AnsiConsole.MarkupLine "  trace <id>             Trace evidence chain for belief"
                AnsiConsole.MarkupLine "  verify <id>            Verify formal proof"
                AnsiConsole.MarkupLine "  architect              Run AI Architectural Reflection"
                return 0

            | args when args.[0] = "architect" -> return! runArchitect ()

            | args when args.[0] = "scan" || (args.[0].StartsWith("--") && args.[0] <> "--help") ->
                // Handle "scan" or legacy flags (defaulting to scan)
                let threshold = parseScanArgs args
                return! runScan threshold

            // Legacy support: "tars reflect" with no args runs scan
            | args when args.Length = 0 -> return! runScan None

            | [| "trace"; id |] -> return! runTrace id

            | [| "verify"; id |] -> return! runVerify id

            | _ -> return! runScan (parseScanArgs args)
        }
        |> Async.StartAsTask
