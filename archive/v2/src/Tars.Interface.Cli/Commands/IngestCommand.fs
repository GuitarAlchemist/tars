namespace Tars.Interface.Cli.Commands

open System
open Tars.Core
open Tars.Knowledge
open Tars.Llm

/// Ingest command - fetch and extract beliefs from Wikipedia
module IngestCommand =

    /// Run ingestion from a Wikipedia URL
    let runIngestion (url: string) =
        async {
            Logging.init LoggingConfig.Development
            let log = Logging.withCategory "Ingest"

            log.Info "╔═══════════════════════════════════════════════════════════╗"
            log.Info "║           TARS Knowledge Ingestion Pipeline               ║"
            log.Info "╚═══════════════════════════════════════════════════════════╝"
            log.Info ""

            // Load configuration
            let config = Tars.Interface.Cli.ConfigurationLoader.load()
            
            try
                // Validate URL
                if not (url.StartsWith("https://en.wikipedia.org/")) then
                    log.Error $"Only Wikipedia URLs are supported. Got: {url}"
                    return 1
                else

                // Create LLM service
                log.Info "🔧 Initializing LLM service..."
                let! llmService = 
                    match config.LLM.DefaultProvider with
                    | "ollama" -> 
                        OllamaClient.createService config.LLM.OllamaHost (config.LLM.OllamaModel |> Option.defaultValue "llama3.2")
                    | "openai" when config.Secrets.OpenAIKey.IsSome ->
                        async { return OpenAIClient.create config.Secrets.OpenAIKey.Value }
                    | _ ->
                        async {
                            log.Warn "No LLM provider configured. Using Ollama default."
                            return! OllamaClient.createService "http://localhost:11434" "llama3.2"
                        }

                // Create ledger
                let ledger = 
                    match config.Memory.PostgresConnectionString with
                    | None ->
                        log.Warn "No Postgres connection. Using in-memory ledger (data will be lost!)"
                        KnowledgeLedger.createInMemory()
                    | Some connString ->
                        log.Info $"Using Postgres ledger: {connString.Split([|';'|]).[0]}"
                        let storage = PostgresLedgerStorage(connString)
                        KnowledgeLedger(storage)

                do! ledger.Initialize() |> Async.AwaitTask

                // Create verifier
                let verifier = VerifierAgent(ledger)

                // Run ingestion pipeline
                log.Info $"🌐 Source: {url}"
                log.Info ""
                
                let! result = IngestionPipeline.ingest llmService ledger verifier url

                match result with
                | Ok ingestionResult ->
                    log.Info ""
                    log.Info "📋 Ingestion Summary:"
                    log.Info $"   Article: {ingestionResult.Stats.ArticleTitle}"
                    log.Info $"   Duration: {ingestionResult.Stats.DurationMs:F0}ms"
                    log.Info $"   Extracted: {ingestionResult.Stats.ProposalsExtracted} proposals"
                    log.Info $"   Accepted: {ingestionResult.Stats.ProposalsAccepted} beliefs"
                    log.Info $"   Denied: {ingestionResult.Stats.ProposalsDenied} proposals"
                    log.Info $"   Contradictions: {ingestionResult.Stats.ContradictionsFound}"
                    log.Info ""

                    if ingestionResult.AcceptedBeliefs.Length > 0 then
                        log.Info "✅ Sample accepted beliefs:"
                        for belief in ingestionResult.AcceptedBeliefs |> List.truncate 5 do
                            log.Info $"   • {belief.TripleString}"
                    
                    if ingestionResult.RejectedProposals.Length > 0 then
                        log.Info ""
                        log.Info "❌ Sample rejected proposals:"
                        for (proposal, reason) in ingestionResult.RejectedProposals |> List.truncate 3 do
                            log.Info $"   • ({proposal.Subject}, {proposal.Predicate}, {proposal.Object})"
                            log.Info $"     Reason: {reason}"

                    return 0

                | Error err ->
                    log.Error $"Ingestion failed: {err}"
                    return 1

            with ex ->
                log.Error("Ingestion command failed", ex)
                Console.ForegroundColor <- ConsoleColor.Red
                Console.WriteLine($"Error: {ex.Message}")
                Console.ResetColor()
                return 1
        }

    /// Parse command line args
    let parseArgs (args: string array) =
        if args.Length = 0 then
            printfn "TARS Knowledge Ingestion Command"
            printfn ""
            printfn "Usage: tars ingest <WIKIPEDIA_URL>"
            printfn ""
            printfn "Examples:"
            printfn "  tars ingest https://en.wikipedia.org/wiki/Quantum_computing"
            printfn "  tars ingest https://en.wikipedia.org/wiki/Artificial_intelligence"
            printfn ""
            None
        else
            Some args.[0]

    /// Main entry point
    let run (args: string array) =
        match parseArgs args with
        | None -> 1
        | Some url -> runIngestion url |> Async.RunSynchronously
