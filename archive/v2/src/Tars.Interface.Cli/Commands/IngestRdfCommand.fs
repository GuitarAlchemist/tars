module IngestRdfCommand

open System
open System.IO
open Tars.Core
open Tars.Knowledge
open Tars.LinkedData
open Tars.Interface.Cli

let run (args: string array) =
    async {
        Logging.init LoggingConfig.Development
        let log = Logging.withCategory "CLI"
        
        if args.Length < 1 then
            log.Error("Please provide an RDF file path.")
            printfn "Usage: tars ingest-rdf <file.ttl>"
            return 1
        else
            let filePath = args.[0]
            if not (File.Exists filePath) then
                log.Error($"File find found: {filePath}")
                return 1
            else
                // Initialize infrastructure
                log.Info "🚀 Initializing TARS knowledge system..."
                
                // Load configuration to get Postgres connection
                let config = ConfigurationLoader.load()
                
                let ledgerResult = 
                    try
                        match config.Memory.PostgresConnectionString with
                        | None ->
                            log.Warn "No Postgres connection configured. Using in-memory ledger."
                            Microsoft.FSharp.Core.Result.Ok (KnowledgeLedger.createInMemory())
                        | Some connString ->
                            try
                                let storage = PostgresLedgerStorage(connString)
                                Microsoft.FSharp.Core.Result.Ok (KnowledgeLedger(storage))
                            with ex ->
                                Microsoft.FSharp.Core.Result.Error (sprintf "Invalid connection string or driver error: %s" ex.Message)
                    with ex ->
                         Microsoft.FSharp.Core.Result.Error (sprintf "Failed to create ledger configuration: %s" ex.Message)

                match ledgerResult with
                | Microsoft.FSharp.Core.Result.Error err ->
                    log.Error(err)
                    return 1
                | Microsoft.FSharp.Core.Result.Ok ledger ->
                    log.Info "⏳ Initializing ledger..."
                    let! initResult = 
                        async {
                            try
                                do! ledger.Initialize() |> Async.AwaitTask
                                return Microsoft.FSharp.Core.Result.Ok ()
                            with ex ->
                                return Microsoft.FSharp.Core.Result.Error ex.Message
                        }

                    match initResult with
                    | Microsoft.FSharp.Core.Result.Error err ->
                        log.Error($"Failed to connect to ledger storage: %s{err}")
                        log.Warn "Ensure PostgreSQL is running and connection string is correct."
                        return 1
                    | Microsoft.FSharp.Core.Result.Ok () ->
                        log.Info $"📂 Ingesting {filePath}..."
                        let sw = System.Diagnostics.Stopwatch.StartNew()
                        
                        let! res = RdfParser.importFile ledger filePath
                        match res with
                        | Microsoft.FSharp.Core.Result.Ok count ->
                            sw.Stop()
                            log.Info $"✅ Successfully imported {count} beliefs in {sw.ElapsedMilliseconds}ms"
                            return 0
                        | Microsoft.FSharp.Core.Result.Error err ->
                            sw.Stop()
                            log.Error($"❌ Ingestion failed: {err}")
                            return 1
    } |> Async.RunSynchronously