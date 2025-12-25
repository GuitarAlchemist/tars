/// TARS Database Migration Runner using DbUp
/// "Schema evolution is logged, not forgotten"
module Tars.Migrations.Program

open System
open DbUp

[<EntryPoint>]
let main args =
    printfn "🗄️  TARS Database Migrator"
    printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printfn ""

    // Get connection string from environment or use default
    let connectionString =
        match Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION") with
        | null
        | "" ->
            printfn "⚠️  No TARS_POSTGRES_CONNECTION environment variable found"
            printfn "Using default: Host=localhost;Database=tars;Username=postgres;Password=postgres"
            printfn ""
            "Host=localhost;Database=tars;Username=postgres;Password=postgres"
        | cs ->
            // Mask password for display
            let parts = cs.Split(';')

            let masked =
                parts
                |> Array.map (fun part ->
                    if part.ToLowerInvariant().Contains("password") then
                        "Password=***"
                    else
                        part)
                |> String.concat ";"

            printfn "Connection: %s" masked
            printfn ""
            cs

    // Create DbUp upgrader
    let upgrader =
        DeployChanges.To
            .PostgresqlDatabase(connectionString)
            .WithScriptsEmbeddedInAssembly(Reflection.Assembly.GetExecutingAssembly())
            .WithTransaction() // Run all scripts in a transaction
            .LogToConsole()
            .Build()

    // Check if any migrations needed
    let scriptsToExecute = upgrader.GetScriptsToExecute()

    if scriptsToExecute.Count = 0 then
        printfn "✅ Database is already up to date!"
        printfn "   No migrations to apply."
        0
    else
        printfn "📋 Found %d migration(s) to apply:" scriptsToExecute.Count

        for script in scriptsToExecute do
            printfn "   - %s" script.Name

        printfn ""

        // Perform upgrade
        printfn "🚀 Applying migrations..."
        let result = upgrader.PerformUpgrade()

        printfn ""

        if result.Successful then
            printfn "✅ Migration completed successfully!"
            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            0
        else
            printfn "❌ Migration failed!"
            printfn "Error: %s" result.Error.Message
            printfn ""
            printfn "Stack trace:"
            printfn "%s" (result.Error.ToString())
            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            1
