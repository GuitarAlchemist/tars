namespace Tars.Interface.Cli.Commands

open System
open Tars.Core

/// Health check CLI commands
module HealthCommand =

    /// Run all health checks
    let runHealthChecks () =
        async {
            // Initialize logging
            Logging.init LoggingConfig.Development
            let log = Logging.withCategory "HealthCheck"

            log.Info "Starting TARS health check..."

            // Create registry and register checks
            let registry = HealthCheckRegistry()

            // Add built-in checks
            registry.Register(HealthChecks.alwaysHealthy "liveness")
            registry.Register(HealthChecks.memoryCheck 500.0)
            registry.Register(HealthChecks.uptimeCheck (TimeSpan.FromSeconds(1.0)) DateTime.UtcNow)

            // Add custom checks
            registry.RegisterSimple(
                "config",
                [ "readiness" ],
                fun () -> System.Threading.Tasks.Task.FromResult(Healthy)
            )

            registry.RegisterSimple(
                "llm-backend",
                [ "readiness"; "dependency" ],
                fun () -> System.Threading.Tasks.Task.FromResult(Degraded "No LLM backend configured - demo mode")
            )

            // Run checks
            let! report = registry.RunAllAsync() |> Async.AwaitTask

            // Display results
            Console.WriteLine()
            Console.WriteLine(HealthChecks.formatReport report)
            Console.WriteLine()

            log.Info $"Health check complete: {report.Checks.Length} checks run"

            // Also show JSON format
            Console.ForegroundColor <- ConsoleColor.DarkGray
            Console.WriteLine("JSON Output:")
            Console.WriteLine(HealthChecks.formatJson report)
            Console.ResetColor()

            return
                match report.Status with
                | Healthy -> 0
                | Degraded _ -> 1
                | Unhealthy _ -> 2
        }
