open System
open System.IO
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Configuration
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open TarsEngine.FSharp.WindowsService.Core
open TarsEngine.FSharp.WindowsService.Tasks
open TarsEngine.FSharp.WindowsService.API

/// <summary>
/// TARS Windows Service Entry Point
/// Autonomous development platform running as a Windows service for unattended operation
/// </summary>
[<EntryPoint>]
let main args =
    try
        printfn "🚀 Starting TARS Windows Service..."
        printfn "   Autonomous Development Platform"
        printfn "   Version: 3.0.0"
        printfn "   Mode: Windows Service"
        printfn ""
        
        // Create host builder
        let hostBuilder = 
            Host.CreateDefaultBuilder(args)
                .UseWindowsService(fun options ->
                    options.ServiceName <- "TarsService")
                .ConfigureAppConfiguration(fun context config ->
                    let env = context.HostingEnvironment
                    
                    // Add configuration sources
                    config.AddJsonFile("appsettings.json", optional = true, reloadOnChange = true)
                        .AddJsonFile($"appsettings.{env.EnvironmentName}.json", optional = true, reloadOnChange = true)
                        .AddJsonFile("Configuration/service.config.json", optional = true, reloadOnChange = true)
                        .AddEnvironmentVariables()
                        .AddCommandLine(args) |> ignore
                )
                .ConfigureLogging(fun context logging ->
                    logging.ClearProviders()
                        .AddConsole()
                        .AddEventLog()
                        .AddDebug() |> ignore
                    
                    // Configure log levels
                    logging.SetMinimumLevel(LogLevel.Information) |> ignore
                )
                .ConfigureServices(fun context services ->
                    // Register core services
                    services.AddSingleton<SimpleServiceConfiguration>() |> ignore
                    services.AddHostedService<SimpleTarsService>() |> ignore

                    // Register documentation task manager
                    services.AddSingleton<DocumentationTaskManager>() |> ignore
                    services.AddHostedService<DocumentationTaskManager>() |> ignore

                    // Add API controllers
                    services.AddControllers() |> ignore
                    services.AddEndpointsApiExplorer() |> ignore
                    services.AddSwaggerGen() |> ignore

                    // Add logging
                    services.AddLogging() |> ignore
                )
                .ConfigureWebHostDefaults(fun webBuilder ->
                    webBuilder.UseUrls("http://localhost:5000")
                        .Configure(fun app ->
                            let env = app.ApplicationServices.GetRequiredService<IWebHostEnvironment>()

                            if env.IsDevelopment() then
                                app.UseSwagger() |> ignore
                                app.UseSwaggerUI() |> ignore

                            app.UseRouting() |> ignore
                            app.UseEndpoints(fun endpoints ->
                                endpoints.MapControllers() |> ignore
                            ) |> ignore
                        ) |> ignore
                )
        
        // Build and run host
        let host = hostBuilder.Build()
        
        printfn "✅ TARS Service configured successfully"
        printfn "🔄 Starting service host..."
        printfn ""
        
        // Run the service
        host.Run()
        
        printfn "🛑 TARS Service stopped"
        0 // Success exit code
        
    with
    | ex ->
        printfn "❌ TARS Service failed to start:"
        printfn $"   Error: {ex.Message}"
        printfn $"   Type: {ex.GetType().Name}"
        
        if ex.InnerException <> null then
            printfn $"   Inner: {ex.InnerException.Message}"
        
        printfn ""
        printfn "💡 Troubleshooting:"
        printfn "   • Check if running as Administrator"
        printfn "   • Verify configuration files exist"
        printfn "   • Check Windows Event Log for details"
        printfn "   • Ensure .NET 9.0 runtime is installed"
        
        1 // Error exit code
