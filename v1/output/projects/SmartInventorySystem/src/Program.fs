namespace SmartInventorySystem

open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open System

module Program =
    
    [<EntryPoint>]
    let main args =
        let builder = WebApplication.CreateBuilder(args)
        
        // Add services
        builder.Services.AddControllers() |> ignore
        builder.Services.AddEndpointsApiExplorer() |> ignore
        builder.Services.AddSwaggerGen() |> ignore
        
        // Configure logging
        builder.Logging.AddConsole() |> ignore
        
        let app = builder.Build()
        
        // Configure pipeline
        if app.Environment.IsDevelopment() then
            app.UseSwagger() |> ignore
            app.UseSwaggerUI() |> ignore
        
        app.UseHttpsRedirection() |> ignore
        app.UseRouting() |> ignore
        app.MapControllers() |> ignore
        
        // Add health check endpoint
        app.MapGet("/health", fun () ->
            {| status = "healthy"; service = "SmartInventorySystem"; version = "1.0.0" |}) |> ignore

        // Add info endpoint
        app.MapGet("/", fun () ->
            {| service = "SmartInventorySystem"; version = "1.0.0"; status = "running" |}) |> ignore
        
        printfn "🚀 SmartInventorySystem API starting on http://localhost:5000"
        printfn "📖 Swagger UI: http://localhost:5000/swagger"
        printfn "❤️ Health check: http://localhost:5000/health"
        
        app.Run("http://0.0.0.0:5000")
        0
