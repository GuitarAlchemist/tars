namespace AutonomousProject

open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting

module Program =
    [<EntryPoint>]
    let main args =
        let builder = WebApplication.CreateBuilder(args)
        
        builder.Services.AddControllers() |> ignore
        builder.Services.AddEndpointsApiExplorer() |> ignore
        builder.Services.AddSwaggerGen() |> ignore
        
        let app = builder.Build()
        
        if app.Environment.IsDevelopment() then
            app.UseSwagger() |> ignore
            app.UseSwaggerUI() |> ignore
        
        app.UseHttpsRedirection() |> ignore
        app.UseRouting() |> ignore
        app.MapControllers() |> ignore
        
        app.MapGet("/", fun () -> "Autonomous TARS Generated API - Working!") |> ignore
        
        printfn "Autonomous TARS API running on http://localhost:5000"
        app.Run("http://0.0.0.0:5000")
        0
