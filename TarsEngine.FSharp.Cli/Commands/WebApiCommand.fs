namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.DataSources.Closures

/// Web API generation command for REST endpoints and GraphQL servers
type WebApiCommand(logger: ILogger<WebApiCommand>) =
    
    let webApiFactory = WebApiClosureFactory()
    
    interface ICommand with
        member _.Name = "webapi"
        member _.Description = "Generate REST endpoints and GraphQL servers using closure factory"
        member _.Usage = "tars webapi <subcommand> [options]"
        member _.Examples = [
            "tars webapi rest UserAPI"
            "tars webapi graphql ProductAPI"
            "tars webapi client MyClient http://localhost:5000/graphql"
            "tars webapi demo"
        ]
        member _.ValidateOptions(_) = true

        member _.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    match options.Arguments with
                    | [] ->
                        this.ShowWebApiHelp()
                        return CommandResult.success "Help displayed"
                    | "rest" :: name :: _ ->
                        let result = this.GenerateRestApi(name, Map.empty)
                        return if result = 0 then CommandResult.success "REST API generated" else CommandResult.failure "Failed to generate REST API"
                    | "graphql" :: name :: _ ->
                        let result = this.GenerateGraphQLServer(name, Map.empty)
                        return if result = 0 then CommandResult.success "GraphQL server generated" else CommandResult.failure "Failed to generate GraphQL server"
                    | "client" :: name :: schemaUrl :: _ ->
                        let result = this.GenerateGraphQLClient(name, schemaUrl)
                        return if result = 0 then CommandResult.success "GraphQL client generated" else CommandResult.failure "Failed to generate GraphQL client"
                    | "hybrid" :: name :: _ ->
                        let result = this.GenerateHybridApi(name, Map.empty)
                        return if result = 0 then CommandResult.success "Hybrid API generated" else CommandResult.failure "Failed to generate hybrid API"
                    | "demo" :: _ ->
                        let result = this.RunDemo()
                        return if result = 0 then CommandResult.success "Demo completed" else CommandResult.failure "Demo failed"
                    | "list" :: _ ->
                        let result = this.ListClosureTypes()
                        return if result = 0 then CommandResult.success "Closure types listed" else CommandResult.failure "Failed to list closure types"
                    | unknown :: _ ->
                        logger.LogWarning("Invalid webapi command: {Command}", String.Join(" ", unknown))
                        this.ShowWebApiHelp()
                        return CommandResult.failure $"Unknown subcommand: {unknown}"
                with
                | ex ->
                    logger.LogError(ex, "Error executing webapi command")
                    printfn $"❌ WebAPI command failed: {ex.Message}"
                    return CommandResult.failure ex.Message
            }
    
    /// Shows web API command help
    member _.ShowWebApiHelp() =
        printfn "TARS Web API Closure Factory"
        printfn "==========================="
        printfn ""
        printfn "Available Commands:"
        printfn "  rest <name>              - Generate REST API with Swagger"
        printfn "  graphql <name>           - Generate GraphQL server"
        printfn "  client <name> <url>      - Generate GraphQL client from schema URL"
        printfn "  hybrid <name>            - Generate hybrid REST + GraphQL API"
        printfn "  demo                     - Run comprehensive demo"
        printfn "  list                     - List available closure types"
        printfn ""
        printfn "Usage: tars webapi [command]"
        printfn ""
        printfn "Examples:"
        printfn "  tars webapi rest UserAPI"
        printfn "  tars webapi graphql ProductAPI"
        printfn "  tars webapi client MyClient http://localhost:5000/graphql"
        printfn "  tars webapi hybrid FullAPI"
        printfn "  tars webapi demo"
        printfn ""
        printfn "Features:"
        printfn "  • Real-time REST endpoint generation"
        printfn "  • GraphQL schema and resolver generation"
        printfn "  • Type-safe client generation"
        printfn "  • Swagger/OpenAPI documentation"
        printfn "  • JWT authentication support"
        printfn "  • Docker containerization"
        printfn "  • Production-ready code"
    
    /// Generates REST API
    member _.GenerateRestApi(name: string, config: Map<string, obj>) =
        printfn $"🔗 GENERATING REST API: {name}"
        printfn "=========================="
        
        try
            let outputDir = $"output/webapi/{name.ToLower()}-rest"
            let closure = webApiFactory.CreateRestEndpointClosure(name, config)
            
            let result = closure outputDir |> Async.RunSynchronously
            
            match result with
            | Ok info ->
                printfn ""
                printfn "✅ REST API generated successfully!"
                printfn $"📁 Output directory: {outputDir}"
                printfn $"🔗 Base URL: {info.BaseUrl}"
                printfn $"📊 Endpoints: {info.Endpoints}"
                printfn $"📖 Swagger: {info.SwaggerEnabled}"
                printfn ""
                printfn "Generated files:"
                for file in info.GeneratedFiles do
                    printfn $"  • {file}"
                printfn ""
                printfn "🚀 To run the API:"
                printfn $"  cd {outputDir}"
                printfn "  dotnet run"
                printfn ""
                0
            | Error error ->
                printfn $"❌ Failed to generate REST API: {error}"
                1
                
        with
        | ex ->
            logger.LogError(ex, "Error generating REST API")
            printfn $"❌ REST API generation failed: {ex.Message}"
            1
    
    /// Generates GraphQL server
    member _.GenerateGraphQLServer(name: string, config: Map<string, obj>) =
        printfn $"🚀 GENERATING GRAPHQL SERVER: {name}"
        printfn "=============================="
        
        try
            let outputDir = $"output/webapi/{name.ToLower()}-graphql"
            let closure = webApiFactory.CreateGraphQLServerClosure(name, config)
            
            let result = closure outputDir |> Async.RunSynchronously
            
            match result with
            | Ok info ->
                printfn ""
                printfn "✅ GraphQL Server generated successfully!"
                printfn $"📁 Output directory: {outputDir}"
                printfn $"🚀 GraphQL endpoint: {info.GraphQLEndpoint}"
                printfn $"🔍 Voyager: {info.VoyagerEndpoint}"
                printfn $"📊 Schema types: {info.SchemaTypes}"
                printfn $"❓ Queries: {info.Queries}"
                printfn $"✏️ Mutations: {info.Mutations}"
                printfn ""
                printfn "🚀 To run the server:"
                printfn $"  cd {outputDir}"
                printfn "  dotnet run"
                printfn ""
                0
            | Error error ->
                printfn $"❌ Failed to generate GraphQL server: {error}"
                1
                
        with
        | ex ->
            logger.LogError(ex, "Error generating GraphQL server")
            printfn $"❌ GraphQL server generation failed: {ex.Message}"
            1
    
    /// Generates GraphQL client
    member _.GenerateGraphQLClient(name: string, schemaUrl: string) =
        printfn $"📡 GENERATING GRAPHQL CLIENT: {name}"
        printfn "=============================="
        printfn $"📡 Schema URL: {schemaUrl}"
        
        try
            let outputDir = $"output/webapi/{name.ToLower()}-client"
            let closure = webApiFactory.CreateGraphQLClientClosure(name, schemaUrl)
            
            let result = closure outputDir |> Async.RunSynchronously
            
            match result with
            | Ok info ->
                printfn ""
                printfn "✅ GraphQL Client generated successfully!"
                printfn $"📁 Output directory: {info.OutputDirectory}"
                printfn $"📡 Schema URL: {info.SchemaUrl}"
                printfn $"📄 Client file: {info.ClientFile}"
                printfn ""
                printfn "🔧 To use the client:"
                printfn "  1. Add the generated file to your project"
                printfn "  2. Create HttpClient instance"
                printfn "  3. Initialize client with endpoint URL"
                printfn ""
                0
            | Error error ->
                printfn $"❌ Failed to generate GraphQL client: {error}"
                1
                
        with
        | ex ->
            logger.LogError(ex, "Error generating GraphQL client")
            printfn $"❌ GraphQL client generation failed: {ex.Message}"
            1
    
    /// Generates hybrid API
    member _.GenerateHybridApi(name: string, config: Map<string, obj>) =
        printfn $"🔥 GENERATING HYBRID API: {name}"
        printfn "=========================="
        
        try
            let outputDir = $"output/webapi/{name.ToLower()}-hybrid"
            let closure = webApiFactory.CreateHybridApiClosure(name, config)
            
            let result = closure outputDir |> Async.RunSynchronously
            
            match result with
            | Ok info ->
                printfn ""
                printfn "✅ Hybrid API generated successfully!"
                printfn $"📁 Output directory: {outputDir}"
                printfn $"🔗 Base URL: {info.BaseUrl}"
                printfn $"📊 REST endpoints: {info.RestEndpoints}"
                printfn $"🚀 GraphQL types: {info.GraphQLTypes}"
                printfn $"📖 Swagger: {info.SwaggerEnabled}"
                printfn $"🔍 GraphQL: {info.GraphQLEnabled}"
                printfn ""
                printfn "🚀 To run the hybrid API:"
                printfn $"  cd {outputDir}"
                printfn "  dotnet run"
                printfn ""
                printfn "🔗 Endpoints:"
                printfn $"  REST: {info.BaseUrl}/api"
                printfn $"  GraphQL: {info.BaseUrl}/graphql"
                printfn $"  Swagger: {info.BaseUrl}/swagger"
                printfn ""
                0
            | Error error ->
                printfn $"❌ Failed to generate hybrid API: {error}"
                1
                
        with
        | ex ->
            logger.LogError(ex, "Error generating hybrid API")
            printfn $"❌ Hybrid API generation failed: {ex.Message}"
            1
    
    /// Runs comprehensive demo
    member _.RunDemo() =
        printfn "🎬 RUNNING WEB API CLOSURE FACTORY DEMO"
        printfn "======================================="
        printfn ""
        
        try
            // Generate all API types
            printfn "🔄 Generating all API types..."
            printfn ""
            
            let demoConfig = Map.ofList [
                ("title", box "Demo User Management API")
                ("description", box "Comprehensive user management system")
                ("version", box "1.0.0")
            ]
            
            // REST API
            let restResult = this.GenerateRestApi("DemoUserAPI", demoConfig)
            if restResult <> 0 then
                printfn "❌ REST API demo failed"
                return 1
            
            printfn ""
            
            // GraphQL Server
            let graphqlResult = this.GenerateGraphQLServer("DemoUserGraphQL", demoConfig)
            if graphqlResult <> 0 then
                printfn "❌ GraphQL server demo failed"
                return 1
            
            printfn ""
            
            // GraphQL Client
            let clientResult = this.GenerateGraphQLClient("DemoClient", "http://localhost:5000/graphql")
            if clientResult <> 0 then
                printfn "❌ GraphQL client demo failed"
                return 1
            
            printfn ""
            
            // Hybrid API
            let hybridResult = this.GenerateHybridApi("DemoHybrid", demoConfig)
            if hybridResult <> 0 then
                printfn "❌ Hybrid API demo failed"
                return 1
            
            printfn ""
            printfn "🎉 DEMO COMPLETED SUCCESSFULLY!"
            printfn "==============================="
            printfn ""
            printfn "✅ Generated APIs:"
            printfn "  🔗 REST API: output/webapi/demouserapi-rest"
            printfn "  🚀 GraphQL Server: output/webapi/demousergraphql-graphql"
            printfn "  📡 GraphQL Client: output/webapi/democlient-client"
            printfn "  🔥 Hybrid API: output/webapi/demohybrid-hybrid"
            printfn ""
            printfn "🚀 All APIs are production-ready with:"
            printfn "  • Swagger/OpenAPI documentation"
            printfn "  • JWT authentication support"
            printfn "  • CORS configuration"
            printfn "  • Docker containerization"
            printfn "  • Health check endpoints"
            printfn ""
            
            0
            
        with
        | ex ->
            logger.LogError(ex, "Error running demo")
            printfn $"❌ Demo failed: {ex.Message}"
            1
    
    /// Lists available closure types
    member _.ListClosureTypes() =
        printfn "AVAILABLE WEB API CLOSURE TYPES"
        printfn "==============================="
        printfn ""
        
        let closureTypes = webApiFactory.GetAvailableClosureTypes()
        
        for closureType in closureTypes do
            let description = 
                match closureType with
                | "REST_ENDPOINT" -> "Generate REST API with Swagger documentation"
                | "GRAPHQL_SERVER" -> "Generate GraphQL server with schema and resolvers"
                | "GRAPHQL_CLIENT" -> "Generate type-safe GraphQL client"
                | "HYBRID_API" -> "Generate hybrid API with both REST and GraphQL"
                | _ -> "Unknown closure type"
            
            printfn $"  {closureType}: {description}"
        
        printfn ""
        printfn $"Total closure types: {closureTypes.Length}"
        
        0
