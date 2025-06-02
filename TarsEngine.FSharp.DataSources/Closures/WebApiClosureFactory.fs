namespace TarsEngine.FSharp.DataSources.Closures

open System
open System.IO
open System.Collections.Generic
open TarsEngine.FSharp.DataSources.Core
open TarsEngine.FSharp.DataSources.Generators

/// Web API closure factory for generating REST endpoints and GraphQL servers
type WebApiClosureFactory() =
    
    let restGenerator = RestEndpointGenerator()
    let graphqlGenerator = GraphQLGenerator()
    
    /// Creates a REST endpoint closure from metascript configuration
    member _.CreateRestEndpointClosure(name: string, config: Map<string, obj>) =
        let webApiConfig = this.ParseWebApiConfig(name, config)
        
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🔧 Generating REST API: {webApiConfig.Name}"
                    
                    // Generate the web API project
                    let generatedApi = restGenerator.GenerateWebApiProject(webApiConfig, outputDir)
                    
                    printfn $"✅ REST API generated successfully at: {outputDir}"
                    printfn $"📖 Swagger UI: {webApiConfig.BaseUrl}/swagger"
                    printfn $"❤️ Health check: {webApiConfig.BaseUrl}/health"
                    
                    return Ok {|
                        Type = "REST_API"
                        Name = webApiConfig.Name
                        OutputDirectory = outputDir
                        BaseUrl = webApiConfig.BaseUrl
                        Endpoints = webApiConfig.RestEndpoints.Length
                        SwaggerEnabled = webApiConfig.Swagger.Enabled
                        GeneratedFiles = generatedApi.ProjectFiles.Keys |> Seq.toList
                    |}
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate REST API: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Creates a GraphQL server closure from metascript configuration
    member _.CreateGraphQLServerClosure(name: string, config: Map<string, obj>) =
        let webApiConfig = this.ParseWebApiConfig(name, config)
        
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🔧 Generating GraphQL Server: {webApiConfig.Name}"
                    
                    // Generate the GraphQL project
                    match graphqlGenerator.GenerateGraphQLProject(webApiConfig, outputDir) with
                    | Some graphqlProject ->
                        printfn $"✅ GraphQL Server generated successfully at: {outputDir}"
                        printfn $"🚀 GraphQL endpoint: {webApiConfig.BaseUrl}/graphql"
                        printfn $"🔍 GraphQL Voyager: {webApiConfig.BaseUrl}/graphql-voyager"
                        
                        return Ok {|
                            Type = "GRAPHQL_SERVER"
                            Name = webApiConfig.Name
                            OutputDirectory = outputDir
                            GraphQLEndpoint = $"{webApiConfig.BaseUrl}/graphql"
                            VoyagerEndpoint = $"{webApiConfig.BaseUrl}/graphql-voyager"
                            SchemaTypes = webApiConfig.GraphQLSchema.Value.Types.Length
                            Queries = webApiConfig.GraphQLSchema.Value.Queries.Length
                            Mutations = webApiConfig.GraphQLSchema.Value.Mutations.Length
                        |}
                    | None ->
                        return Error "No GraphQL schema configured"
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate GraphQL Server: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Creates a GraphQL client closure from schema URL
    member _.CreateGraphQLClientClosure(name: string, schemaUrl: string) =
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🔧 Generating GraphQL Client: {name}"
                    printfn $"📡 Schema URL: {schemaUrl}"
                    
                    // TODO: Fetch schema from URL and generate client
                    let clientConfig = WebApiHelpers.defaultWebApiConfig name
                    let clientCode = graphqlGenerator.GenerateGraphQLClientCode(clientConfig)
                    
                    let clientDir = Path.Combine(outputDir, "GraphQLClient")
                    Directory.CreateDirectory(clientDir) |> ignore
                    File.WriteAllText(Path.Combine(clientDir, "GraphQLClient.fs"), clientCode)
                    
                    printfn $"✅ GraphQL Client generated successfully at: {clientDir}"
                    
                    return Ok {|
                        Type = "GRAPHQL_CLIENT"
                        Name = name
                        OutputDirectory = clientDir
                        SchemaUrl = schemaUrl
                        ClientFile = "GraphQLClient.fs"
                    |}
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate GraphQL Client: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Creates a hybrid REST + GraphQL closure
    member _.CreateHybridApiClosure(name: string, config: Map<string, obj>) =
        fun (outputDir: string) ->
            async {
                try
                    printfn $"🔧 Generating Hybrid API (REST + GraphQL): {name}"
                    
                    let webApiConfig = this.ParseWebApiConfig(name, config)
                    
                    // Generate REST API
                    let generatedApi = restGenerator.GenerateWebApiProject(webApiConfig, outputDir)
                    
                    // Generate GraphQL if schema is configured
                    let graphqlResult = 
                        match webApiConfig.GraphQLSchema with
                        | Some _ -> graphqlGenerator.GenerateGraphQLProject(webApiConfig, outputDir)
                        | None -> None
                    
                    printfn $"✅ Hybrid API generated successfully at: {outputDir}"
                    printfn $"🔗 REST endpoints: {webApiConfig.RestEndpoints.Length}"
                    printfn $"📊 GraphQL types: {webApiConfig.GraphQLSchema |> Option.map (fun s -> s.Types.Length) |> Option.defaultValue 0}"
                    printfn $"📖 Swagger UI: {webApiConfig.BaseUrl}/swagger"
                    printfn $"🚀 GraphQL endpoint: {webApiConfig.BaseUrl}/graphql"
                    
                    return Ok {|
                        Type = "HYBRID_API"
                        Name = webApiConfig.Name
                        OutputDirectory = outputDir
                        BaseUrl = webApiConfig.BaseUrl
                        RestEndpoints = webApiConfig.RestEndpoints.Length
                        GraphQLTypes = webApiConfig.GraphQLSchema |> Option.map (fun s -> s.Types.Length) |> Option.defaultValue 0
                        SwaggerEnabled = webApiConfig.Swagger.Enabled
                        GraphQLEnabled = graphqlResult.IsSome
                    |}
                    
                with
                | ex ->
                    printfn $"❌ Failed to generate Hybrid API: {ex.Message}"
                    return Error ex.Message
            }
    
    /// Parses web API configuration from metascript parameters
    member _.ParseWebApiConfig(name: string, config: Map<string, obj>) =
        let mutable webApiConfig = WebApiHelpers.defaultWebApiConfig name
        
        // Parse basic configuration
        if config.ContainsKey("title") then
            webApiConfig <- { webApiConfig with Title = config.["title"].ToString() }
        
        if config.ContainsKey("description") then
            webApiConfig <- { webApiConfig with Description = config.["description"].ToString() }
        
        if config.ContainsKey("version") then
            webApiConfig <- { webApiConfig with Version = config.["version"].ToString() }
        
        if config.ContainsKey("base_url") then
            webApiConfig <- { webApiConfig with BaseUrl = config.["base_url"].ToString() }
        
        // Parse REST endpoints
        if config.ContainsKey("endpoints") then
            let endpoints = this.ParseRestEndpoints(config.["endpoints"])
            webApiConfig <- { webApiConfig with RestEndpoints = endpoints }
        
        // Parse GraphQL schema
        if config.ContainsKey("graphql") then
            let schema = this.ParseGraphQLSchema(config.["graphql"])
            webApiConfig <- { webApiConfig with GraphQLSchema = Some schema }
        
        // Parse authentication
        if config.ContainsKey("auth") then
            let auth = this.ParseAuthConfig(config.["auth"])
            webApiConfig <- { webApiConfig with Authentication = Some auth }
        
        webApiConfig
    
    /// Parses REST endpoints from configuration
    member _.ParseRestEndpoints(endpointsObj: obj) =
        // TODO: Implement proper parsing from metascript configuration
        // For now, return sample endpoints
        [
            WebApiHelpers.restEndpoint()
                .Route("/api/users")
                .Method(GET)
                .Name("GetUsers")
                .Description("Get all users")
                .Response(200, "User[]")
                .Build()
            
            WebApiHelpers.restEndpoint()
                .Route("/api/users/{id}")
                .Method(GET)
                .Name("GetUser")
                .Description("Get user by ID")
                .Parameter("id", Route "id", "int")
                .Response(200, "User")
                .Build()
            
            WebApiHelpers.restEndpoint()
                .Route("/api/users")
                .Method(POST)
                .Name("CreateUser")
                .Description("Create a new user")
                .Parameter("user", Body "user", "CreateUserRequest")
                .Response(201, "User")
                .Build()
        ]
    
    /// Parses GraphQL schema from configuration
    member _.ParseGraphQLSchema(schemaObj: obj) =
        // TODO: Implement proper parsing from metascript configuration
        // For now, return sample schema
        {
            Types = [
                WebApiHelpers.graphqlType "User" Object
                    .Description("A user in the system")
                    .Field("id", "ID!", "User identifier")
                    .Field("name", "String!", "User name")
                    .Field("email", "String!", "User email")
                    .Build()
                
                WebApiHelpers.graphqlType "CreateUserInput" InputObject
                    .Description("Input for creating a user")
                    .Field("name", "String!", "User name")
                    .Field("email", "String!", "User email")
                    .Build()
            ]
            Queries = [
                {
                    Name = "users"
                    Type = "[User!]!"
                    Description = Some "Get all users"
                    Arguments = []
                    Resolver = "resolveUsers"
                    Nullable = false
                }
                {
                    Name = "user"
                    Type = "User"
                    Description = Some "Get user by ID"
                    Arguments = [
                        {
                            Name = "id"
                            Type = Route "id"
                            DataType = "ID!"
                            Required = true
                            Description = Some "User ID"
                            DefaultValue = None
                        }
                    ]
                    Resolver = "resolveUser"
                    Nullable = true
                }
            ]
            Mutations = [
                {
                    Name = "createUser"
                    Type = "User!"
                    Description = Some "Create a new user"
                    Arguments = [
                        {
                            Name = "input"
                            Type = Body "input"
                            DataType = "CreateUserInput!"
                            Required = true
                            Description = Some "User creation input"
                            DefaultValue = None
                        }
                    ]
                    Resolver = "resolveCreateUser"
                    Nullable = false
                }
            ]
            Subscriptions = []
        }
    
    /// Parses authentication configuration
    member _.ParseAuthConfig(authObj: obj) =
        // TODO: Implement proper parsing
        {
            Type = JWT
            JwtSecret = Some "your-secret-key"
            ApiKeyHeader = None
            OAuthConfig = None
        }
    
    /// Gets available closure types
    member _.GetAvailableClosureTypes() =
        [
            "REST_ENDPOINT"
            "GRAPHQL_SERVER"
            "GRAPHQL_CLIENT"
            "HYBRID_API"
        ]
    
    /// Creates closure based on type
    member _.CreateClosure(closureType: string, name: string, config: Map<string, obj>) =
        match closureType.ToUpper() with
        | "REST_ENDPOINT" -> this.CreateRestEndpointClosure(name, config)
        | "GRAPHQL_SERVER" -> this.CreateGraphQLServerClosure(name, config)
        | "GRAPHQL_CLIENT" -> 
            let schemaUrl = config.TryFind("schema_url") |> Option.map (fun x -> x.ToString()) |> Option.defaultValue ""
            this.CreateGraphQLClientClosure(name, schemaUrl)
        | "HYBRID_API" -> this.CreateHybridApiClosure(name, config)
        | _ -> failwith $"Unknown closure type: {closureType}"
