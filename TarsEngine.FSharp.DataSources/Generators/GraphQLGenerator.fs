namespace TarsEngine.FSharp.DataSources.Generators

open System
open System.IO
open System.Text
open TarsEngine.FSharp.DataSources.Core

/// GraphQL server and client code generator
type GraphQLGenerator() =
    
    /// Generates GraphQL schema definition (SDL)
    member _.GenerateSchemaDefinition(schema: GraphQLSchema) =
        let sb = StringBuilder()
        
        // Generate custom types
        for graphqlType in schema.Types do
            match graphqlType.Kind with
            | Object ->
                sb.AppendLine($"type {graphqlType.Name} {{") |> ignore
                for field in graphqlType.Fields do
                    let nullable = if field.Nullable then "" else "!"
                    let args = 
                        if field.Arguments.IsEmpty then ""
                        else $"({field.Arguments |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}") |> String.concat ", "})"
                    sb.AppendLine($"  {field.Name}{args}: {field.Type}{nullable}") |> ignore
                sb.AppendLine("}") |> ignore
                sb.AppendLine() |> ignore
            
            | Interface ->
                sb.AppendLine($"interface {graphqlType.Name} {{") |> ignore
                for field in graphqlType.Fields do
                    sb.AppendLine($"  {field.Name}: {field.Type}") |> ignore
                sb.AppendLine("}") |> ignore
                sb.AppendLine() |> ignore
            
            | Enum ->
                sb.AppendLine($"enum {graphqlType.Name} {{") |> ignore
                for field in graphqlType.Fields do
                    sb.AppendLine($"  {field.Name}") |> ignore
                sb.AppendLine("}") |> ignore
                sb.AppendLine() |> ignore
            
            | _ -> ()
        
        // Generate Query type
        if not schema.Queries.IsEmpty then
            sb.AppendLine("type Query {") |> ignore
            for query in schema.Queries do
                let args = 
                    if query.Arguments.IsEmpty then ""
                    else $"({query.Arguments |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}") |> String.concat ", "})"
                sb.AppendLine($"  {query.Name}{args}: {query.Type}") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
        
        // Generate Mutation type
        if not schema.Mutations.IsEmpty then
            sb.AppendLine("type Mutation {") |> ignore
            for mutation in schema.Mutations do
                let args = 
                    if mutation.Arguments.IsEmpty then ""
                    else $"({mutation.Arguments |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}") |> String.concat ", "})"
                sb.AppendLine($"  {mutation.Name}{args}: {mutation.Type}") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
        
        // Generate Subscription type
        if not schema.Subscriptions.IsEmpty then
            sb.AppendLine("type Subscription {") |> ignore
            for subscription in schema.Subscriptions do
                sb.AppendLine($"  {subscription.Name}: {subscription.Type}") |> ignore
            sb.AppendLine("}") |> ignore
        
        sb.ToString()
    
    /// Generates F# GraphQL server code using HotChocolate
    member _.GenerateGraphQLServerCode(config: WebApiClosureConfig) =
        match config.GraphQLSchema with
        | None -> ""
        | Some schema ->
            let sb = StringBuilder()
            
            sb.AppendLine($"namespace {config.Name}.GraphQL") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine("open HotChocolate") |> ignore
            sb.AppendLine("open HotChocolate.Types") |> ignore
            sb.AppendLine("open Microsoft.Extensions.DependencyInjection") |> ignore
            sb.AppendLine("open System.Threading.Tasks") |> ignore
            sb.AppendLine() |> ignore
            
            // Generate data types
            for graphqlType in schema.Types do
                if graphqlType.Kind = Object then
                    sb.AppendLine($"type {graphqlType.Name} = {{") |> ignore
                    for field in graphqlType.Fields do
                        sb.AppendLine($"    {field.Name}: {field.Type}") |> ignore
                    sb.AppendLine("}") |> ignore
                    sb.AppendLine() |> ignore
            
            // Generate Query class
            if not schema.Queries.IsEmpty then
                sb.AppendLine("type Query() =") |> ignore
                for query in schema.Queries do
                    let parameters = 
                        query.Arguments 
                        |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}")
                        |> String.concat ", "
                    
                    sb.AppendLine($"    member _.{query.Name}({parameters}): Task<{query.Type}> =") |> ignore
                    sb.AppendLine("        task {") |> ignore
                    sb.AppendLine($"            // TODO: Implement {query.Name} resolver") |> ignore
                    sb.AppendLine($"            return Unchecked.defaultof<{query.Type}>") |> ignore
                    sb.AppendLine("        }") |> ignore
                    sb.AppendLine() |> ignore
            
            // Generate Mutation class
            if not schema.Mutations.IsEmpty then
                sb.AppendLine("type Mutation() =") |> ignore
                for mutation in schema.Mutations do
                    let parameters = 
                        mutation.Arguments 
                        |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}")
                        |> String.concat ", "
                    
                    sb.AppendLine($"    member _.{mutation.Name}({parameters}): Task<{mutation.Type}> =") |> ignore
                    sb.AppendLine("        task {") |> ignore
                    sb.AppendLine($"            // TODO: Implement {mutation.Name} resolver") |> ignore
                    sb.AppendLine($"            return Unchecked.defaultof<{mutation.Type}>") |> ignore
                    sb.AppendLine("        }") |> ignore
                    sb.AppendLine() |> ignore
            
            // Generate Subscription class
            if not schema.Subscriptions.IsEmpty then
                sb.AppendLine("type Subscription() =") |> ignore
                for subscription in schema.Subscriptions do
                    sb.AppendLine($"    member _.{subscription.Name}(): IAsyncEnumerable<{subscription.Type}> =") |> ignore
                    sb.AppendLine($"        // TODO: Implement {subscription.Name} subscription") |> ignore
                    sb.AppendLine($"        AsyncEnumerable.Empty<{subscription.Type}>()") |> ignore
                    sb.AppendLine() |> ignore
            
            sb.ToString()
    
    /// Generates GraphQL client code
    member _.GenerateGraphQLClientCode(config: WebApiClosureConfig) =
        match config.GraphQLSchema with
        | None -> ""
        | Some schema ->
            let sb = StringBuilder()
            
            sb.AppendLine($"namespace {config.Name}.Client") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine("open System") |> ignore
            sb.AppendLine("open System.Net.Http") |> ignore
            sb.AppendLine("open System.Text") |> ignore
            sb.AppendLine("open System.Text.Json") |> ignore
            sb.AppendLine("open System.Threading.Tasks") |> ignore
            sb.AppendLine() |> ignore
            
            // Generate GraphQL request types
            sb.AppendLine("type GraphQLRequest = {") |> ignore
            sb.AppendLine("    Query: string") |> ignore
            sb.AppendLine("    Variables: Map<string, obj> option") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
            
            sb.AppendLine("type GraphQLResponse<'T> = {") |> ignore
            sb.AppendLine("    Data: 'T option") |> ignore
            sb.AppendLine("    Errors: GraphQLError list option") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
            
            sb.AppendLine("and GraphQLError = {") |> ignore
            sb.AppendLine("    Message: string") |> ignore
            sb.AppendLine("    Path: string list option") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
            
            // Generate client class
            sb.AppendLine($"type {config.Name}GraphQLClient(httpClient: HttpClient, endpoint: string) =") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine("    member private _.ExecuteAsync<'T>(request: GraphQLRequest): Task<GraphQLResponse<'T>> =") |> ignore
            sb.AppendLine("        task {") |> ignore
            sb.AppendLine("            let json = JsonSerializer.Serialize(request)") |> ignore
            sb.AppendLine("            let content = new StringContent(json, Encoding.UTF8, \"application/json\")") |> ignore
            sb.AppendLine("            let! response = httpClient.PostAsync(endpoint, content)") |> ignore
            sb.AppendLine("            let! responseJson = response.Content.ReadAsStringAsync()") |> ignore
            sb.AppendLine("            return JsonSerializer.Deserialize<GraphQLResponse<'T>>(responseJson)") |> ignore
            sb.AppendLine("        }") |> ignore
            sb.AppendLine() |> ignore
            
            // Generate query methods
            for query in schema.Queries do
                let parameters = 
                    query.Arguments 
                    |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}")
                    |> String.concat ", "
                
                let variables = 
                    if query.Arguments.IsEmpty then "None"
                    else 
                        let varMap = 
                            query.Arguments 
                            |> List.map (fun arg -> $"(\"{arg.Name}\", box {arg.Name})")
                            |> String.concat "; "
                        $"Some (Map.ofList [{varMap}])"
                
                sb.AppendLine($"    member _.{query.Name}({parameters}): Task<GraphQLResponse<{query.Type}>> =") |> ignore
                sb.AppendLine("        let request = {") |> ignore
                sb.AppendLine($"            Query = \"query {{ {query.Name} }}\"") |> ignore
                sb.AppendLine($"            Variables = {variables}") |> ignore
                sb.AppendLine("        }") |> ignore
                sb.AppendLine($"        this.ExecuteAsync<{query.Type}>(request)") |> ignore
                sb.AppendLine() |> ignore
            
            // Generate mutation methods
            for mutation in schema.Mutations do
                let parameters = 
                    mutation.Arguments 
                    |> List.map (fun arg -> $"{arg.Name}: {arg.DataType}")
                    |> String.concat ", "
                
                sb.AppendLine($"    member _.{mutation.Name}({parameters}): Task<GraphQLResponse<{mutation.Type}>> =") |> ignore
                sb.AppendLine("        let request = {") |> ignore
                sb.AppendLine($"            Query = \"mutation {{ {mutation.Name} }}\"") |> ignore
                sb.AppendLine("            Variables = None") |> ignore
                sb.AppendLine("        }") |> ignore
                sb.AppendLine($"        this.ExecuteAsync<{mutation.Type}>(request)") |> ignore
                sb.AppendLine() |> ignore
            
            sb.ToString()
    
    /// Generates GraphQL server startup configuration
    member _.GenerateGraphQLStartupCode(config: WebApiClosureConfig) =
        match config.GraphQLSchema with
        | None -> ""
        | Some schema ->
            let sb = StringBuilder()
            
            sb.AppendLine("        // Add GraphQL services") |> ignore
            sb.AppendLine("        builder.Services") |> ignore
            sb.AppendLine("            .AddGraphQLServer()") |> ignore
            
            if not schema.Queries.IsEmpty then
                sb.AppendLine("            .AddQueryType<Query>()") |> ignore
            
            if not schema.Mutations.IsEmpty then
                sb.AppendLine("            .AddMutationType<Mutation>()") |> ignore
            
            if not schema.Subscriptions.IsEmpty then
                sb.AppendLine("            .AddSubscriptionType<Subscription>()") |> ignore
                sb.AppendLine("            .AddInMemorySubscriptions()") |> ignore
            
            sb.AppendLine("            .AddProjections()") |> ignore
            sb.AppendLine("            .AddFiltering()") |> ignore
            sb.AppendLine("            .AddSorting() |> ignore") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine("        // Configure GraphQL endpoint") |> ignore
            sb.AppendLine("        app.MapGraphQL(\"/graphql\") |> ignore") |> ignore
            
            if config.Swagger.Enabled then
                sb.AppendLine("        app.MapGraphQLVoyager(\"/graphql-voyager\") |> ignore") |> ignore
            
            sb.ToString()
    
    /// Generates complete GraphQL project
    member _.GenerateGraphQLProject(config: WebApiClosureConfig, outputDir: string) =
        match config.GraphQLSchema with
        | None -> None
        | Some schema ->
            let schemaCode = this.GenerateGraphQLServerCode(config)
            let clientCode = this.GenerateGraphQLClientCode(config)
            let schemaDefinition = this.GenerateSchemaDefinition(schema)
            
            // Write GraphQL files
            let graphqlDir = Path.Combine(outputDir, "GraphQL")
            Directory.CreateDirectory(graphqlDir) |> ignore
            
            File.WriteAllText(Path.Combine(graphqlDir, "Schema.fs"), schemaCode)
            File.WriteAllText(Path.Combine(graphqlDir, "Client.fs"), clientCode)
            File.WriteAllText(Path.Combine(graphqlDir, "schema.graphql"), schemaDefinition)
            
            Some {|
                SchemaCode = schemaCode
                ClientCode = clientCode
                SchemaDefinition = schemaDefinition
                StartupCode = this.GenerateGraphQLStartupCode(config)
            |}
    
    /// Generates GraphQL project file additions
    member _.GenerateGraphQLProjectReferences() =
        """    <PackageReference Include="HotChocolate.AspNetCore" Version="13.5.1" />
    <PackageReference Include="HotChocolate.Data" Version="13.5.1" />
    <PackageReference Include="HotChocolate.Subscriptions.InMemory" Version="13.5.1" />
    <PackageReference Include="GraphQL.Client" Version="6.0.0" />
    <PackageReference Include="GraphQL.Client.Serializer.SystemTextJson" Version="6.0.0" />"""
