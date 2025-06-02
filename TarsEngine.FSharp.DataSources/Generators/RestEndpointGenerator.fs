namespace TarsEngine.FSharp.DataSources.Generators

open System
open System.IO
open System.Text
open TarsEngine.FSharp.DataSources.Core

/// REST endpoint code generator
type RestEndpointGenerator() =
    
    /// Generates F# controller code for REST endpoints
    member _.GenerateControllerCode(config: WebApiClosureConfig) =
        let sb = StringBuilder()
        
        // Generate namespace and imports
        sb.AppendLine($"namespace {config.Name}.Controllers") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine("open Microsoft.AspNetCore.Mvc") |> ignore
        sb.AppendLine("open Microsoft.Extensions.Logging") |> ignore
        sb.AppendLine("open System") |> ignore
        sb.AppendLine("open System.Threading.Tasks") |> ignore
        sb.AppendLine() |> ignore
        
        // Group endpoints by controller
        let controllerGroups = 
            config.RestEndpoints
            |> List.groupBy (fun ep -> 
                let parts = ep.Route.Split('/', StringSplitOptions.RemoveEmptyEntries)
                if parts.Length > 1 then parts.[1] else "Default")
        
        for (controllerName, endpoints) in controllerGroups do
            sb.AppendLine($"[<ApiController>]") |> ignore
            sb.AppendLine($"[<Route(\"api/[controller]\")>]") |> ignore
            sb.AppendLine($"type {controllerName}Controller(logger: ILogger<{controllerName}Controller>) =") |> ignore
            sb.AppendLine("    inherit ControllerBase()") |> ignore
            sb.AppendLine() |> ignore
            
            for endpoint in endpoints do
                this.GenerateEndpointMethod(sb, endpoint)
                sb.AppendLine() |> ignore
        
        sb.ToString()
    
    /// Generates a single endpoint method
    member _.GenerateEndpointMethod(sb: StringBuilder, endpoint: RestEndpoint) =
        // Generate attributes
        let httpMethod = WebApiHelpers.httpMethodToString endpoint.Method
        let route = endpoint.Route.Replace($"/api/{endpoint.Route.Split('/')[2]}", "")
        
        sb.AppendLine($"    [<Http{httpMethod}(\"{route}\")>]") |> ignore
        
        if endpoint.RequiresAuth then
            sb.AppendLine("    [<Authorize>]") |> ignore
        
        // Generate method signature
        let parameters = this.GenerateParameterList(endpoint.Parameters)
        let returnType = 
            match endpoint.Responses |> List.tryFind (fun r -> r.StatusCode = 200) with
            | Some response -> $"Task<ActionResult<{response.DataType}>>"
            | None -> "Task<IActionResult>"
        
        sb.AppendLine($"    member _.{endpoint.Name}({parameters}): {returnType} =") |> ignore
        sb.AppendLine("        task {") |> ignore
        sb.AppendLine($"            logger.LogInformation(\"Executing {endpoint.Name}\")") |> ignore
        sb.AppendLine() |> ignore
        
        // Generate implementation
        if String.IsNullOrEmpty(endpoint.Implementation) then
            // Generate default implementation
            match endpoint.Method with
            | GET -> 
                sb.AppendLine("            // TODO: Implement GET logic") |> ignore
                sb.AppendLine("            return Ok(\"GET response\")") |> ignore
            | POST ->
                sb.AppendLine("            // TODO: Implement POST logic") |> ignore
                sb.AppendLine("            return CreatedAtAction(nameof(Get), new { id = 1 }, \"Created\")") |> ignore
            | PUT ->
                sb.AppendLine("            // TODO: Implement PUT logic") |> ignore
                sb.AppendLine("            return Ok(\"Updated\")") |> ignore
            | DELETE ->
                sb.AppendLine("            // TODO: Implement DELETE logic") |> ignore
                sb.AppendLine("            return NoContent()") |> ignore
            | _ ->
                sb.AppendLine("            // TODO: Implement endpoint logic") |> ignore
                sb.AppendLine("            return Ok()") |> ignore
        else
            // Use custom implementation
            let lines = endpoint.Implementation.Split('\n')
            for line in lines do
                sb.AppendLine($"            {line.Trim()}") |> ignore
        
        sb.AppendLine("        }") |> ignore
    
    /// Generates parameter list for method signature
    member _.GenerateParameterList(parameters: EndpointParameter list) =
        parameters
        |> List.map (fun param ->
            match param.Type with
            | Route _ -> $"{param.Name}: {param.DataType}"
            | Query _ -> $"[<FromQuery>] {param.Name}: {param.DataType}"
            | Header _ -> $"[<FromHeader>] {param.Name}: {param.DataType}"
            | Body _ -> $"[<FromBody>] {param.Name}: {param.DataType}"
            | Form _ -> $"[<FromForm>] {param.Name}: {param.DataType}"
        )
        |> String.concat ", "
    
    /// Generates Program.fs for the web API
    member _.GenerateProgramCode(config: WebApiClosureConfig) =
        let sb = StringBuilder()
        
        sb.AppendLine($"namespace {config.Name}") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine("open Microsoft.AspNetCore.Builder") |> ignore
        sb.AppendLine("open Microsoft.Extensions.DependencyInjection") |> ignore
        sb.AppendLine("open Microsoft.Extensions.Hosting") |> ignore
        sb.AppendLine("open Microsoft.OpenApi.Models") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine("module Program =") |> ignore
        sb.AppendLine("    [<EntryPoint>]") |> ignore
        sb.AppendLine("    let main args =") |> ignore
        sb.AppendLine("        let builder = WebApplication.CreateBuilder(args)") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine("        // Add services") |> ignore
        sb.AppendLine("        builder.Services.AddControllers() |> ignore") |> ignore
        sb.AppendLine("        builder.Services.AddEndpointsApiExplorer() |> ignore") |> ignore
        
        // Add Swagger configuration
        if config.Swagger.Enabled then
            sb.AppendLine("        builder.Services.AddSwaggerGen(fun c ->") |> ignore
            sb.AppendLine($"            c.SwaggerDoc(\"v1\", OpenApiInfo(") |> ignore
            sb.AppendLine($"                Title = \"{config.Swagger.Title}\",") |> ignore
            sb.AppendLine($"                Version = \"{config.Swagger.Version}\",") |> ignore
            sb.AppendLine($"                Description = \"{config.Swagger.Description}\"") |> ignore
            sb.AppendLine("            ))") |> ignore
            sb.AppendLine("        ) |> ignore") |> ignore
        
        // Add authentication if configured
        match config.Authentication with
        | Some auth when auth.Type = JWT ->
            sb.AppendLine("        builder.Services.AddAuthentication(\"Bearer\")") |> ignore
            sb.AppendLine("            .AddJwtBearer() |> ignore") |> ignore
        | Some auth when auth.Type = ApiKey ->
            sb.AppendLine("        // TODO: Add API Key authentication") |> ignore
        | _ -> ()
        
        // Add CORS if configured
        match config.Cors with
        | Some cors ->
            sb.AppendLine("        builder.Services.AddCors(fun options ->") |> ignore
            sb.AppendLine("            options.AddDefaultPolicy(fun policy ->") |> ignore
            sb.AppendLine($"                policy.WithOrigins({cors.AllowedOrigins |> List.map (sprintf "\"%s\"") |> String.concat ", "})") |> ignore
            sb.AppendLine($"                      .WithMethods({cors.AllowedMethods |> List.map (sprintf "\"%s\"") |> String.concat ", "})") |> ignore
            sb.AppendLine($"                      .WithHeaders({cors.AllowedHeaders |> List.map (sprintf "\"%s\"") |> String.concat ", "})") |> ignore
            if cors.AllowCredentials then
                sb.AppendLine("                      .AllowCredentials()") |> ignore
            sb.AppendLine("            )") |> ignore
            sb.AppendLine("        ) |> ignore") |> ignore
        | None -> ()
        
        sb.AppendLine() |> ignore
        sb.AppendLine("        let app = builder.Build()") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine("        // Configure pipeline") |> ignore
        sb.AppendLine("        if app.Environment.IsDevelopment() then") |> ignore
        
        if config.Swagger.Enabled then
            sb.AppendLine("            app.UseSwagger() |> ignore") |> ignore
            sb.AppendLine("            app.UseSwaggerUI() |> ignore") |> ignore
        
        sb.AppendLine() |> ignore
        sb.AppendLine("        app.UseHttpsRedirection() |> ignore") |> ignore
        
        if config.Cors.IsSome then
            sb.AppendLine("        app.UseCors() |> ignore") |> ignore
        
        if config.Authentication.IsSome then
            sb.AppendLine("        app.UseAuthentication() |> ignore") |> ignore
            sb.AppendLine("        app.UseAuthorization() |> ignore") |> ignore
        
        sb.AppendLine("        app.UseRouting() |> ignore") |> ignore
        sb.AppendLine("        app.MapControllers() |> ignore") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine("        // Add health check") |> ignore
        sb.AppendLine("        app.MapGet(\"/health\", fun () ->") |> ignore
        sb.AppendLine($"            {{\"status\": \"healthy\", \"service\": \"{config.Name}\", \"version\": \"{config.Version}\"}}) |> ignore") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine($"        printfn \"🚀 {config.Title} starting on {config.BaseUrl}\"") |> ignore
        sb.AppendLine($"        printfn \"📖 Swagger UI: {config.BaseUrl}/swagger\"") |> ignore
        sb.AppendLine($"        printfn \"❤️ Health check: {config.BaseUrl}/health\"") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine($"        app.Run(\"{config.BaseUrl}\")") |> ignore
        sb.AppendLine("        0") |> ignore
        
        sb.ToString()
    
    /// Generates project file (.fsproj)
    member _.GenerateProjectFile(config: WebApiClosureConfig) =
        $"""<Project Sdk="Microsoft.NET.Sdk.Web">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <AssemblyName>{config.Name}</AssemblyName>
    <RootNamespace>{config.Name}</RootNamespace>
  </PropertyGroup>

  <ItemGroup>
    <Compile Include="Controllers/*.fs" />
    <Compile Include="Program.fs" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.AspNetCore.OpenApi" Version="8.0.0" />
    <PackageReference Include="Swashbuckle.AspNetCore" Version="6.5.0" />
    <PackageReference Include="Microsoft.AspNetCore.Authentication.JwtBearer" Version="8.0.0" />
    <PackageReference Include="Microsoft.AspNetCore.Cors" Version="2.2.0" />
  </ItemGroup>

</Project>"""
    
    /// Generates Swagger/OpenAPI specification
    member _.GenerateSwaggerSpec(config: WebApiClosureConfig) =
        let sb = StringBuilder()
        
        sb.AppendLine("{") |> ignore
        sb.AppendLine("  \"openapi\": \"3.0.1\",") |> ignore
        sb.AppendLine("  \"info\": {") |> ignore
        sb.AppendLine($"    \"title\": \"{config.Swagger.Title}\",") |> ignore
        sb.AppendLine($"    \"version\": \"{config.Swagger.Version}\",") |> ignore
        sb.AppendLine($"    \"description\": \"{config.Swagger.Description}\"") |> ignore
        sb.AppendLine("  },") |> ignore
        sb.AppendLine("  \"paths\": {") |> ignore
        
        let endpointPaths = 
            config.RestEndpoints
            |> List.map (fun ep ->
                let method = (WebApiHelpers.httpMethodToString ep.Method).ToLower()
                $"""    "{ep.Route}": {{
      "{method}": {{
        "tags": [{ep.Tags |> List.map (sprintf "\"%s\"") |> String.concat ", "}],
        "summary": "{ep.Description}",
        "operationId": "{ep.Name}",
        "responses": {{
          "200": {{
            "description": "Success"
          }}
        }}
      }}
    }}""")
        
        sb.AppendLine(String.Join(",\n", endpointPaths)) |> ignore
        sb.AppendLine("  }") |> ignore
        sb.AppendLine("}") |> ignore
        
        sb.ToString()
    
    /// Generates complete web API project
    member _.GenerateWebApiProject(config: WebApiClosureConfig, outputDir: string) =
        let projectFiles = Dictionary<string, string>()
        
        // Generate main files
        projectFiles.["Program.fs"] <- this.GenerateProgramCode(config)
        projectFiles.[$"{config.Name}.fsproj"] <- this.GenerateProjectFile(config)
        projectFiles.["Controllers/ApiControllers.fs"] <- this.GenerateControllerCode(config)
        
        if config.Swagger.Enabled then
            projectFiles.["swagger.json"] <- this.GenerateSwaggerSpec(config)
        
        // Generate Docker file
        projectFiles.["Dockerfile"] <- this.GenerateDockerFile(config)
        
        // Generate README
        projectFiles.["README.md"] <- this.GenerateReadme(config)
        
        // Create output directory and write files
        Directory.CreateDirectory(outputDir) |> ignore
        Directory.CreateDirectory(Path.Combine(outputDir, "Controllers")) |> ignore
        
        for kvp in projectFiles do
            let filePath = Path.Combine(outputDir, kvp.Key)
            File.WriteAllText(filePath, kvp.Value)
        
        {
            Config = config
            ProjectFiles = projectFiles |> Seq.map (|KeyValue|) |> Map.ofSeq
            StartupCode = this.GenerateProgramCode(config)
            ControllerCode = [this.GenerateControllerCode(config)]
            GraphQLCode = None
            SwaggerDefinition = this.GenerateSwaggerSpec(config)
            DockerFile = this.GenerateDockerFile(config)
            ReadmeContent = this.GenerateReadme(config)
        }
    
    /// Generates Dockerfile
    member _.GenerateDockerFile(config: WebApiClosureConfig) =
        $"""FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 5000

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["{config.Name}.fsproj", "."]
RUN dotnet restore "{config.Name}.fsproj"
COPY . .
WORKDIR "/src/."
RUN dotnet build "{config.Name}.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "{config.Name}.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "{config.Name}.dll"]"""
    
    /// Generates README.md
    member _.GenerateReadme(config: WebApiClosureConfig) =
        $"""# {config.Title}

{config.Description}

## 🚀 Quick Start

### Prerequisites
- .NET 8.0 SDK
- Docker (optional)

### Build and Run

```bash
# Build the project
dotnet build

# Run the application
dotnet run
```

### Docker Deployment

```bash
# Build Docker image
docker build -t {config.Name.ToLower()} .

# Run container
docker run -p 5000:5000 {config.Name.ToLower()}
```

## 📖 API Documentation

- Base URL: {config.BaseUrl}
- Health Check: {config.BaseUrl}/health
- Swagger UI: {config.BaseUrl}/swagger

## 🔗 Endpoints

{config.RestEndpoints |> List.map (fun ep -> $"- {WebApiHelpers.httpMethodToString ep.Method} {ep.Route} - {ep.Description}") |> String.concat "\n"}

## 🧪 Testing

```bash
# Run tests
dotnet test
```

Generated by TARS REST Endpoint Closure Factory 🤖
"""
