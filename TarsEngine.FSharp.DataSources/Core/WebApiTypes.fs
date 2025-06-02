namespace TarsEngine.FSharp.DataSources.Core

open System
open System.Collections.Generic

/// HTTP methods supported by REST endpoints
type HttpMethod = 
    | GET | POST | PUT | DELETE | PATCH | HEAD | OPTIONS

/// Parameter types for REST endpoints
type ParameterType =
    | Route of string       // {id}
    | Query of string       // ?name=value
    | Header of string      // Authorization header
    | Body of string        // Request body type
    | Form of string        // Form data

/// REST endpoint parameter definition
type EndpointParameter = {
    Name: string
    Type: ParameterType
    DataType: string        // int, string, User, etc.
    Required: bool
    Description: string option
    DefaultValue: obj option
}

/// REST endpoint response definition
type EndpointResponse = {
    StatusCode: int
    DataType: string
    Description: string
    Headers: Map<string, string>
    Example: obj option
}

/// REST endpoint definition
type RestEndpoint = {
    Id: string
    Route: string           // "/api/users/{id}"
    Method: HttpMethod
    Name: string
    Description: string
    Parameters: EndpointParameter list
    Responses: EndpointResponse list
    Tags: string list
    RequiresAuth: bool
    RateLimit: int option   // requests per minute
    CacheSeconds: int option
    Implementation: string  // F# code for the endpoint
}

/// GraphQL field definition
type GraphQLField = {
    Name: string
    Type: string
    Description: string option
    Arguments: EndpointParameter list
    Resolver: string        // F# resolver function
    Nullable: bool
}

/// GraphQL type definition
type GraphQLType = {
    Name: string
    Kind: GraphQLTypeKind
    Description: string option
    Fields: GraphQLField list
    Interfaces: string list
}

and GraphQLTypeKind =
    | Object
    | Interface
    | Union
    | Enum
    | Scalar
    | InputObject

/// GraphQL schema definition
type GraphQLSchema = {
    Types: GraphQLType list
    Queries: GraphQLField list
    Mutations: GraphQLField list
    Subscriptions: GraphQLField list
}

/// Web API closure configuration
type WebApiClosureConfig = {
    Name: string
    BaseUrl: string
    Version: string
    Title: string
    Description: string
    RestEndpoints: RestEndpoint list
    GraphQLSchema: GraphQLSchema option
    Authentication: AuthConfig option
    Cors: CorsConfig option
    RateLimit: RateLimitConfig option
    Swagger: SwaggerConfig
}

and AuthConfig = {
    Type: AuthType
    JwtSecret: string option
    ApiKeyHeader: string option
    OAuthConfig: OAuthConfig option
}

and AuthType =
    | None
    | ApiKey
    | JWT
    | OAuth2
    | Basic

and OAuthConfig = {
    Authority: string
    ClientId: string
    Scopes: string list
}

and CorsConfig = {
    AllowedOrigins: string list
    AllowedMethods: string list
    AllowedHeaders: string list
    AllowCredentials: bool
}

and RateLimitConfig = {
    RequestsPerMinute: int
    BurstLimit: int
    ByIpAddress: bool
}

and SwaggerConfig = {
    Enabled: bool
    Title: string
    Version: string
    Description: string
    ContactName: string option
    ContactEmail: string option
    LicenseName: string option
    LicenseUrl: string option
}

/// Generated web API project
type GeneratedWebApi = {
    Config: WebApiClosureConfig
    ProjectFiles: Map<string, string>  // filename -> content
    StartupCode: string
    ControllerCode: string list
    GraphQLCode: string option
    SwaggerDefinition: string
    DockerFile: string
    ReadmeContent: string
}

/// Web API closure generation parameters
type WebApiClosureParameters = {
    Name: string
    OutputDirectory: string
    Config: WebApiClosureConfig
    Template: WebApiTemplate
    GenerateTests: bool
    GenerateDocumentation: bool
    GenerateDocker: bool
}

and WebApiTemplate =
    | MinimalApi           // ASP.NET Core Minimal APIs
    | ControllerBased      // Traditional MVC controllers
    | GraphQLOnly          // GraphQL server only
    | Hybrid               // REST + GraphQL

/// REST endpoint builder for fluent API
type RestEndpointBuilder() =
    let mutable endpoint = {
        Id = Guid.NewGuid().ToString("N")[..7]
        Route = ""
        Method = GET
        Name = ""
        Description = ""
        Parameters = []
        Responses = []
        Tags = []
        RequiresAuth = false
        RateLimit = None
        CacheSeconds = None
        Implementation = ""
    }
    
    member _.Route(route: string) =
        endpoint <- { endpoint with Route = route }
        this
    
    member _.Method(method: HttpMethod) =
        endpoint <- { endpoint with Method = method }
        this
    
    member _.Name(name: string) =
        endpoint <- { endpoint with Name = name }
        this
    
    member _.Description(description: string) =
        endpoint <- { endpoint with Description = description }
        this
    
    member _.Parameter(name: string, paramType: ParameterType, dataType: string, ?required: bool, ?description: string) =
        let param = {
            Name = name
            Type = paramType
            DataType = dataType
            Required = defaultArg required true
            Description = description
            DefaultValue = None
        }
        endpoint <- { endpoint with Parameters = param :: endpoint.Parameters }
        this
    
    member _.Response(statusCode: int, dataType: string, ?description: string) =
        let response = {
            StatusCode = statusCode
            DataType = dataType
            Description = defaultArg description ""
            Headers = Map.empty
            Example = None
        }
        endpoint <- { endpoint with Responses = response :: endpoint.Responses }
        this
    
    member _.RequiresAuth() =
        endpoint <- { endpoint with RequiresAuth = true }
        this
    
    member _.RateLimit(requestsPerMinute: int) =
        endpoint <- { endpoint with RateLimit = Some requestsPerMinute }
        this
    
    member _.Implementation(code: string) =
        endpoint <- { endpoint with Implementation = code }
        this
    
    member _.Build() = endpoint

/// GraphQL type builder for fluent API
type GraphQLTypeBuilder(name: string, kind: GraphQLTypeKind) =
    let mutable graphqlType = {
        Name = name
        Kind = kind
        Description = None
        Fields = []
        Interfaces = []
    }
    
    member _.Description(description: string) =
        graphqlType <- { graphqlType with Description = Some description }
        this
    
    member _.Field(name: string, fieldType: string, ?description: string, ?nullable: bool, ?resolver: string) =
        let field = {
            Name = name
            Type = fieldType
            Description = description
            Arguments = []
            Resolver = defaultArg resolver $"resolve{name}"
            Nullable = defaultArg nullable false
        }
        graphqlType <- { graphqlType with Fields = field :: graphqlType.Fields }
        this
    
    member _.Interface(interfaceName: string) =
        graphqlType <- { graphqlType with Interfaces = interfaceName :: graphqlType.Interfaces }
        this
    
    member _.Build() = graphqlType

/// Helper functions for web API generation
module WebApiHelpers =
    
    /// Creates a new REST endpoint builder
    let restEndpoint() = RestEndpointBuilder()
    
    /// Creates a new GraphQL type builder
    let graphqlType name kind = GraphQLTypeBuilder(name, kind)
    
    /// Creates a default web API configuration
    let defaultWebApiConfig name =
        {
            Name = name
            BaseUrl = "http://localhost:5000"
            Version = "1.0.0"
            Title = $"{name} API"
            Description = $"REST and GraphQL API for {name}"
            RestEndpoints = []
            GraphQLSchema = None
            Authentication = None
            Cors = Some {
                AllowedOrigins = ["*"]
                AllowedMethods = ["GET"; "POST"; "PUT"; "DELETE"]
                AllowedHeaders = ["*"]
                AllowCredentials = false
            }
            RateLimit = Some {
                RequestsPerMinute = 100
                BurstLimit = 20
                ByIpAddress = true
            }
            Swagger = {
                Enabled = true
                Title = $"{name} API"
                Version = "1.0.0"
                Description = $"API documentation for {name}"
                ContactName = None
                ContactEmail = None
                LicenseName = Some "MIT"
                LicenseUrl = Some "https://opensource.org/licenses/MIT"
            }
        }
    
    /// Converts HTTP method to string
    let httpMethodToString = function
        | GET -> "GET"
        | POST -> "POST"
        | PUT -> "PUT"
        | DELETE -> "DELETE"
        | PATCH -> "PATCH"
        | HEAD -> "HEAD"
        | OPTIONS -> "OPTIONS"
    
    /// Converts parameter type to route template
    let parameterToRoute param =
        match param.Type with
        | Route name -> $"{{{name}}}"
        | Query name -> $"?{name}={{value}}"
        | _ -> ""
