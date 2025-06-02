
def generate_api_client(openapi_spec):
    """Generate F# API client from OpenAPI specification"""
    
    client_code = f"""
type {openapi_spec['info']['title']}Client(baseUrl: string, apiKey: string) =
    let httpClient = new HttpClient()
    
    // Generated endpoint methods
    {generate_endpoint_methods(openapi_spec['paths'])}
    
    // Generated data types
    {generate_data_types(openapi_spec['components']['schemas'])}
    
    // Generated authentication
    {generate_auth_methods(openapi_spec['security'])}
"""
    
    return compile_fsharp_client(client_code)

# Example: Generate GitHub API client
github_client = generate_api_client({
    "info": {"title": "GitHub"},
    "paths": {
        "/repos/{owner}/{repo}": {"get": "getRepository"},
        "/user/repos": {"get": "getUserRepos"}
    },
    "components": {"schemas": {"Repository": "..."}},
    "security": [{"bearerAuth": []}]
})
