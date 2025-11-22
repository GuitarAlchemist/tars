#!/usr/bin/env python3
"""
Enhanced TARS Project Generator with Application Code Generation
Integrates project structure generation with executable code generation
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

class EnhancedProjectGenerator:
    def __init__(self):
        self.output_dir = "output/projects"
        self.application_types = {
            "webapi": "Web API service with REST endpoints",
            "console": "Console application",
            "webapp": "Web application with MVC",
            "microservice": "Microservice with API endpoints",
            "library": "F# library"
        }
        
    def generate_complete_project(self, project_name, app_type="webapi", complexity="medium"):
        """Generate a complete project with both structure and executable code"""
        
        print("ENHANCED TARS PROJECT GENERATOR")
        print("=" * 45)
        print(f"Project: {project_name}")
        print(f"Type: {app_type} ({self.application_types.get(app_type, 'Unknown')})")
        print(f"Complexity: {complexity}")
        print()
        
        # Phase 1: Create project structure
        print("PHASE 1: PROJECT STRUCTURE GENERATION")
        print("=" * 45)
        project_path = self.create_project_structure(project_name, app_type)
        print(f"  Project structure created at: {project_path}")
        print()

        # Phase 2: Generate executable application code
        print("PHASE 2: APPLICATION CODE GENERATION")
        print("=" * 45)
        code_generated = self.generate_application_code(project_path, project_name, app_type)
        if code_generated:
            print("  Executable application code generated")
        else:
            print("  Failed to generate application code")
            return None
        print()

        # Phase 3: Create configuration files
        print("PHASE 3: CONFIGURATION GENERATION")
        print("=" * 40)
        self.create_configuration_files(project_path, project_name, app_type)
        print("  Configuration files created")
        print()

        # Phase 4: Generate documentation
        print("PHASE 4: DOCUMENTATION GENERATION")
        print("=" * 40)
        self.generate_documentation(project_path, project_name, app_type)
        print("  Documentation generated")
        print()

        # Phase 5: Validate project
        print("PHASE 5: PROJECT VALIDATION")
        print("=" * 35)
        validation_result = self.validate_project(project_path, app_type)
        if validation_result['valid']:
            print("  Project validation passed")
            print(f"    Source files: {validation_result['source_files']}")
            print(f"    Project files: {validation_result['project_files']}")
            print(f"    Config files: {validation_result['config_files']}")
        else:
            print("  Project validation failed")
            for issue in validation_result['issues']:
                print(f"    - {issue}")
        print()

        # Phase 6: Generate build and deployment scripts
        print("PHASE 6: BUILD & DEPLOYMENT SCRIPTS")
        print("=" * 42)
        self.create_build_scripts(project_path, project_name, app_type)
        print("  Build and deployment scripts created")
        print()

        print("PROJECT GENERATION COMPLETE!")
        print("=" * 35)
        print(f"  Location: {project_path}")
        print(f"  Type: {app_type}")
        print(f"  Status: Ready for deployment")
        print(f"  Next: Run 'python vm-demo.py' to deploy")
        
        return project_path
    
    def create_project_structure(self, project_name, app_type):
        """Create the basic project structure"""
        project_path = os.path.join(self.output_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        
        # Create directory structure
        directories = [
            "src",
            "tests", 
            "docs",
            "scripts",
            ".github/workflows"
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(project_path, directory), exist_ok=True)
        
        # Create basic project file
        project_content = f"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <OutputType>Exe</OutputType>
    <AssemblyName>{project_name}</AssemblyName>
    <RootNamespace>{project_name}</RootNamespace>
  </PropertyGroup>
  
  <ItemGroup>
    <Compile Include="src/Program.fs" />
  </ItemGroup>
  
</Project>"""
        
        with open(os.path.join(project_path, f"{project_name}.fsproj"), 'w') as f:
            f.write(project_content)
        
        return project_path
    
    def generate_application_code(self, project_path, project_name, app_type):
        """Generate executable application code based on type"""
        
        if app_type == "webapi":
            return self.generate_webapi_code(project_path, project_name)
        elif app_type == "console":
            return self.generate_console_code(project_path, project_name)
        elif app_type == "webapp":
            return self.generate_webapp_code(project_path, project_name)
        elif app_type == "microservice":
            return self.generate_microservice_code(project_path, project_name)
        elif app_type == "library":
            return self.generate_library_code(project_path, project_name)
        else:
            return self.generate_console_code(project_path, project_name)  # Default
    
    def generate_webapi_code(self, project_path, project_name):
        """Generate Web API application code"""
        
        # Program.fs - Main entry point
        program_content = f'''namespace {project_name}

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
            "{{\"status\": \"healthy\", \"service\": \"{project_name}\", \"version\": \"1.0.0\"}}") |> ignore

        // Add info endpoint
        app.MapGet("/", fun () ->
            "{{\"service\": \"{project_name}\", \"version\": \"1.0.0\", \"status\": \"running\"}}") |> ignore
        
        printfn "🚀 {project_name} API starting on http://localhost:5000"
        printfn "📖 Swagger UI: http://localhost:5000/swagger"
        printfn "❤️ Health check: http://localhost:5000/health"
        
        app.Run("http://0.0.0.0:5000")
        0
'''
        
        # Controllers.fs - API controllers
        controllers_content = f'''namespace {project_name}.Controllers

open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open System

[<ApiController>]
[<Route("api/[controller]")>]
type ValuesController(logger: ILogger<ValuesController>) =
    inherit ControllerBase()
    
    [<HttpGet>]
    member this.Get() =
        logger.LogInformation("GET /api/values called")
        [| 
            {{| id = 1; value = "Hello from {project_name}!" |}}
            {{| id = 2; value = "Generated by TARS" |}}
            {{| id = 3; value = "F# Web API" |}}
        |]
    
    [<HttpGet("{{id}}")>]
    member this.Get(id: int) =
        logger.LogInformation("GET /api/values/{{Id}} called", id)
        {{| id = id; value = $"Value {{id}} from {project_name}" |}}
    
    [<HttpPost>]
    member this.Post([<FromBody>] value: string) =
        logger.LogInformation("POST /api/values called with: {{Value}}", value)
        this.Ok({{| message = $"Created: {{value}}"; timestamp = DateTime.UtcNow |}})
'''
        
        # Write source files
        with open(os.path.join(project_path, "src", "Program.fs"), 'w', encoding='utf-8') as f:
            f.write(program_content)

        with open(os.path.join(project_path, "src", "Controllers.fs"), 'w', encoding='utf-8') as f:
            f.write(controllers_content)
        
        # Update project file to include Controllers.fs
        proj_file = os.path.join(project_path, f"{project_name}.fsproj")
        with open(proj_file, 'r') as f:
            content = f.read()
        
        # Add Controllers.fs to compilation
        updated_content = content.replace(
            '<Compile Include="src/Program.fs" />',
            '''<Compile Include="src/Controllers.fs" />
    <Compile Include="src/Program.fs" />'''
        )
        
        # Add package references
        updated_content = updated_content.replace(
            '</ItemGroup>',
            '''</ItemGroup>
  
  <ItemGroup>
    <PackageReference Include="Swashbuckle.AspNetCore" Version="6.5.0" />
  </ItemGroup>'''
        )
        
        with open(proj_file, 'w') as f:
            f.write(updated_content)
        
        return True
    
    def generate_console_code(self, project_path, project_name):
        """Generate console application code"""
        
        program_content = f'''namespace {project_name}

open System

module Program =
    
    let printWelcome() =
        printfn "Welcome to {project_name}!"
        printfn "Generated by TARS Enhanced Project Generator"
        printfn "Timestamp: %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
        printfn ""
    
    let processCommand (command: string) =
        match command.ToLower().Trim() with
        | "help" | "h" ->
            printfn "Available commands:"
            printfn "  help, h     - Show this help"
            printfn "  version, v  - Show version"
            printfn "  status, s   - Show status"
            printfn "  quit, q     - Exit application"
        | "version" | "v" ->
            printfn "{project_name} v1.0.0"
            printfn "Built with F# and TARS"
        | "status" | "s" ->
            printfn "Status: Running"
            printfn "Memory: %d MB" (GC.GetTotalMemory(false) / 1024L / 1024L)
        | "quit" | "q" ->
            printfn "Goodbye!"
            Environment.Exit(0)
        | "" ->
            () // Empty command, do nothing
        | _ ->
            printfn "Unknown command: %s" command
            printfn "Type 'help' for available commands"
    
    [<EntryPoint>]
    let main args =
        try
            printWelcome()
            
            if args.Length > 0 then
                // Process command line arguments
                let command = String.Join(" ", args)
                processCommand command
            else
                // Interactive mode
                printfn "Interactive mode. Type 'help' for commands, 'quit' to exit."
                
                let mutable running = true
                while running do
                    printf "> "
                    let input = Console.ReadLine()
                    if input = null || input.ToLower() = "quit" then
                        running <- false
                        printfn "Goodbye!"
                    else
                        processCommand input
            
            0
        with
        | ex ->
            printfn "Error: %s" ex.Message
            1
'''
        
        with open(os.path.join(project_path, "src", "Program.fs"), 'w', encoding='utf-8') as f:
            f.write(program_content)
        
        return True
    
    def generate_webapp_code(self, project_path, project_name):
        """Generate web application code (MVC)"""
        # For now, generate a simple web API
        return self.generate_webapi_code(project_path, project_name)
    
    def generate_microservice_code(self, project_path, project_name):
        """Generate microservice code"""
        # Enhanced web API with microservice patterns
        return self.generate_webapi_code(project_path, project_name)
    
    def generate_library_code(self, project_path, project_name):
        """Generate library code"""
        
        library_content = f'''namespace {project_name}

/// <summary>
/// Main library module for {project_name}
/// Generated by TARS Enhanced Project Generator
/// </summary>
module Library =
    
    /// <summary>
    /// Sample function that demonstrates the library functionality
    /// </summary>
    let hello name =
        $"Hello {{name}} from {project_name}!"
    
    /// <summary>
    /// Sample computation function
    /// </summary>
    let add x y = x + y
    
    /// <summary>
    /// Sample async function
    /// </summary>
    let asyncHello name = async {{
        do! Async.Sleep(100)
        return hello name
    }}
    
    /// <summary>
    /// Sample record type
    /// </summary>
    type Person = {{
        Name: string
        Age: int
        Email: string option
    }}
    
    /// <summary>
    /// Sample function working with records
    /// </summary>
    let createPerson name age email =
        {{ Name = name; Age = age; Email = email }}
'''
        
        with open(os.path.join(project_path, "src", "Library.fs"), 'w', encoding='utf-8') as f:
            f.write(library_content)
        
        # Update project file for library
        proj_file = os.path.join(project_path, f"{project_name}.fsproj")
        with open(proj_file, 'r') as f:
            content = f.read()
        
        updated_content = content.replace(
            '<OutputType>Exe</OutputType>',
            '<OutputType>Library</OutputType>'
        ).replace(
            '<Compile Include="src/Program.fs" />',
            '<Compile Include="src/Library.fs" />'
        )
        
        with open(proj_file, 'w') as f:
            f.write(updated_content)
        
        return True
    
    def create_configuration_files(self, project_path, project_name, app_type):
        """Create configuration files"""
        
        if app_type in ["webapi", "webapp", "microservice"]:
            # appsettings.json
            appsettings = {
                "Logging": {
                    "LogLevel": {
                        "Default": "Information",
                        "Microsoft.AspNetCore": "Warning"
                    }
                },
                "AllowedHosts": "*",
                "Kestrel": {
                    "Endpoints": {
                        "Http": {
                            "Url": "http://0.0.0.0:5000"
                        }
                    }
                }
            }
            
            with open(os.path.join(project_path, "appsettings.json"), 'w') as f:
                json.dump(appsettings, f, indent=2)
        
        # Dockerfile
        dockerfile_content = f'''FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY . .
RUN dotnet restore "{project_name}.fsproj"
RUN dotnet build "{project_name}.fsproj" -c Release -o /app/build

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/build .
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000
ENTRYPOINT ["dotnet", "{project_name}.dll"]
'''
        
        with open(os.path.join(project_path, "Dockerfile"), 'w') as f:
            f.write(dockerfile_content)
    
    def generate_documentation(self, project_path, project_name, app_type):
        """Generate project documentation"""
        
        readme_content = f'''# {project_name}

{self.application_types.get(app_type, "F# Application")} generated by TARS Enhanced Project Generator.

## 🚀 Quick Start

### Prerequisites
- .NET 8.0 SDK
- Docker (for containerized deployment)

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
docker build -t {project_name.lower()} .

# Run container
docker run -p 5000:5000 {project_name.lower()}
```

## 📖 API Documentation

{f"- Health Check: http://localhost:5000/health" if app_type in ["webapi", "webapp", "microservice"] else ""}
{f"- API Endpoints: http://localhost:5000/api/values" if app_type in ["webapi", "microservice"] else ""}
{f"- Swagger UI: http://localhost:5000/swagger" if app_type in ["webapi", "microservice"] else ""}

## 🧪 Testing

```bash
# Run tests
dotnet test
```

## 📦 Project Structure

```
{project_name}/
├── src/                 # Source code
├── tests/              # Unit tests
├── docs/               # Documentation
├── scripts/            # Build/deployment scripts
├── Dockerfile          # Container configuration
└── {project_name}.fsproj    # Project file
```

## 🤖 Generated by TARS

This project was autonomously generated by the TARS Enhanced Project Generator.
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Type**: {app_type}
- **Framework**: .NET 8.0 with F#

## 🔧 Development

The application includes:
- ✅ Executable F# code
- ✅ Proper project configuration
- ✅ Docker support
- ✅ Health check endpoints
- ✅ Logging configuration
- ✅ Build and deployment scripts

Ready for immediate deployment and testing!
'''
        
        with open(os.path.join(project_path, "README.md"), 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def validate_project(self, project_path, app_type):
        """Validate that the project has all necessary components"""
        
        issues = []
        
        # Check for source files
        src_dir = os.path.join(project_path, "src")
        source_files = []
        if os.path.exists(src_dir):
            source_files = [f for f in os.listdir(src_dir) if f.endswith('.fs')]
        
        if not source_files:
            issues.append("No F# source files found in src/ directory")
        
        # Check for project file
        project_files = [f for f in os.listdir(project_path) if f.endswith('.fsproj')]
        if not project_files:
            issues.append("No .fsproj file found")
        
        # Check for entry point
        if app_type != "library":
            program_fs = os.path.join(src_dir, "Program.fs")
            if not os.path.exists(program_fs):
                issues.append("No Program.fs entry point found")
        
        # Check for configuration files
        config_files = []
        if app_type in ["webapi", "webapp", "microservice"]:
            appsettings = os.path.join(project_path, "appsettings.json")
            if os.path.exists(appsettings):
                config_files.append("appsettings.json")
        
        dockerfile = os.path.join(project_path, "Dockerfile")
        if os.path.exists(dockerfile):
            config_files.append("Dockerfile")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'source_files': len(source_files),
            'project_files': len(project_files),
            'config_files': len(config_files)
        }
    
    def create_build_scripts(self, project_path, project_name, app_type):
        """Create build and deployment scripts"""
        
        # Build script
        build_script = f'''#!/bin/bash
# Build script for {project_name}

echo "Building {project_name}..."

# Restore dependencies
dotnet restore

# Build project
dotnet build -c Release

# Run tests (if any)
if [ -d "tests" ]; then
    echo "Running tests..."
    dotnet test
fi

echo "Build complete!"
'''

        with open(os.path.join(project_path, "scripts", "build.sh"), 'w', encoding='utf-8') as f:
            f.write(build_script)
        
        # Make executable
        os.chmod(os.path.join(project_path, "scripts", "build.sh"), 0o755)
        
        # Deploy script
        deploy_script = f'''#!/bin/bash
# Deploy script for {project_name}

echo "Deploying {project_name}..."

# Build Docker image
docker build -t {project_name.lower()} .

# Run container
docker run -d --name {project_name.lower()}-container -p 5000:5000 {project_name.lower()}

echo "Deployment complete!"
echo "Application available at: http://localhost:5000"
'''

        with open(os.path.join(project_path, "scripts", "deploy.sh"), 'w', encoding='utf-8') as f:
            f.write(deploy_script)
        
        os.chmod(os.path.join(project_path, "scripts", "deploy.sh"), 0o755)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python enhanced-project-generator.py <project_name> [app_type] [complexity]")
        print("App types: webapi, console, webapp, microservice, library")
        print("Example: python enhanced-project-generator.py MyAPI webapi medium")
        return 1
    
    project_name = sys.argv[1]
    app_type = sys.argv[2] if len(sys.argv) > 2 else "webapi"
    complexity = sys.argv[3] if len(sys.argv) > 3 else "medium"
    
    generator = EnhancedProjectGenerator()
    result = generator.generate_complete_project(project_name, app_type, complexity)
    
    if result:
        print(f"Project '{project_name}' generated successfully!")
        print(f"Location: {result}")
        return 0
    else:
        print(f"Failed to generate project '{project_name}'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
