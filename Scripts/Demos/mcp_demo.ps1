# MCP Integration Demo Script
Write-Host "TARS CLI MCP Integration Demo" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

# Create a test directory
$testDir = "mcp_test_$(Get-Random)"
Write-Host "Creating test directory: $testDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Path $testDir | Out-Null

try {
    # Step 1: Use MCP to execute a command
    Write-Host "Step 1: Using MCP to execute a command without permission..." -ForegroundColor Green
    dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "echo Hello from MCP!"
    Write-Host ""

    # Step 2: Use MCP to generate a simple text file
    Write-Host "Step 2: Using MCP to generate a simple text file..." -ForegroundColor Green
    $textFilePath = Join-Path $testDir "hello.txt"
    dotnet run --project TarsCli/TarsCli.csproj -- mcp code $textFilePath "Hello, World from MCP!"
    
    # Verify the file was created
    if (Test-Path $textFilePath) {
        Write-Host "File created successfully: $textFilePath" -ForegroundColor Green
        Write-Host "Content: $(Get-Content $textFilePath)" -ForegroundColor Gray
    } else {
        Write-Host "Failed to create file: $textFilePath" -ForegroundColor Red
    }
    Write-Host ""

    # Step 3: Use MCP with triple-quoted syntax to generate a C# file
    Write-Host "Step 3: Using MCP with triple-quoted syntax to generate a C# file..." -ForegroundColor Green
    $csharpFilePath = Join-Path $testDir "Program.cs"
    
    # The triple-quoted content
    $csharpCode = @"
using System;

namespace McpDemo
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello from MCP-generated code!");
            Console.WriteLine("Current time: " + DateTime.Now);
        }
    }
}
"@
    
    # Save the code to a temporary file to use as input
    $tempFile = Join-Path $testDir "temp_code.txt"
    Set-Content -Path $tempFile -Value $csharpCode
    
    # Use the MCP to generate the C# file
    dotnet run --project TarsCli/TarsCli.csproj -- mcp code $csharpFilePath "@$tempFile"
    
    # Verify the file was created
    if (Test-Path $csharpFilePath) {
        Write-Host "C# file created successfully: $csharpFilePath" -ForegroundColor Green
        Write-Host "Content preview:" -ForegroundColor Gray
        Get-Content $csharpFilePath | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    } else {
        Write-Host "Failed to create C# file: $csharpFilePath" -ForegroundColor Red
    }
    Write-Host ""
    
    # Step 4: Compile and run the generated C# code
    Write-Host "Step 4: Compiling and running the generated C# code..." -ForegroundColor Green
    $projectDir = Join-Path $testDir "McpDemo"
    
    # Create a project directory
    New-Item -ItemType Directory -Path $projectDir | Out-Null
    
    # Move the C# file to the project directory
    Move-Item -Path $csharpFilePath -Destination (Join-Path $projectDir "Program.cs")
    
    # Use MCP to create a project file
    $csprojPath = Join-Path $projectDir "McpDemo.csproj"
    $csprojContent = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
</Project>
"@
    
    # Save the project file content to a temporary file
    $tempProjFile = Join-Path $testDir "temp_proj.txt"
    Set-Content -Path $tempProjFile -Value $csprojContent
    
    # Use the MCP to generate the project file
    dotnet run --project TarsCli/TarsCli.csproj -- mcp code $csprojPath "@$tempProjFile"
    
    # Use MCP to build and run the project
    Write-Host "Building the project..." -ForegroundColor Yellow
    dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "dotnet build $csprojPath"
    
    Write-Host "Running the project..." -ForegroundColor Yellow
    dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "dotnet run --project $csprojPath"
    
    Write-Host ""
    Write-Host "MCP Integration Demo completed successfully!" -ForegroundColor Cyan
}
finally {
    # Clean up
    Write-Host "Cleaning up test directory..." -ForegroundColor Yellow
    Remove-Item -Path $testDir -Recurse -Force -ErrorAction SilentlyContinue
}
