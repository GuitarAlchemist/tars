# Script to migrate TarsCli code from the Rescue solution to the target solution
$sourceRoot = "C:\Users\spare\source\repos\tars\Rescue\tars"
$targetRoot = "C:\Users\spare\source\repos\tars"

# Define the mapping of source directories/namespaces to target projects
$mappings = @(
    @{
        SourcePath = "$sourceRoot\TarsCli\Commands";
        TargetPath = "$targetRoot\TarsCli.Commands";
        Description = "Command classes for CLI commands";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Models";
        TargetPath = "$targetRoot\TarsCli.Models";
        Description = "Data models and DTOs";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Services";
        Description = "General services";
        ExcludePaths = @(
            "*Docker*",
            "*DSL*",
            "*CodeAnalysis*",
            "*Mcp*",
            "*WebUI*",
            "*Testing*"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.CodeAnalysis";
        Description = "Code analysis services";
        IncludePaths = @(
            "*CodeAnalysis*",
            "*AiCodeUnderstanding*",
            "*Compilation*"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Mcp";
        Description = "MCP-related services";
        IncludePaths = @(
            "*Mcp*"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Mcp";
        TargetPath = "$targetRoot\TarsCli.Mcp";
        Description = "MCP controllers and models";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Docker";
        Description = "Docker-related services";
        IncludePaths = @(
            "*Docker*"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.DSL";
        Description = "DSL-related services";
        IncludePaths = @(
            "*DSL*"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Testing";
        Description = "Testing-related services";
        IncludePaths = @(
            "*Testing*",
            "*Test*Runner*"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Controllers";
        TargetPath = "$targetRoot\TarsCli.WebUI";
        Description = "Web UI controllers";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\WebUI";
        TargetPath = "$targetRoot\TarsCli.WebUI";
        Description = "Web UI components";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Common";
        TargetPath = "$targetRoot\TarsCli.Core";
        Description = "Common utilities";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Constants";
        TargetPath = "$targetRoot\TarsCli.Core";
        Description = "Constants and configuration keys";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Extensions";
        TargetPath = "$targetRoot\TarsCli.Core";
        Description = "Extension methods";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Parsing";
        TargetPath = "$targetRoot\TarsCli.Core";
        Description = "Parsing utilities";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Examples";
        TargetPath = "$targetRoot\TarsCli.App";
        Description = "Example code";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Prompts";
        TargetPath = "$targetRoot\TarsCli.Core\Prompts";
        Description = "Prompt templates";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Scripts";
        TargetPath = "$targetRoot\TarsCli.Core\Scripts";
        Description = "Script files";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Python";
        TargetPath = "$targetRoot\TarsCli.Core\Python";
        Description = "Python scripts";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\wwwroot";
        TargetPath = "$targetRoot\TarsCli.WebUI\wwwroot";
        Description = "Web assets";
    }
)

# Process each mapping
foreach ($mapping in $mappings) {
    $sourcePath = $mapping.SourcePath
    $targetPath = $mapping.TargetPath
    $description = $mapping.Description
    
    if (-not (Test-Path $sourcePath)) {
        Write-Host "Source path not found: $sourcePath" -ForegroundColor Yellow
        continue
    }
    
    # Create target directory if it doesn't exist
    if (-not (Test-Path $targetPath)) {
        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        Write-Host "Created directory: $targetPath" -ForegroundColor Green
    }
    
    Write-Host "Migrating $description from $sourcePath to $targetPath..." -ForegroundColor Cyan
    
    # Get all .cs files in the source path
    $files = Get-ChildItem -Path $sourcePath -Recurse -File -Include "*.cs", "*.json", "*.md", "*.txt", "*.py", "*.js", "*.html", "*.css" -ErrorAction SilentlyContinue
    
    $copiedCount = 0
    
    foreach ($file in $files) {
        $shouldCopy = $true
        
        # Check exclude paths
        if ($mapping.ContainsKey("ExcludePaths")) {
            foreach ($excludePath in $mapping.ExcludePaths) {
                if ($file.FullName -like $excludePath) {
                    $shouldCopy = $false
                    break
                }
            }
        }
        
        # Check include paths
        if ($mapping.ContainsKey("IncludePaths")) {
            $shouldCopy = $false
            foreach ($includePath in $mapping.IncludePaths) {
                if ($file.FullName -like $includePath) {
                    $shouldCopy = $true
                    break
                }
            }
        }
        
        if ($shouldCopy) {
            $relativePath = $file.FullName.Substring($sourcePath.Length)
            $destination = Join-Path $targetPath $relativePath
            
            # Create destination directory if it doesn't exist
            $destinationFolder = Split-Path -Path $destination -Parent
            if (-not (Test-Path $destinationFolder)) {
                New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
            }
            
            # Copy the file
            Copy-Item -Path $file.FullName -Destination $destination -Force
            $copiedCount++
        }
    }
    
    Write-Host "  Copied $copiedCount files" -ForegroundColor Green
}

# Update project references
$projectFiles = Get-ChildItem -Path $targetRoot -Recurse -Filter "*.csproj" -ErrorAction SilentlyContinue

foreach ($projectFile in $projectFiles) {
    $projectContent = Get-Content -Path $projectFile.FullName -Raw
    
    # Check if it's a TarsCli project
    if ($projectFile.Name -like "TarsCli.*") {
        # Add reference to TarsCli.Core if it doesn't exist and it's not the Core project itself
        if (-not ($projectFile.Name -eq "TarsCli.Core.csproj") -and -not ($projectContent -match "TarsCli.Core")) {
            $projectContent = $projectContent -replace "</Project>", "  <ItemGroup>
    <ProjectReference Include="..\TarsCli.Core\TarsCli.Core.csproj" />
  </ItemGroup>
</Project>"
            
            Set-Content -Path $projectFile.FullName -Value $projectContent
            Write-Host "Added TarsCli.Core reference to $($projectFile.Name)" -ForegroundColor Green
        }
    }
}

Write-Host "Migration completed successfully!" -ForegroundColor Green
