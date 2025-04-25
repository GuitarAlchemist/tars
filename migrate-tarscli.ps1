# Script to migrate TarsCli from monolith to feature-based projects
$sourceRoot = "C:\Users\spare\source\repos\tars\Rescue\tars"
$targetRoot = "C:\Users\spare\source\repos\tars"

# Create backup directory
$backupDir = "$targetRoot\Backups\$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Backup the target solution file
Copy-Item -Path "$targetRoot\tars.sln" -Destination "$backupDir\tars.sln" -Force

# Define the mapping of source directories/namespaces to target projects
$mappings = @(
    @{
        SourcePath = "$sourceRoot\TarsCli\Commands";
        TargetPath = "$targetRoot\TarsCli.Commands";
        Namespace = "TarsCli.Commands";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Models";
        TargetPath = "$targetRoot\TarsCli.Models";
        Namespace = "TarsCli.Models";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Services";
        Namespace = "TarsCli.Services";
        ExcludeNamespaces = @(
            "TarsCli.Services.CodeAnalysis",
            "TarsCli.Services.Mcp",
            "TarsCli.Services.Docker",
            "TarsCli.Services.DSL",
            "TarsCli.Services.Testing",
            "TarsCli.Services.WebUI"
        );
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.CodeAnalysis";
        Namespace = "TarsCli.Services.CodeAnalysis";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Mcp";
        Namespace = "TarsCli.Services.Mcp";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Mcp";
        TargetPath = "$targetRoot\TarsCli.Mcp";
        Namespace = "TarsCli.Mcp";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Docker";
        NamespacePattern = "TarsCli.Services.*Docker.*";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.DSL";
        NamespacePattern = "TarsCli.Services.*DSL.*";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Services";
        TargetPath = "$targetRoot\TarsCli.Testing";
        Namespace = "TarsCli.Services.Testing";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Controllers";
        TargetPath = "$targetRoot\TarsCli.WebUI";
        Namespace = "TarsCli.Controllers";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Common";
        TargetPath = "$targetRoot\TarsCli.Core";
        Namespace = "TarsCli.Common";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Constants";
        TargetPath = "$targetRoot\TarsCli.Core";
        Namespace = "TarsCli.Constants";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Extensions";
        TargetPath = "$targetRoot\TarsCli.Core";
        Namespace = "TarsCli.Extensions";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Parsing";
        TargetPath = "$targetRoot\TarsCli.Core";
        Namespace = "TarsCli.Parsing";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Examples";
        TargetPath = "$targetRoot\TarsCli.App";
        Namespace = "TarsCli.Examples";
    }
)

# Process each mapping
foreach ($mapping in $mappings) {
    $sourcePath = $mapping.SourcePath
    $targetPath = $mapping.TargetPath
    
    # Create target directory if it doesn't exist
    if (-not (Test-Path $targetPath)) {
        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        Write-Host "Created directory: $targetPath"
    }
    
    # Copy files based on namespace
    if ($mapping.ContainsKey("Namespace")) {
        $namespace = $mapping.Namespace
        Write-Host "Copying files from namespace $namespace to $targetPath..."
        
        # Find all .cs files in the source path
        Get-ChildItem -Path $sourcePath -Recurse -Filter "*.cs" | ForEach-Object {
            $content = Get-Content -Path $_.FullName -Raw
            
            # Check if the file belongs to the specified namespace
            if ($content -match "namespace\s+$namespace") {
                $relativePath = $_.FullName.Substring($sourcePath.Length)
                $destination = Join-Path $targetPath $relativePath
                
                # Create destination directory if it doesn't exist
                $destinationFolder = Split-Path -Path $destination -Parent
                if (-not (Test-Path $destinationFolder)) {
                    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
                }
                
                # Copy the file
                Copy-Item -Path $_.FullName -Destination $destination -Force
                Write-Host "  Copied $($_.Name) to $destination"
            }
        }
    }
    elseif ($mapping.ContainsKey("NamespacePattern")) {
        $pattern = $mapping.NamespacePattern
        Write-Host "Copying files matching namespace pattern $pattern to $targetPath..."
        
        # Find all .cs files in the source path
        Get-ChildItem -Path $sourcePath -Recurse -Filter "*.cs" | ForEach-Object {
            $content = Get-Content -Path $_.FullName -Raw
            
            # Check if the file belongs to the specified namespace pattern
            if ($content -match "namespace\s+($pattern)") {
                $relativePath = $_.FullName.Substring($sourcePath.Length)
                $destination = Join-Path $targetPath $relativePath
                
                # Create destination directory if it doesn't exist
                $destinationFolder = Split-Path -Path $destination -Parent
                if (-not (Test-Path $destinationFolder)) {
                    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
                }
                
                # Copy the file
                Copy-Item -Path $_.FullName -Destination $destination -Force
                Write-Host "  Copied $($_.Name) to $destination"
            }
        }
    }
    else {
        # Copy all files from source to target
        Write-Host "Copying all files from $sourcePath to $targetPath..."
        
        Get-ChildItem -Path $sourcePath -Recurse -Exclude "bin", "obj" | ForEach-Object {
            $relativePath = $_.FullName.Substring($sourcePath.Length)
            $destination = Join-Path $targetPath $relativePath
            
            if ($_.PSIsContainer) {
                if (-not (Test-Path $destination)) {
                    New-Item -ItemType Directory -Path $destination -Force | Out-Null
                }
            } else {
                $destinationFolder = Split-Path -Path $destination -Parent
                if (-not (Test-Path $destinationFolder)) {
                    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $destination -Force
                Write-Host "  Copied $($_.Name) to $destination"
            }
        }
    }
}

# Copy project files for each target project
$projectFiles = @(
    @{
        SourcePath = "$sourceRoot\TarsCli\TarsCli.csproj";
        TargetPath = "$targetRoot\TarsCli\TarsCli.csproj";
    }
)

foreach ($projectFile in $projectFiles) {
    $sourcePath = $projectFile.SourcePath
    $targetPath = $projectFile.TargetPath
    
    if (Test-Path $sourcePath) {
        # Create destination directory if it doesn't exist
        $destinationFolder = Split-Path -Path $targetPath -Parent
        if (-not (Test-Path $destinationFolder)) {
            New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
        }
        
        # Copy the project file
        Copy-Item -Path $sourcePath -Destination $targetPath -Force
        Write-Host "Copied project file from $sourcePath to $targetPath"
    } else {
        Write-Host "Warning: Source project file $sourcePath not found"
    }
}

# Create project files for each target project if they don't exist
$targetProjects = @(
    "TarsCli.Commands",
    "TarsCli.Models",
    "TarsCli.Services",
    "TarsCli.CodeAnalysis",
    "TarsCli.Mcp",
    "TarsCli.Docker",
    "TarsCli.DSL",
    "TarsCli.Testing",
    "TarsCli.WebUI",
    "TarsCli.Core",
    "TarsCli.App"
)

foreach ($project in $targetProjects) {
    $projectPath = "$targetRoot\$project\$project.csproj"
    
    if (-not (Test-Path $projectPath)) {
        # Create a basic project file
        $projectContent = @"
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\TarsCli.Core\TarsCli.Core.csproj" Condition="'$(ProjectName)' != 'TarsCli.Core'" />
  </ItemGroup>

</Project>
"@
        
        # Create destination directory if it doesn't exist
        $destinationFolder = Split-Path -Path $projectPath -Parent
        if (-not (Test-Path $destinationFolder)) {
            New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
        }
        
        # Save the project file
        Set-Content -Path $projectPath -Value $projectContent
        Write-Host "Created project file: $projectPath"
    }
}

# Copy additional resources (Prompts, Scripts, etc.)
$additionalResources = @(
    @{
        SourcePath = "$sourceRoot\TarsCli\Prompts";
        TargetPath = "$targetRoot\TarsCli.Core\Prompts";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Scripts";
        TargetPath = "$targetRoot\TarsCli.Core\Scripts";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\Python";
        TargetPath = "$targetRoot\TarsCli.Core\Python";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\WebUI";
        TargetPath = "$targetRoot\TarsCli.WebUI\WebUI";
    },
    @{
        SourcePath = "$sourceRoot\TarsCli\wwwroot";
        TargetPath = "$targetRoot\TarsCli.WebUI\wwwroot";
    }
)

foreach ($resource in $additionalResources) {
    $sourcePath = $resource.SourcePath
    $targetPath = $resource.TargetPath
    
    if (Test-Path $sourcePath) {
        # Create destination directory if it doesn't exist
        if (-not (Test-Path $targetPath)) {
            New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
        }
        
        # Copy all files
        Get-ChildItem -Path $sourcePath -Recurse | ForEach-Object {
            $relativePath = $_.FullName.Substring($sourcePath.Length)
            $destination = Join-Path $targetPath $relativePath
            
            if ($_.PSIsContainer) {
                if (-not (Test-Path $destination)) {
                    New-Item -ItemType Directory -Path $destination -Force | Out-Null
                }
            } else {
                $destinationFolder = Split-Path -Path $destination -Parent
                if (-not (Test-Path $destinationFolder)) {
                    New-Item -ItemType Directory -Path $destinationFolder -Force | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $destination -Force
            }
        }
        
        Write-Host "Copied additional resources from $sourcePath to $targetPath"
    }
}

Write-Host "Migration complete. Please check the target projects to ensure all files were copied correctly."
