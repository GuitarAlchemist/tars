# Script to install F# compiler packages

Write-Host "Installing F# compiler packages..."

# Create a temporary project to install the packages
$tempDir = Join-Path $env:TEMP "FSharpCompilerInstall"
if (Test-Path $tempDir) {
    Remove-Item -Path $tempDir -Recurse -Force
}
New-Item -Path $tempDir -ItemType Directory | Out-Null

# Create a temporary project file
$projectFile = Join-Path $tempDir "FSharpCompilerInstall.csproj"
@"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="FSharp.Compiler.Service" Version="43.7.300" />
    <PackageReference Include="FSharp.Core" Version="7.0.300" />
  </ItemGroup>
</Project>
"@ | Out-File -FilePath $projectFile -Encoding utf8

# Restore the packages
Write-Host "Restoring packages..."
dotnet restore $projectFile

Write-Host "F# compiler packages installed successfully."
