# Create directories if they don't exist
$directories = @("Core", "Analysis", "Services", "DSL")
foreach ($dir in $directories) {
    $path = "TarsEngineFSharp\$dir"
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force
        Write-Host "Created directory: $path"
    }
}

# Move Core files
$coreFiles = @("Option.fs", "Result.fs", "ModelProvider.fs", "AsyncExecution.fs", "DataProcessing.fs", "DataSources.fs")
foreach ($file in $coreFiles) {
    $source = "TarsEngineFSharp\$file"
    $destination = "TarsEngineFSharp\Core\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $destination -Force
        Write-Host "Copied $source to $destination"
    } else {
        Write-Host "Warning: Source file $source not found"
    }
}

# Move Agent files
$agentFiles = @("RetroactionAnalysis.fs", "SampleAgents.fs")
foreach ($file in $agentFiles) {
    $source = "TarsEngineFSharp\$file"
    $destination = "TarsEngineFSharp\Agents\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $destination -Force
        Write-Host "Copied $source to $destination"
    } else {
        Write-Host "Warning: Source file $source not found"
    }
}

# Move Analysis files
$analysisFiles = @("CodeAnalysis.fs", "EnhancedCodeAnalysis.fs")
foreach ($file in $analysisFiles) {
    $source = "TarsEngineFSharp\$file"
    $destination = "TarsEngineFSharp\Analysis\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $destination -Force
        Write-Host "Copied $source to $destination"
    } else {
        Write-Host "Warning: Source file $source not found"
    }
}

# Move DSL files
$dslFiles = @("MetascriptEngine.fs", "PromptEngine.fs", "TarsDsl.fs", "Examples.fs", "TarsBuilder.fs", "TypeProviderPatternMatching.fs")
foreach ($file in $dslFiles) {
    $source = "TarsEngineFSharp\$file"
    $destination = "TarsEngineFSharp\DSL\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $destination -Force
        Write-Host "Copied $source to $destination"
    } else {
        Write-Host "Warning: Source file $source not found"
    }
}

# Move Service files
$serviceFiles = @("APIDataFetcher.fs", "WebSearch.fs", "ChatBotService.fs", "ChatService.fs", "RivaService.fs", "WeatherService.fs", "LlmService.fs")
foreach ($file in $serviceFiles) {
    $source = "TarsEngineFSharp\$file"
    $destination = "TarsEngineFSharp\Services\$file"
    if (Test-Path $source) {
        Copy-Item -Path $source -Destination $destination -Force
        Write-Host "Copied $source to $destination"
    } else {
        Write-Host "Warning: Source file $source not found"
    }
}

Write-Host "All files have been copied to their new locations."
Write-Host "Please review the files and then rename TarsEngineFSharp.fsproj.new to TarsEngineFSharp.fsproj"
