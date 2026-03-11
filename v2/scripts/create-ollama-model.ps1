<#
.SYNOPSIS
    Creates a custom Ollama model from a GGUF file.

.DESCRIPTION
    Generates a temporary Modelfile using the specified GGUF and imports it into Ollama.

.EXAMPLE
    .\create-ollama-model.ps1 -Name "glm-4.7-udtq1" -Path "C:\models\glm47\GLM-4.7-UD-TQ1_0.gguf"
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$Path,

    [string]$Temperature = "0"
)

if (-not (Test-Path $Path)) {
    Write-Error "GGUF file not found at $Path"
    exit 1
}

# Resolve absolute path for Ollama (it doesn't always like relative paths in Modelfile)
$AbsPath = (Resolve-Path $Path).Path
# Escape backslashes for Modelfile
$AbsPath = $AbsPath -replace "\\", "\\"

$ModelFileContent = @"
FROM "$AbsPath"
PARAMETER temperature $Temperature
"@

$ModelFilePath = "Modelfile.$Name"
$ModelFileContent | Out-File -Encoding utf8 -FilePath $ModelFilePath

Write-Host "Creating Ollama model '$Name' from '$AbsPath'..."
ollama create $Name -f $ModelFilePath

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Model '$Name' created successfully."
    Remove-Item $ModelFilePath
}
else {
    Write-Error "❌ Failed to create model."
}
