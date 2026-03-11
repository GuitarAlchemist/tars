<#
.SYNOPSIS
    Downloads GGUF models for TARS using the Python model manager.

.DESCRIPTION
    Wrapper for manage_models.py. Requires the .venv to be set up with huggingface_hub[hf_transfer].
    
.EXAMPLE
    .\download-models.ps1 --list
    .\download-models.ps1 --download qwen-14b
    .\download-models.ps1 --repo "TheBloke/Mistral-7B-v0.1-GGUF" --file "mistral-7b-v0.1.Q4_K_M.gguf"
#>

param(
    [string]$Download,
    [string]$Repo,
    [string]$File,
    [string]$Out = "models",
    [switch]$List
)

$VenvPython = "$PSScriptRoot\..\.venv\Scripts\python.exe"
$ScriptPath = "$PSScriptRoot\manage_models.py"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Python virtual environment not found at $VenvPython. Please run setup-tars.ps1 or create the venv manually."
    exit 1
}

$ArgsList = @()
if ($List) { $ArgsList += "--list" }
if ($Download) { $ArgsList += "--download", $Download }
if ($Repo) { $ArgsList += "--repo", $Repo }
if ($File) { $ArgsList += "--file", $File }
if ($Out) { $ArgsList += "--out", $Out }

& $VenvPython $ScriptPath @ArgsList
