# PowerShell script to update TARS metascripts from old format to new format
# This script converts .tars files to .trsx and updates the format

param(
    [string]$SourcePath = ".tars",
    [switch]$DryRun = $false
)

Write-Host "🔄 TARS Metascript Format Updater" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Find all .tars files
$tarsFiles = Get-ChildItem -Path $SourcePath -Recurse -Filter "*.tars" | Where-Object { $_.Name -notlike "*template*" }

Write-Host "📁 Found $($tarsFiles.Count) .tars files to process" -ForegroundColor Yellow

foreach ($file in $tarsFiles) {
    Write-Host "`n🔍 Processing: $($file.FullName)" -ForegroundColor Green
    
    try {
        # Read the file content
        $content = Get-Content -Path $file.FullName -Raw
        
        # Check if it's already in new format (has FSHARP blocks)
        if ($content -match "FSHARP\s*\{") {
            Write-Host "  ✅ Already in new format, just renaming to .trsx" -ForegroundColor Green
            
            if (-not $DryRun) {
                $newPath = $file.FullName -replace "\.tars$", ".trsx"
                Move-Item -Path $file.FullName -Destination $newPath
                Write-Host "  📝 Renamed to: $newPath" -ForegroundColor Blue
            }
            continue
        }
        
        # Check if it uses old format patterns
        $needsUpdate = $false
        $oldPatterns = @(
            "REASONING\s*\{",
            "DESIGN\s*\{", 
            "GENERATE\s*\{",
            "EXECUTE\s*\{",
            "VALIDATION\s*\{",
            "REFLECTION\s*\{",
            "AUTONOMOUS_ANALYSIS\s*\{",
            "AUTONOMOUS_CODING\s*\{"
        )
        
        foreach ($pattern in $oldPatterns) {
            if ($content -match $pattern) {
                $needsUpdate = $true
                break
            }
        }
        
        if ($needsUpdate) {
            Write-Host "  ⚠️  Uses old format patterns, needs manual review" -ForegroundColor Yellow
            Write-Host "  📋 File contains old-style blocks that need F# conversion" -ForegroundColor Yellow
            
            # For now, just rename to .trsx and add a comment
            if (-not $DryRun) {
                $newPath = $file.FullName -replace "\.tars$", ".trsx"
                
                # Add a comment at the top indicating it needs format update
                $updatedContent = "// TODO: Convert old format blocks to FSHARP blocks`n" + $content
                Set-Content -Path $file.FullName -Value $updatedContent
                
                Move-Item -Path $file.FullName -Destination $newPath
                Write-Host "  📝 Renamed to .trsx with TODO comment: $newPath" -ForegroundColor Blue
            }
        } else {
            Write-Host "  ✅ Standard format, just renaming to .trsx" -ForegroundColor Green
            
            if (-not $DryRun) {
                $newPath = $file.FullName -replace "\.tars$", ".trsx"
                Move-Item -Path $file.FullName -Destination $newPath
                Write-Host "  📝 Renamed to: $newPath" -ForegroundColor Blue
            }
        }
        
    } catch {
        Write-Host "  ❌ Error processing file: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n✅ Metascript format update completed!" -ForegroundColor Green
Write-Host "📊 Processed $($tarsFiles.Count) files" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n🔍 This was a dry run. Use -DryRun:`$false to actually update files." -ForegroundColor Yellow
}

# Summary of what needs to be done manually
Write-Host "`n📋 Manual Steps Required:" -ForegroundColor Cyan
Write-Host "1. Review files marked with TODO comments" -ForegroundColor White
Write-Host "2. Convert old format blocks (REASONING, DESIGN, etc.) to FSHARP blocks" -ForegroundColor White
Write-Host "3. Update GENERATE blocks to use F# file writing" -ForegroundColor White
Write-Host "4. Convert EXECUTE blocks to ACTION + FSHARP blocks" -ForegroundColor White
Write-Host "5. Test metascript execution with TARS CLI" -ForegroundColor White
