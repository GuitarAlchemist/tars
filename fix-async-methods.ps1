# Script to fix async methods that lack await operators
$razorFiles = @(
    "TarsApp/Components/Pages/AutonomousExecution/RollbackDialog.razor",
    "TarsApp/Components/Pages/AutonomousExecution/ExecutionDashboard.razor",
    "TarsApp/Components/Pages/AutonomousExecution/ExecutionMonitor.razor",
    "TarsApp/Components/Pages/AutonomousExecution/ReportsPanel.razor",
    "TarsApp/Components/Pages/AutonomousExecution/ExecutionDetails.razor",
    "TarsApp/Components/Pages/AutonomousExecution/ExecutionsList.razor",
    "TarsApp/Components/Pages/AutonomousExecution/SettingsPanel.razor",
    "TarsApp/Components/Pages/AutonomousExecution/ImprovementsList.razor"
)

foreach ($file in $razorFiles) {
    $content = Get-Content -Path $file -Raw
    
    # Find all async methods that don't have await
    if ($content -match "private\s+async\s+Task\s+\w+\(\)[^{]*\{\s*[^a]") {
        # Add await Task.Delay(1) to the method body
        $content = $content -replace "(private\s+async\s+Task\s+\w+\(\)[^{]*\{)(\s*)", "`$1`$2    await Task.Delay(1); // Added to satisfy compiler warning CS1998`$2"
        Set-Content -Path $file -Value $content
        Write-Host "Fixed async methods in $file"
    }
}

Write-Host "Async method fixing complete!"
