# Test script for F# code analysis and metascript engine

# Analyze the example code
Write-Host "Analyzing example_code.cs..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- fsharp analyze example_code.cs

# Apply metascript rules to the example code
Write-Host "`nApplying metascript rules to example_code.cs..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- fsharp metascript example_code.cs --rules TarsEngineFSharp/improvement_rules.meta --output output/transformed

# Compare the original and transformed code
Write-Host "`nComparing original and transformed code..." -ForegroundColor Green
Write-Host "Original code:" -ForegroundColor Yellow
Get-Content example_code.cs | Select-Object -First 20
Write-Host "`nTransformed code:" -ForegroundColor Yellow
Get-Content output/transformed/example_code.cs | Select-Object -First 20
