# Demo script for multi-agent collaboration

# Step 1: List registered agents
Write-Host "Step 1: Listing registered agents..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction agent list

# Step 2: Analyze the test file with multiple agents
Write-Host "`nStep 2: Analyzing the test file with multiple agents..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction agent analyze TarsCli/Examples/TestCalculator.cs

# Step 3: Transform the test file with multiple agents
Write-Host "`nStep 3: Transforming the test file with multiple agents..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction agent transform TarsCli/Examples/TestCalculator.cs --output TarsCli/Examples/TestCalculator_MultiAgent.cs

# Step 4: Learn from the transformation
Write-Host "`nStep 4: Learning from the transformation..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction agent learn TarsCli/Examples/TestCalculator.cs TarsCli/Examples/TestCalculator_MultiAgent.cs --accepted true

# Step 5: Show the differences between the original and improved files
Write-Host "`nStep 5: Showing differences between original and improved files..." -ForegroundColor Yellow
Write-Host "Original file:" -ForegroundColor Yellow
Get-Content TarsCli/Examples/TestCalculator.cs | Select-Object -First 10
Write-Host "`nImproved file:" -ForegroundColor Yellow
Get-Content TarsCli/Examples/TestCalculator_MultiAgent.cs | Select-Object -First 10
