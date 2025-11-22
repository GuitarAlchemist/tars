# Demo script for retroaction coding

# Step 1: Generate F# code from metascript rules
Write-Host "Step 1: Generating F# code from metascript rules..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction generate TarsCli/Examples/basic_transformations.meta --output TarsCli/Examples/Transformations.fs

# Step 2: Compile the generated F# code
Write-Host "`nStep 2: Compiling the generated F# code..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction compile TarsCli/Examples/Transformations.fs --output DynamicTransformations

# Step 3: Preview transformations on the test file
Write-Host "`nStep 3: Previewing transformations on the test file..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction transform TarsCli/Examples/TestCalculator.cs --rules TarsCli/Examples/basic_transformations.meta --preview

# Step 4: Apply transformations and save to a new file
Write-Host "`nStep 4: Applying transformations and saving to a new file..." -ForegroundColor Green
dotnet run --project TarsCli/TarsCli.csproj -- retroaction transform TarsCli/Examples/TestCalculator.cs --rules TarsCli/Examples/basic_transformations.meta --output TarsCli/Examples/TestCalculator_Improved.cs

# Step 5: Show the differences between the original and improved files
Write-Host "`nStep 5: Showing differences between original and improved files..." -ForegroundColor Green
Write-Host "Original file:" -ForegroundColor Yellow
Get-Content TarsCli/Examples/TestCalculator.cs | Select-Object -First 10
Write-Host "`nImproved file:" -ForegroundColor Yellow
Get-Content TarsCli/Examples/TestCalculator_Improved.cs | Select-Object -First 10
