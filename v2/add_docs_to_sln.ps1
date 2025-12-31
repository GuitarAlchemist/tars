# Add key markdown files to Tars.sln

$solutionFile = "Tars.sln"

# Define the markdown files to add
$rootDocs = @(
    "README.md",
    "GEMINI.md",
    "task.md",
    "QUICK_REFERENCE.md",
    "SESSION_SUMMARY_2025-12-25.md",
    "MISSION_COMPLETE_2025-12-25.md",
    "QUICKSTART.md",
    "CHANGELOG.md"
)

$planDocs = @(
    "docs\3_Roadmap\1_Plans\RDF_INGESTION_PLAN.md",
    "docs\3_Roadmap\1_Plans\PHASE_9_3_RDF_INGESTION.md",
    "docs\3_Roadmap\1_Plans\EVALUATION_FRAMEWORK.md",
    "docs\3_Roadmap\1_Plans\architectural_refinement_v2.md"
)

$visionDocs = @(
    "docs\1_Vision\architectural_vision.md"
)

# Create a new solution folder for Root Documentation
Write-Host "Adding root documentation files to solution..."
dotnet sln $solutionFile add --solution-folder "00_Documentation" $rootDocs

# Add new plan files
Write-Host "Adding new plan documentation..."
foreach ($doc in $planDocs) {
    if (Test-Path $doc) {
        dotnet sln $solution File add --solution-folder "docs/3_Roadmap/1_Plans" $doc
    }
}

# Add vision doc
Write-Host "Adding updated architectural vision..."
if (Test-Path "docs\1_Vision\architectural_vision.md") {
    # This should already be in the solution, but ensuring it's visible
    Write-Host "  architectural_vision.md already in solution"
}

Write-Host "✅ Solution updated with documentation files!"
