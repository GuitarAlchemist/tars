# Script to fix MudBlazor attribute casing issues
$razorFiles = Get-ChildItem -Path "TarsApp/Components/Pages" -Filter "*.razor" -Recurse

# Define the patterns to search for and their replacements
$patterns = @(
    # Bind attributes
    @{ Pattern = '@bind-clickable="'; Replacement = '@bind-Clickable="' },
    @{ Pattern = '@bind-disableGutters="'; Replacement = '@bind-DisableGutters="' },
    @{ Pattern = '@bind-isVisible="'; Replacement = '@bind-IsVisible="' },
    @{ Pattern = '@bind-isVisibleChanged="'; Replacement = '@bind-IsVisibleChanged="' },
    @{ Pattern = '@bind-selectedOption="'; Replacement = '@bind-SelectedOption="' },
    @{ Pattern = '@bind-selectedOptionChanged="'; Replacement = '@bind-SelectedOptionChanged="' },
    @{ Pattern = '@bind-option="'; Replacement = '@bind-Option="' },
    @{ Pattern = '@bind-checked="'; Replacement = '@bind-Checked="' },
    @{ Pattern = '@bind-checkedChanged="'; Replacement = '@bind-CheckedChanged="' },
    @{ Pattern = '@bind-items="'; Replacement = '@bind-Items="' },
    @{ Pattern = '@bind-filter="'; Replacement = '@bind-Filter="' },
    @{ Pattern = '@bind-mandatory="'; Replacement = '@bind-Mandatory="' },
    @{ Pattern = '@bind-multiSelection="'; Replacement = '@bind-MultiSelection="' },
    @{ Pattern = '@bind-selectedChips="'; Replacement = '@bind-SelectedChips="' },
    @{ Pattern = '@bind-selectedChipsChanged="'; Replacement = '@bind-SelectedChipsChanged="' },
    @{ Pattern = '@bind-type="'; Replacement = '@bind-Type="' },
    
    # Regular attributes
    @{ Pattern = ' clickable="'; Replacement = ' Clickable="' },
    @{ Pattern = ' disableGutters="'; Replacement = ' DisableGutters="' },
    @{ Pattern = ' isVisible="'; Replacement = ' IsVisible="' },
    @{ Pattern = ' isVisibleChanged="'; Replacement = ' IsVisibleChanged="' },
    @{ Pattern = ' selectedOption="'; Replacement = ' SelectedOption="' },
    @{ Pattern = ' selectedOptionChanged="'; Replacement = ' SelectedOptionChanged="' },
    @{ Pattern = ' option="'; Replacement = ' Option="' },
    @{ Pattern = ' checked="'; Replacement = ' Checked="' },
    @{ Pattern = ' checkedChanged="'; Replacement = ' CheckedChanged="' },
    @{ Pattern = ' items="'; Replacement = ' Items="' },
    @{ Pattern = ' filter="'; Replacement = ' Filter="' },
    @{ Pattern = ' mandatory="'; Replacement = ' Mandatory="' },
    @{ Pattern = ' multiSelection="'; Replacement = ' MultiSelection="' },
    @{ Pattern = ' selectedChips="'; Replacement = ' SelectedChips="' },
    @{ Pattern = ' selectedChipsChanged="'; Replacement = ' SelectedChipsChanged="' },
    @{ Pattern = ' type="'; Replacement = ' Type="' }
)

foreach ($file in $razorFiles) {
    $content = Get-Content -Path $file.FullName -Raw
    $modified = $false
    
    foreach ($pattern in $patterns) {
        if ($content -match $pattern.Pattern) {
            $content = $content -replace $pattern.Pattern, $pattern.Replacement
            $modified = $true
        }
    }
    
    if ($modified) {
        Set-Content -Path $file.FullName -Value $content
        Write-Host "Fixed attributes in $($file.Name)"
    }
}

Write-Host "Attribute fixing complete!"
