# Script to fix MudBlazor attribute casing issues
$razorFiles = Get-ChildItem -Path "TarsApp/Components/Pages" -Filter "*.razor" -Recurse

$attributeMappings = @{
    "clickable" = "Clickable"
    "disableGutters" = "DisableGutters"
    "isVisible" = "IsVisible"
    "isVisibleChanged" = "IsVisibleChanged"
    "selectedOption" = "SelectedOption"
    "selectedOptionChanged" = "SelectedOptionChanged"
    "option" = "Option"
    "checked" = "Checked"
    "checkedChanged" = "CheckedChanged"
    "items" = "Items"
    "filter" = "Filter"
    "mandatory" = "Mandatory"
    "multiSelection" = "MultiSelection"
    "selectedChips" = "SelectedChips"
    "selectedChipsChanged" = "SelectedChipsChanged"
    "type" = "Type"
}

foreach ($file in $razorFiles) {
    $content = Get-Content -Path $file.FullName -Raw
    $modified = $false
    
    foreach ($key in $attributeMappings.Keys) {
        $value = $attributeMappings[$key]
        # Replace the attribute in various contexts (as a standalone attribute or in bind expressions)
        if ($content -match $key) {
            $content = $content -replace "(?<=\s)$key(?=\s*=)", $value
            $content = $content -replace "(?<=@bind-)$key(?=\s*=)", $value
            $content = $content -replace "(?<=@bind-)$key(?=\s*@)", $value
            $content = $content -replace "(?<=@bind-)$key(?=\s*>)", $value
            $modified = $true
        }
    }
    
    if ($modified) {
        Set-Content -Path $file.FullName -Value $content
        Write-Host "Fixed attributes in $($file.Name)"
    }
}

Write-Host "Attribute fixing complete!"
