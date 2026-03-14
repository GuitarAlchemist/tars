# PowerShell Script to Configure Mark Text as Default Markdown Viewer
# Run this script as Administrator

Write-Host "🔧 CONFIGURING MARK TEXT AS DEFAULT MARKDOWN VIEWER" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

# Check if running as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "❌ This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host "   Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📋 To run as Administrator:" -ForegroundColor Cyan
    Write-Host "   1. Right-click on PowerShell" -ForegroundColor White
    Write-Host "   2. Select 'Run as Administrator'" -ForegroundColor White
    Write-Host "   3. Navigate to: $(Get-Location)" -ForegroundColor White
    Write-Host "   4. Run: .\setup_markdown_file_associations.ps1" -ForegroundColor White
    pause
    exit 1
}

Write-Host "✅ Running with Administrator privileges" -ForegroundColor Green
Write-Host ""

# Find Mark Text installation
$markTextPaths = @(
    "${env:LOCALAPPDATA}\Programs\marktext\Mark Text.exe",
    "${env:ProgramFiles}\Mark Text\Mark Text.exe",
    "${env:ProgramFiles(x86)}\Mark Text\Mark Text.exe",
    "${env:USERPROFILE}\AppData\Local\Programs\marktext\Mark Text.exe"
)

$markTextPath = $null
foreach ($path in $markTextPaths) {
    if (Test-Path $path) {
        $markTextPath = $path
        break
    }
}

if (-not $markTextPath) {
    Write-Host "❌ Mark Text not found!" -ForegroundColor Red
    Write-Host "   Please install Mark Text first from: https://github.com/marktext/marktext/releases" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📥 Installation steps:" -ForegroundColor Cyan
    Write-Host "   1. Download marktext-setup.exe" -ForegroundColor White
    Write-Host "   2. Run the installer" -ForegroundColor White
    Write-Host "   3. Run this script again" -ForegroundColor White
    pause
    exit 1
}

Write-Host "✅ Found Mark Text at: $markTextPath" -ForegroundColor Green
Write-Host ""

# Markdown file extensions to associate
$markdownExtensions = @(".md", ".markdown", ".mdown", ".mkd", ".mkdn", ".mdx")

Write-Host "🔗 SETTING UP FILE ASSOCIATIONS" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

foreach ($ext in $markdownExtensions) {
    Write-Host "📄 Configuring $ext files..." -ForegroundColor Yellow
    
    try {
        # Create registry entries for file association
        $regPath = "HKEY_CLASSES_ROOT\$ext"
        $progId = "MarkText$ext"
        
        # Set file extension to point to our ProgID
        reg add $regPath /ve /d $progId /f | Out-Null
        
        # Create ProgID entry
        $progIdPath = "HKEY_CLASSES_ROOT\$progId"
        reg add $progIdPath /ve /d "Markdown Document" /f | Out-Null
        
        # Set default icon
        $iconPath = "$progIdPath\DefaultIcon"
        reg add $iconPath /ve /d "`"$markTextPath`",0" /f | Out-Null
        
        # Set shell command
        $shellPath = "$progIdPath\shell\open\command"
        reg add $shellPath /ve /d "`"$markTextPath`" `"%1`"" /f | Out-Null
        
        Write-Host "   ✅ $ext configured successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "   ❌ Failed to configure $ext : $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎯 SETTING UP CONTEXT MENU" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan
Write-Host ""

# Add "Open with Mark Text" to context menu for all files
try {
    $contextMenuPath = "HKEY_CLASSES_ROOT\*\shell\MarkText"
    reg add $contextMenuPath /ve /d "Open with Mark Text" /f | Out-Null
    reg add $contextMenuPath /v "Icon" /d "`"$markTextPath`",0" /f | Out-Null
    
    $contextMenuCommand = "$contextMenuPath\command"
    reg add $contextMenuCommand /ve /d "`"$markTextPath`" `"%1`"" /f | Out-Null
    
    Write-Host "✅ Context menu 'Open with Mark Text' added" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to add context menu: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "🔄 REFRESHING SYSTEM" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Refresh file associations
try {
    # Notify Windows of the changes
    Add-Type -TypeDefinition @"
        using System;
        using System.Runtime.InteropServices;
        public class Shell32 {
            [DllImport("shell32.dll", CharSet = CharSet.Auto, SetLastError = true)]
            public static extern void SHChangeNotify(uint wEventId, uint uFlags, IntPtr dwItem1, IntPtr dwItem2);
        }
"@
    
    [Shell32]::SHChangeNotify(0x08000000, 0x0000, [IntPtr]::Zero, [IntPtr]::Zero)
    Write-Host "✅ File associations refreshed" -ForegroundColor Green
}
catch {
    Write-Host "⚠️ Manual refresh may be required" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 CONFIGURATION COMPLETE!" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""
Write-Host "📋 What was configured:" -ForegroundColor Cyan
foreach ($ext in $markdownExtensions) {
    Write-Host "   ✅ $ext files now open with Mark Text" -ForegroundColor White
}
Write-Host "   ✅ 'Open with Mark Text' added to context menu" -ForegroundColor White
Write-Host ""

Write-Host "🚀 TESTING THE CONFIGURATION" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

# Test by opening one of the TARS documentation files
$testFiles = @(
    "C:\Users\spare\source\repos\tars\TARS_Comprehensive_Documentation\TARS_Executive_Summary_Comprehensive.md",
    "C:\Users\spare\source\repos\tars\TARS_Comprehensive_Documentation\TARS_Technical_Specification_Comprehensive.md"
)

$testFile = $null
foreach ($file in $testFiles) {
    if (Test-Path $file) {
        $testFile = $file
        break
    }
}

if ($testFile) {
    Write-Host "📄 Opening test file: $(Split-Path $testFile -Leaf)" -ForegroundColor Yellow
    Write-Host "   File: $testFile" -ForegroundColor Gray
    Write-Host ""
    
    try {
        Start-Process -FilePath $markTextPath -ArgumentList "`"$testFile`""
        Write-Host "✅ Test file opened successfully!" -ForegroundColor Green
        Write-Host "   You should see your TARS documentation with Mermaid diagrams rendered" -ForegroundColor White
    }
    catch {
        Write-Host "❌ Failed to open test file: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️ No TARS documentation files found for testing" -ForegroundColor Yellow
    Write-Host "   You can test by double-clicking any .md file" -ForegroundColor White
}

Write-Host ""
Write-Host "📋 USAGE INSTRUCTIONS" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 To open Markdown files with Mark Text:" -ForegroundColor White
Write-Host "   • Double-click any .md file" -ForegroundColor Gray
Write-Host "   • Right-click any file → 'Open with Mark Text'" -ForegroundColor Gray
Write-Host "   • Drag and drop files onto Mark Text" -ForegroundColor Gray
Write-Host ""

Write-Host "📄 Your TARS documentation files:" -ForegroundColor White
Write-Host "   • Executive Summary: TARS_Executive_Summary_Comprehensive.md" -ForegroundColor Gray
Write-Host "   • Technical Spec: TARS_Technical_Specification_Comprehensive.md" -ForegroundColor Gray
Write-Host "   • API Documentation: TARS_API_Documentation.md" -ForegroundColor Gray
Write-Host ""

Write-Host "🎉 SETUP COMPLETE! Mark Text is now your default Markdown viewer!" -ForegroundColor Green
Write-Host "   All Mermaid diagrams and mathematical formulas will render beautifully!" -ForegroundColor White
Write-Host ""

pause
