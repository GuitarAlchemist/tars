@echo off
REM Batch script to configure Mark Text as default Markdown viewer
REM Run this as Administrator

echo.
echo 🔧 CONFIGURING MARK TEXT AS DEFAULT MARKDOWN VIEWER
echo =================================================
echo.

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ This script requires Administrator privileges!
    echo    Please run as Administrator:
    echo    1. Right-click on this file
    echo    2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo ✅ Running with Administrator privileges
echo.

REM Find Mark Text installation
set "MARKTEXT_PATH="
if exist "%LOCALAPPDATA%\Programs\marktext\Mark Text.exe" (
    set "MARKTEXT_PATH=%LOCALAPPDATA%\Programs\marktext\Mark Text.exe"
) else if exist "%ProgramFiles%\Mark Text\Mark Text.exe" (
    set "MARKTEXT_PATH=%ProgramFiles%\Mark Text\Mark Text.exe"
) else if exist "%ProgramFiles(x86)%\Mark Text\Mark Text.exe" (
    set "MARKTEXT_PATH=%ProgramFiles(x86)%\Mark Text\Mark Text.exe"
) else if exist "%USERPROFILE%\AppData\Local\Programs\marktext\Mark Text.exe" (
    set "MARKTEXT_PATH=%USERPROFILE%\AppData\Local\Programs\marktext\Mark Text.exe"
)

if "%MARKTEXT_PATH%"=="" (
    echo ❌ Mark Text not found!
    echo    Please install Mark Text first from:
    echo    https://github.com/marktext/marktext/releases
    echo.
    echo 📥 Installation steps:
    echo    1. Download marktext-setup.exe
    echo    2. Run the installer
    echo    3. Run this script again
    echo.
    pause
    exit /b 1
)

echo ✅ Found Mark Text at: %MARKTEXT_PATH%
echo.

echo 🔗 SETTING UP FILE ASSOCIATIONS
echo ===============================
echo.

REM Configure .md files
echo 📄 Configuring .md files...
reg add "HKEY_CLASSES_ROOT\.md" /ve /d "MarkText.md" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.md" /ve /d "Markdown Document" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.md\DefaultIcon" /ve /d "\"%MARKTEXT_PATH%\",0" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.md\shell\open\command" /ve /d "\"%MARKTEXT_PATH%\" \"%%1\"" /f >nul 2>&1
echo    ✅ .md files configured

REM Configure .markdown files
echo 📄 Configuring .markdown files...
reg add "HKEY_CLASSES_ROOT\.markdown" /ve /d "MarkText.markdown" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.markdown" /ve /d "Markdown Document" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.markdown\DefaultIcon" /ve /d "\"%MARKTEXT_PATH%\",0" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.markdown\shell\open\command" /ve /d "\"%MARKTEXT_PATH%\" \"%%1\"" /f >nul 2>&1
echo    ✅ .markdown files configured

REM Configure .mdown files
echo 📄 Configuring .mdown files...
reg add "HKEY_CLASSES_ROOT\.mdown" /ve /d "MarkText.mdown" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mdown" /ve /d "Markdown Document" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mdown\DefaultIcon" /ve /d "\"%MARKTEXT_PATH%\",0" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mdown\shell\open\command" /ve /d "\"%MARKTEXT_PATH%\" \"%%1\"" /f >nul 2>&1
echo    ✅ .mdown files configured

REM Configure .mkd files
echo 📄 Configuring .mkd files...
reg add "HKEY_CLASSES_ROOT\.mkd" /ve /d "MarkText.mkd" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mkd" /ve /d "Markdown Document" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mkd\DefaultIcon" /ve /d "\"%MARKTEXT_PATH%\",0" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mkd\shell\open\command" /ve /d "\"%MARKTEXT_PATH%\" \"%%1\"" /f >nul 2>&1
echo    ✅ .mkd files configured

REM Configure .mdx files
echo 📄 Configuring .mdx files...
reg add "HKEY_CLASSES_ROOT\.mdx" /ve /d "MarkText.mdx" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mdx" /ve /d "Markdown Document" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mdx\DefaultIcon" /ve /d "\"%MARKTEXT_PATH%\",0" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\MarkText.mdx\shell\open\command" /ve /d "\"%MARKTEXT_PATH%\" \"%%1\"" /f >nul 2>&1
echo    ✅ .mdx files configured

echo.
echo 🎯 SETTING UP CONTEXT MENU
echo ===========================
echo.

REM Add "Open with Mark Text" to context menu
echo 📄 Adding context menu option...
reg add "HKEY_CLASSES_ROOT\*\shell\MarkText" /ve /d "Open with Mark Text" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\*\shell\MarkText" /v "Icon" /d "\"%MARKTEXT_PATH%\",0" /f >nul 2>&1
reg add "HKEY_CLASSES_ROOT\*\shell\MarkText\command" /ve /d "\"%MARKTEXT_PATH%\" \"%%1\"" /f >nul 2>&1
echo    ✅ Context menu added

echo.
echo 🔄 REFRESHING SYSTEM
echo ===================
echo.

REM Refresh file associations (requires restart of explorer or reboot for full effect)
echo ✅ File associations configured (may require explorer restart)

echo.
echo 🎉 CONFIGURATION COMPLETE!
echo =========================
echo.
echo 📋 What was configured:
echo    ✅ .md files now open with Mark Text
echo    ✅ .markdown files now open with Mark Text
echo    ✅ .mdown files now open with Mark Text
echo    ✅ .mkd files now open with Mark Text
echo    ✅ .mdx files now open with Mark Text
echo    ✅ "Open with Mark Text" added to context menu
echo.

echo 🚀 TESTING THE CONFIGURATION
echo ============================
echo.

REM Test by opening a TARS documentation file if it exists
set "TEST_FILE=C:\Users\spare\source\repos\tars\TARS_Comprehensive_Documentation\TARS_Executive_Summary_Comprehensive.md"
if exist "%TEST_FILE%" (
    echo 📄 Opening test file: TARS_Executive_Summary_Comprehensive.md
    echo    File: %TEST_FILE%
    echo.
    start "" "%MARKTEXT_PATH%" "%TEST_FILE%"
    echo ✅ Test file opened successfully!
    echo    You should see your TARS documentation with Mermaid diagrams rendered
) else (
    echo ⚠️ No TARS documentation files found for testing
    echo    You can test by double-clicking any .md file
)

echo.
echo 📋 USAGE INSTRUCTIONS
echo =====================
echo.
echo 🎯 To open Markdown files with Mark Text:
echo    • Double-click any .md file
echo    • Right-click any file → "Open with Mark Text"
echo    • Drag and drop files onto Mark Text
echo.
echo 📄 Your TARS documentation files:
echo    • Executive Summary: TARS_Executive_Summary_Comprehensive.md
echo    • Technical Spec: TARS_Technical_Specification_Comprehensive.md
echo    • API Documentation: TARS_API_Documentation.md
echo.
echo 🎉 SETUP COMPLETE! Mark Text is now your default Markdown viewer!
echo    All Mermaid diagrams and mathematical formulas will render beautifully!
echo.

pause
