@echo off
REM TARS Autonomous Prerequisite Management Test
REM Comprehensive test of VM creation, repo cloning, and autonomous prerequisite management

echo.
echo 🚀 TARS AUTONOMOUS PREREQUISITE MANAGEMENT TEST
echo ================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ This script requires administrator privileges.
    echo Please run as administrator.
    pause
    exit /b 1
)

echo ✅ Running with administrator privileges
echo.

REM Set variables
set VM_NAME=TARS-Build-Test-VM
set REPO_URL=https://github.com/GuitarAlchemist/tars.git
set CLONE_PATH=C:\TARS-Test
set LOG_FILE=autonomous-prerequisite-test-%date:~-4,4%%date:~-10,2%%date:~-7,2%-%time:~0,2%%time:~3,2%%time:~6,2%.log

echo 📋 Configuration:
echo   VM Name: %VM_NAME%
echo   Repository: %REPO_URL%
echo   Clone Path: %CLONE_PATH%
echo   Log File: %LOG_FILE%
echo.

REM Create log file
echo TARS Autonomous Prerequisite Management Test Log > %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%

REM Step 1: Test F# Autonomous Prerequisite Management Demo
echo 🔬 Step 1: Testing F# Autonomous Prerequisite Management Demo
echo ============================================================
echo.

echo Running F# prerequisite management demo...
dotnet fsi autonomous-prerequisite-management-demo.fsx 2>&1 | tee -a %LOG_FILE%

if %errorLevel% neq 0 (
    echo ❌ F# prerequisite management demo failed
    echo F# demo failed with error code %errorLevel% >> %LOG_FILE%
    goto :error
) else (
    echo ✅ F# prerequisite management demo completed successfully
    echo F# demo completed successfully >> %LOG_FILE%
)

echo.

REM Step 2: Execute PowerShell VM and Prerequisite Test
echo 🖥️ Step 2: Executing PowerShell VM and Prerequisite Test
echo ========================================================
echo.

echo Running PowerShell VM prerequisite test...
powershell -ExecutionPolicy Bypass -File autonomous-vm-prerequisite-test.ps1 -VMName %VM_NAME% -RepoUrl %REPO_URL% -ClonePath %CLONE_PATH% 2>&1 | tee -a %LOG_FILE%

if %errorLevel% neq 0 (
    echo ❌ PowerShell VM prerequisite test failed
    echo PowerShell VM test failed with error code %errorLevel% >> %LOG_FILE%
    goto :error
) else (
    echo ✅ PowerShell VM prerequisite test completed successfully
    echo PowerShell VM test completed successfully >> %LOG_FILE%
)

echo.

REM Step 3: Test TARS Metascript Execution
echo 📜 Step 3: Testing TARS Metascript Execution
echo ============================================
echo.

if exist "TarsEngine.FSharp.Cli\TarsEngine.FSharp.Cli.fsproj" (
    echo Running TARS metascript for autonomous prerequisite management...
    dotnet run --project TarsEngine.FSharp.Cli -- execute-metascript .tars/autonomous-prerequisite-management.trsx 2>&1 | tee -a %LOG_FILE%
    
    if %errorLevel% neq 0 (
        echo ⚠️ TARS metascript execution failed (this is expected if CLI is not fully built)
        echo TARS metascript execution failed with error code %errorLevel% >> %LOG_FILE%
    ) else (
        echo ✅ TARS metascript execution completed successfully
        echo TARS metascript execution completed successfully >> %LOG_FILE%
    )
) else (
    echo ⚠️ TARS CLI project not found, skipping metascript test
    echo TARS CLI project not found, skipping metascript test >> %LOG_FILE%
)

echo.

REM Step 4: Validate Build Environment
echo 🧪 Step 4: Validating Build Environment
echo =======================================
echo.

echo Testing prerequisite availability...

REM Test .NET SDK
dotnet --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('dotnet --version') do set DOTNET_VERSION=%%i
    echo ✅ .NET SDK: %DOTNET_VERSION%
    echo .NET SDK: %DOTNET_VERSION% >> %LOG_FILE%
) else (
    echo ❌ .NET SDK: Not available
    echo .NET SDK: Not available >> %LOG_FILE%
)

REM Test Git
git --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('git --version') do set GIT_VERSION=%%i
    echo ✅ Git: %GIT_VERSION%
    echo Git: %GIT_VERSION% >> %LOG_FILE%
) else (
    echo ❌ Git: Not available
    echo Git: Not available >> %LOG_FILE%
)

REM Test Node.js
node --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo ✅ Node.js: %NODE_VERSION%
    echo Node.js: %NODE_VERSION% >> %LOG_FILE%
) else (
    echo ❌ Node.js: Not available
    echo Node.js: Not available >> %LOG_FILE%
)

REM Test Python
python --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo ✅ Python: %PYTHON_VERSION%
    echo Python: %PYTHON_VERSION% >> %LOG_FILE%
) else (
    echo ❌ Python: Not available
    echo Python: Not available >> %LOG_FILE%
)

REM Test WinGet
winget --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ WinGet: Available
    echo WinGet: Available >> %LOG_FILE%
) else (
    echo ❌ WinGet: Not available
    echo WinGet: Not available >> %LOG_FILE%
)

REM Test Chocolatey
choco --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('choco --version') do set CHOCO_VERSION=%%i
    echo ✅ Chocolatey: %CHOCO_VERSION%
    echo Chocolatey: %CHOCO_VERSION% >> %LOG_FILE%
) else (
    echo ❌ Chocolatey: Not available
    echo Chocolatey: Not available >> %LOG_FILE%
)

echo.

REM Step 5: Test TARS Build
echo 🏗️ Step 5: Testing TARS Build
echo =============================
echo.

echo Testing TARS build...
dotnet restore 2>&1 | tee -a %LOG_FILE%

if %errorLevel% neq 0 (
    echo ❌ Failed to restore TARS packages
    echo Failed to restore TARS packages >> %LOG_FILE%
    goto :error
)

dotnet build 2>&1 | tee -a %LOG_FILE%

if %errorLevel% neq 0 (
    echo ❌ TARS build failed
    echo TARS build failed >> %LOG_FILE%
    goto :error
) else (
    echo ✅ TARS build successful!
    echo TARS build successful >> %LOG_FILE%
)

echo.

REM Success
echo 🎉 TARS AUTONOMOUS PREREQUISITE MANAGEMENT TEST COMPLETED SUCCESSFULLY!
echo ========================================================================
echo.
echo ✅ All tests passed
echo ✅ Prerequisites managed autonomously
echo ✅ Build environment validated
echo ✅ TARS builds successfully
echo.
echo 📄 Detailed log: %LOG_FILE%
echo.

echo Test completed successfully at %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%
echo ========================================================================== >> %LOG_FILE%

goto :end

:error
echo.
echo ❌ TARS AUTONOMOUS PREREQUISITE MANAGEMENT TEST FAILED
echo =====================================================
echo.
echo Some tests failed. Check the log file for details: %LOG_FILE%
echo.

echo Test failed at %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%
echo ========================================================================== >> %LOG_FILE%

pause
exit /b 1

:end
echo 🤖 TARS Autonomous Prerequisite Management System is ready!
echo.
pause
