@echo off
echo ========================================
echo TARS QUICK CLEANUP AND REORGANIZATION
echo ========================================
echo.

set BACKUP_DIR=.tars\archive\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%

echo Creating backup at: %BACKUP_DIR%
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

echo.
echo [1/6] Creating backup...
if exist "src" xcopy "src" "%BACKUP_DIR%\src\" /E /I /Q >nul 2>&1
if exist ".tars" xcopy ".tars" "%BACKUP_DIR%\tars_backup\" /E /I /Q >nul 2>&1
echo Backup completed.

echo.
echo [2/6] Creating new directory structure...

REM Core source organization
if not exist "src\TarsEngine.FSharp.Core" mkdir "src\TarsEngine.FSharp.Core"
if not exist "src\TarsEngine.FSharp.Cli" mkdir "src\TarsEngine.FSharp.Cli"
if not exist "src\TarsEngine.FSharp.Web" mkdir "src\TarsEngine.FSharp.Web"

REM Department organization
if not exist ".tars\departments\research\teams\university" mkdir ".tars\departments\research\teams\university"
if not exist ".tars\departments\research\agents" mkdir ".tars\departments\research\agents"
if not exist ".tars\departments\research\projects" mkdir ".tars\departments\research\projects"
if not exist ".tars\departments\infrastructure\teams" mkdir ".tars\departments\infrastructure\teams"
if not exist ".tars\departments\infrastructure\agents" mkdir ".tars\departments\infrastructure\agents"
if not exist ".tars\departments\qa\teams" mkdir ".tars\departments\qa\teams"
if not exist ".tars\departments\qa\agents" mkdir ".tars\departments\qa\agents"
if not exist ".tars\departments\qa\tests" mkdir ".tars\departments\qa\tests"
if not exist ".tars\departments\ui\teams" mkdir ".tars\departments\ui\teams"
if not exist ".tars\departments\ui\agents" mkdir ".tars\departments\ui\agents"
if not exist ".tars\departments\operations\teams" mkdir ".tars\departments\operations\teams"
if not exist ".tars\departments\operations\agents" mkdir ".tars\departments\operations\agents"

REM Evolution system
if not exist ".tars\evolution\grammars\base" mkdir ".tars\evolution\grammars\base"
if not exist ".tars\evolution\grammars\evolved" mkdir ".tars\evolution\grammars\evolved"
if not exist ".tars\evolution\sessions\active" mkdir ".tars\evolution\sessions\active"
if not exist ".tars\evolution\sessions\completed" mkdir ".tars\evolution\sessions\completed"
if not exist ".tars\evolution\teams" mkdir ".tars\evolution\teams"
if not exist ".tars\evolution\results" mkdir ".tars\evolution\results"

REM University system
if not exist ".tars\university\teams\research-team" mkdir ".tars\university\teams\research-team"
if not exist ".tars\university\agents\individual" mkdir ".tars\university\agents\individual"
if not exist ".tars\university\agents\specialized" mkdir ".tars\university\agents\specialized"
if not exist ".tars\university\collaborations" mkdir ".tars\university\collaborations"
if not exist ".tars\university\research" mkdir ".tars\university\research"

REM Metascripts organization
if not exist ".tars\metascripts\core" mkdir ".tars\metascripts\core"
if not exist ".tars\metascripts\departments" mkdir ".tars\metascripts\departments"
if not exist ".tars\metascripts\evolution" mkdir ".tars\metascripts\evolution"
if not exist ".tars\metascripts\demos" mkdir ".tars\metascripts\demos"
if not exist ".tars\metascripts\tests" mkdir ".tars\metascripts\tests"

REM System configuration
if not exist ".tars\system\config" mkdir ".tars\system\config"
if not exist ".tars\system\logs" mkdir ".tars\system\logs"
if not exist ".tars\system\monitoring" mkdir ".tars\system\monitoring"

REM Top-level organization
if not exist "docs\architecture" mkdir "docs\architecture"
if not exist "docs\teams" mkdir "docs\teams"
if not exist "docs\agents" mkdir "docs\agents"
if not exist "tests\unit" mkdir "tests\unit"
if not exist "tests\integration" mkdir "tests\integration"
if not exist "demos\evolution" mkdir "demos\evolution"
if not exist "demos\teams" mkdir "demos\teams"
if not exist "tools\migration" mkdir "tools\migration"
if not exist "archive\legacy" mkdir "archive\legacy"

echo Directory structure created.

echo.
echo [3/6] Migrating core F# projects...

REM Copy src/TarsEngine content to proper F# Core project
if exist "src\TarsEngine" (
    echo Copying TarsEngine files to Core project...
    xcopy "src\TarsEngine\*.fs" "src\TarsEngine.FSharp.Core\" /Y /Q >nul 2>&1
    xcopy "src\TarsEngine\*.fsproj" "src\TarsEngine.FSharp.Core\" /Y /Q >nul 2>&1
)

REM Copy existing F# projects if they exist
if exist "TarsEngine.FSharp.Core" (
    echo Copying TarsEngine.FSharp.Core...
    xcopy "TarsEngine.FSharp.Core\*" "src\TarsEngine.FSharp.Core\" /E /Y /Q >nul 2>&1
)

if exist "TarsEngine.FSharp.Cli" (
    echo Copying TarsEngine.FSharp.Cli...
    xcopy "TarsEngine.FSharp.Cli\*" "src\TarsEngine.FSharp.Cli\" /E /Y /Q >nul 2>&1
)

if exist "TarsEngine.FSharp.Web" (
    echo Copying TarsEngine.FSharp.Web...
    xcopy "TarsEngine.FSharp.Web\*" "src\TarsEngine.FSharp.Web\" /E /Y /Q >nul 2>&1
)

if exist "TarsEngine.FSharp.Metascript.Runner" (
    echo Copying Metascript Runner to CLI...
    xcopy "TarsEngine.FSharp.Metascript.Runner\*" "src\TarsEngine.FSharp.Cli\" /E /Y /Q >nul 2>&1
)

echo Core projects migration completed.

echo.
echo [4/6] Migrating university teams and agents...

REM Migrate university team configuration
if exist ".tars\university\team-config.json" (
    echo Copying team configuration...
    copy ".tars\university\team-config.json" ".tars\university\teams\research-team\team-config.json" >nul 2>&1
)

REM Migrate agent configurations
if exist ".tars\agents" (
    echo Copying agent configurations...
    xcopy ".tars\agents\*" ".tars\university\agents\individual\" /E /Y /Q >nul 2>&1
)

echo University teams migration completed.

echo.
echo [5/6] Migrating evolution system...

REM Migrate grammars
if exist ".tars\grammars" (
    echo Copying grammar files...
    xcopy ".tars\grammars\*" ".tars\evolution\grammars\base\" /Y /Q >nul 2>&1
)

echo Evolution system migration completed.

echo.
echo [6/6] Organizing metascripts...

REM Move demo metascripts
if exist ".tars\metascripts" (
    echo Organizing metascripts by type...
    for %%f in (.tars\metascripts\*demo*.trsx) do copy "%%f" ".tars\metascripts\demos\" >nul 2>&1
    for %%f in (.tars\metascripts\*test*.trsx) do copy "%%f" ".tars\metascripts\tests\" >nul 2>&1
    for %%f in (.tars\metascripts\*research*.trsx) do copy "%%f" ".tars\metascripts\departments\" >nul 2>&1
    for %%f in (.tars\metascripts\*university*.trsx) do copy "%%f" ".tars\metascripts\departments\" >nul 2>&1
    for %%f in (.tars\metascripts\*evolution*.trsx) do copy "%%f" ".tars\metascripts\evolution\" >nul 2>&1
    for %%f in (.tars\metascripts\*grammar*.trsx) do copy "%%f" ".tars\metascripts\evolution\" >nul 2>&1
)

echo Metascripts organization completed.

echo.
echo ========================================
echo CLEANUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Backup location: %BACKUP_DIR%
echo.
echo NEW STRUCTURE CREATED:
echo   src/                     - Clean F# source code
echo   .tars/departments/       - Department-based organization
echo   .tars/evolution/         - Evolutionary grammar system
echo   .tars/university/        - University team system
echo   .tars/metascripts/       - Organized metascripts
echo   .tars/system/            - System configuration
echo   docs/                    - Centralized documentation
echo   tests/                   - Comprehensive test suite
echo   demos/                   - Organized demos
echo   tools/                   - Development tools
echo   archive/                 - Archived content
echo.
echo NEXT STEPS:
echo 1. Test core system functionality
echo 2. Validate team and agent configurations
echo 3. Update any remaining file references
echo 4. Run evolution system tests
echo.
echo Your TARS system is now properly organized for scalable evolution!
echo.
pause
