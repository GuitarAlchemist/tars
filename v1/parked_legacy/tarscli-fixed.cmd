@echo off
REM TARS CLI Fixed Launcher
REM This batch file makes it easy to run the fixed TARS CLI from the repository root

REM Set console code page to UTF-8 to handle special characters correctly
chcp 65001 > nul

REM Check for the executable in standard build output locations
set POSSIBLE_PATHS=TarsCliMinimal\bin\Debug\net9.0\tarscli.exe TarsCliMinimal\bin\Release\net9.0\tarscli.exe

for %%p in (%POSSIBLE_PATHS%) do (
    if exist %%p (
        echo Found TARS CLI Fixed at: %%p
        %%p %*
        exit /b %errorlevel%
    )
)

REM If we get here, we didn't find the executable, so try to build it
echo TARS CLI Fixed executable not found. Attempting to build it...
dotnet build TarsCliMinimal\TarsCliMinimal.csproj -c Debug

REM Check if the build was successful
if %errorlevel% neq 0 (
    echo Failed to build TARS CLI Fixed.
    exit /b %errorlevel%
)

REM Try to run the executable again
for %%p in (%POSSIBLE_PATHS%) do (
    if exist %%p (
        echo Found TARS CLI Fixed at: %%p
        %%p %*
        exit /b %errorlevel%
    )
)

REM If we still can't find the executable, run it using dotnet run
echo Running TARS CLI Fixed using dotnet run...
dotnet run --project TarsCliMinimal\TarsCliMinimal.csproj -- %*
exit /b %errorlevel%
