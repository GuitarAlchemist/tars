@echo off
REM TARS CLI Launcher
REM This batch file makes it easy to run the TARS CLI from the repository root

REM Set console code page to UTF-8 to handle special characters correctly
chcp 65001 > nul

REM Check for the executable in standard build output locations
set POSSIBLE_PATHS=TarsCli\bin\Debug\net9.0\tarscli.exe TarsCli\bin\Release\net9.0\tarscli.exe

for %%p in (%POSSIBLE_PATHS%) do (
    if exist %%p (
        echo Found TARS CLI at: %%p
        %%p %*
        exit /b %errorlevel%
    )
)

REM If we get here, we didn't find the executable
echo TARS CLI executable not found. Make sure you've built the solution.
echo Checked the following locations:
for %%p in (%POSSIBLE_PATHS%) do (
    echo - %%p
)
echo.
echo Try building the solution with: dotnet build TarsCli\TarsCli.csproj
exit /b 1
