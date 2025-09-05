@echo off
echo Fixing TARS CLI Chat Command Duplication...
echo.

set "file=src\TarsEngine.FSharp.Cli\TarsEngine.FSharp.Cli\Commands\ChatbotCommand.fs"

if exist "%file%" (
    echo Creating backup...
    copy "%file%" "%file%.backup" >nul
    
    echo Applying fix...
    powershell -Command "(Get-Content '%file%') -replace '\[yellow\]🎯 Available Commands:\[/\].*?\[green\]exit\[/\] - Exit chatbot', '[yellow]Type ''[green]help[/]'' to see available commands or ''[green]exit[/]'' to quit.[/]' | Set-Content '%file%'"
    
    echo Fix applied successfully!
    echo Commands are now only shown when user types 'help'
) else (
    echo File not found: %file%
)

echo.
echo Done!
pause
