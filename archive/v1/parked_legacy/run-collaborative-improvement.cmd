@echo off
echo Starting Collaborative Improvement between Augment Code and TARS CLI...

REM Start the TARS MCP service
echo Starting TARS MCP service...
start /b cmd /c "tarscli.cmd mcp start --url http://localhost:9000/"

REM Wait for the MCP service to initialize
echo Waiting for MCP service to initialize...
timeout /t 10 /nobreak > nul

REM Run the collaborative improvement script
echo Running collaborative improvement script...
npm start

REM Stop the TARS MCP service when done
echo Stopping TARS MCP service...
tarscli.cmd mcp stop --url http://localhost:9000/

echo Collaborative improvement process completed.
pause
