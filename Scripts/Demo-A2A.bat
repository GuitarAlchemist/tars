@echo off
REM TARS A2A Protocol Demo Batch File
REM This script demonstrates TARS A2A protocol capabilities

echo.
echo ===================================================
echo         TARS A2A PROTOCOL DEMONSTRATION
echo ===================================================
echo.
echo This batch file will demonstrate the A2A (Agent-to-Agent) protocol capabilities of TARS.
echo The A2A protocol enables interoperability between different AI agents.
echo Press Ctrl+C at any time to exit the demo.
echo.

REM Check if TARS CLI exists
if not exist "%~dp0..\TarsCli\bin\Debug\net9.0\tarscli.exe" (
    echo Error: TARS CLI not found.
    echo Please build the TARS CLI project first by running 'dotnet build' in the TarsCli directory.
    goto :EOF
)

set TARS_CLI="%~dp0..\TarsCli\bin\Debug\net9.0\tarscli.exe"
set CURRENT_DIR=%CD%
cd "%~dp0.."

echo Starting demo in 3 seconds...
timeout /t 3 > nul

REM Section 1: A2A Server
echo.
echo ===================================================
echo                  A2A SERVER
echo ===================================================
echo.

echo Starting the A2A Server
echo ----------------------
echo.
echo ^> tarscli a2a start
%TARS_CLI% a2a start
echo.

timeout /t 3 > nul

REM Section 2: Agent Card
echo.
echo ===================================================
echo                  AGENT CARD
echo ===================================================
echo.

echo Getting the TARS Agent Card
echo -------------------------
echo.
echo ^> tarscli a2a get-agent-card --agent-url http://localhost:8998/
%TARS_CLI% a2a get-agent-card --agent-url http://localhost:8998/
echo.

timeout /t 3 > nul

REM Section 3: Code Generation
echo.
echo ===================================================
echo              CODE GENERATION SKILL
echo ===================================================
echo.

echo Sending a Code Generation Task
echo ---------------------------
echo.
echo ^> tarscli a2a send --agent-url http://localhost:8998/ --message "Generate a C# class for a Customer entity with properties for ID, Name, Email, and Address" --skill-id code_generation
%TARS_CLI% a2a send --agent-url http://localhost:8998/ --message "Generate a C# class for a Customer entity with properties for ID, Name, Email, and Address" --skill-id code_generation
echo.

timeout /t 3 > nul

REM Section 4: Code Analysis
echo.
echo ===================================================
echo               CODE ANALYSIS SKILL
echo ===================================================
echo.

echo Sending a Code Analysis Task
echo -------------------------
echo.
echo ^> tarscli a2a send --agent-url http://localhost:8998/ --message "Analyze this code for potential issues: public void ProcessData(string data) { var result = data.Split(','); Console.WriteLine(result[0]); }" --skill-id code_analysis
%TARS_CLI% a2a send --agent-url http://localhost:8998/ --message "Analyze this code for potential issues: public void ProcessData(string data) { var result = data.Split(','); Console.WriteLine(result[0]); }" --skill-id code_analysis
echo.

timeout /t 3 > nul

REM Section 5: Knowledge Extraction
echo.
echo ===================================================
echo           KNOWLEDGE EXTRACTION SKILL
echo ===================================================
echo.

echo Sending a Knowledge Extraction Task
echo -------------------------------
echo.
echo ^> tarscli a2a send --agent-url http://localhost:8998/ --message "Extract key concepts from the A2A protocol documentation" --skill-id knowledge_extraction
%TARS_CLI% a2a send --agent-url http://localhost:8998/ --message "Extract key concepts from the A2A protocol documentation" --skill-id knowledge_extraction
echo.

timeout /t 3 > nul

REM Section 6: Self Improvement
echo.
echo ===================================================
echo             SELF IMPROVEMENT SKILL
echo ===================================================
echo.

echo Sending a Self Improvement Task
echo ----------------------------
echo.
echo ^> tarscli a2a send --agent-url http://localhost:8998/ --message "Suggest improvements for the A2A protocol implementation" --skill-id self_improvement
%TARS_CLI% a2a send --agent-url http://localhost:8998/ --message "Suggest improvements for the A2A protocol implementation" --skill-id self_improvement
echo.

timeout /t 3 > nul

REM Section 7: MCP Bridge
echo.
echo ===================================================
echo                  MCP BRIDGE
echo ===================================================
echo.

echo Using A2A through MCP
echo ------------------
echo.
echo ^> tarscli mcp execute --action a2a --operation send_task --agent_url http://localhost:8998/ --content "Generate a simple logging class in C#" --skill_id code_generation
%TARS_CLI% mcp execute --action a2a --operation send_task --agent_url http://localhost:8998/ --content "Generate a simple logging class in C#" --skill_id code_generation
echo.

timeout /t 3 > nul

REM Section 8: Stopping the Server
echo.
echo ===================================================
echo              STOPPING THE SERVER
echo ===================================================
echo.

echo Stopping the A2A Server
echo ---------------------
echo.
echo ^> tarscli a2a stop
%TARS_CLI% a2a stop
echo.

REM Conclusion
echo.
echo ===================================================
echo                 DEMO COMPLETE
echo ===================================================
echo.
echo This concludes the demonstration of TARS A2A protocol capabilities.
echo The A2A protocol enables TARS to communicate with other A2A-compatible agents
echo and expose its capabilities through a standardized interface.
echo.
echo For more information, see the A2A protocol documentation:
echo docs/A2A-Protocol.md
echo.

REM Restore original directory
cd %CURRENT_DIR%
