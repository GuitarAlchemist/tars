@echo off
echo Starting TARS MCP Server in background...
start /b cmd /c ".\tarscli.cmd mcp start > tars-mcp.log 2>&1"
echo TARS MCP Server started. Check tars-mcp.log for details.
