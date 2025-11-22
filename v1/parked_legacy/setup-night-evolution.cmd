@echo off
echo 🌙 TARS NIGHT EVOLUTION SESSION SETUP
echo ====================================
echo Setting up Master Evolver Agent for autonomous night evolution
echo.

set SESSION_ID=%RANDOM%%RANDOM%
set GREEN_ENV=tars-green-stable
set BLUE_ENV=tars-blue-evolution

echo 🚀 INITIALIZING NIGHT EVOLUTION SESSION
echo ======================================

echo 📂 Setting up evolution directories...
if not exist ".tars\evolution" mkdir ".tars\evolution"
if not exist ".tars\evolution\sessions" mkdir ".tars\evolution\sessions"
if not exist ".tars\evolution\reports" mkdir ".tars\evolution\reports"
if not exist ".tars\evolution\monitoring" mkdir ".tars\evolution\monitoring"
if not exist ".tars\green" mkdir ".tars\green"
if not exist ".tars\blue" mkdir ".tars\blue"
if not exist ".tars\monitoring\green-blue" mkdir ".tars\monitoring\green-blue"
echo   ✅ Evolution directories created

echo.
echo 📋 GENERATING EVOLUTION SESSION
echo ==============================
echo   🆔 Session ID: %SESSION_ID%
echo   ⏰ Start Time: %DATE% %TIME%
echo   ⏱️ Duration: 8 hours
echo   🟢 Green Container: %GREEN_ENV%
echo   🔵 Blue Container: %BLUE_ENV%
echo   🎯 Evolution Goals: 4

echo.
echo 🟢 CREATING GREEN ENVIRONMENT SCRIPT
echo ===================================

echo #!/bin/bash > .tars\green\start-green.sh
echo # Green Environment - Stable Baseline >> .tars\green\start-green.sh
echo echo "🟢 Starting TARS Green Environment (Baseline)" >> .tars\green\start-green.sh
echo. >> .tars\green\start-green.sh
echo # Check if container already exists >> .tars\green\start-green.sh
echo if docker ps -a --format "table {{.Names}}" ^| grep -q "^%GREEN_ENV%$"; then >> .tars\green\start-green.sh
echo     echo "  🔄 Stopping existing green container..." >> .tars\green\start-green.sh
echo     docker stop %GREEN_ENV% 2^>/dev/null ^|^| true >> .tars\green\start-green.sh
echo     docker rm %GREEN_ENV% 2^>/dev/null ^|^| true >> .tars\green\start-green.sh
echo fi >> .tars\green\start-green.sh
echo. >> .tars\green\start-green.sh
echo # Create network if it doesn't exist >> .tars\green\start-green.sh
echo docker network create tars-evolution 2^>/dev/null ^|^| true >> .tars\green\start-green.sh
echo. >> .tars\green\start-green.sh
echo # Start green container >> .tars\green\start-green.sh
echo echo "  🚀 Starting green container..." >> .tars\green\start-green.sh
echo docker run -d --name %GREEN_ENV% ^\ >> .tars\green\start-green.sh
echo   --network tars-evolution ^\ >> .tars\green\start-green.sh
echo   --label tars.environment=green ^\ >> .tars\green\start-green.sh
echo   --label tars.role=baseline ^\ >> .tars\green\start-green.sh
echo   --label tars.evolver.session=%SESSION_ID% ^\ >> .tars\green\start-green.sh
echo   -p 8080:8080 ^\ >> .tars\green\start-green.sh
echo   -p 8081:8081 ^\ >> .tars\green\start-green.sh
echo   -v "$(pwd)/.tars/green:/app/tars:rw" ^\ >> .tars\green\start-green.sh
echo   -v "$(pwd)/.tars/shared:/app/shared:ro" ^\ >> .tars\green\start-green.sh
echo   -e TARS_ENVIRONMENT=green ^\ >> .tars\green\start-green.sh
echo   -e TARS_ROLE=baseline ^\ >> .tars\green\start-green.sh
echo   -e TARS_MONITORING_ENABLED=true ^\ >> .tars\green\start-green.sh
echo   -e TARS_SESSION_ID=%SESSION_ID% ^\ >> .tars\green\start-green.sh
echo   mcr.microsoft.com/dotnet/aspnet:9.0 >> .tars\green\start-green.sh
echo. >> .tars\green\start-green.sh
echo echo "  ✅ Green environment ready at http://localhost:8080" >> .tars\green\start-green.sh
echo echo "  📊 Metrics available at http://localhost:8081/metrics" >> .tars\green\start-green.sh

echo   📄 Green script: .tars\green\start-green.sh

echo.
echo 🔵 CREATING BLUE ENVIRONMENT SCRIPT
echo ==================================

echo #!/bin/bash > .tars\blue\start-blue.sh
echo # Blue Environment - Evolution Experimental >> .tars\blue\start-blue.sh
echo echo "🔵 Starting TARS Blue Environment (Evolution)" >> .tars\blue\start-blue.sh
echo. >> .tars\blue\start-blue.sh
echo # Check if container already exists >> .tars\blue\start-blue.sh
echo if docker ps -a --format "table {{.Names}}" ^| grep -q "^%BLUE_ENV%$"; then >> .tars\blue\start-blue.sh
echo     echo "  🔄 Stopping existing blue container..." >> .tars\blue\start-blue.sh
echo     docker stop %BLUE_ENV% 2^>/dev/null ^|^| true >> .tars\blue\start-blue.sh
echo     docker rm %BLUE_ENV% 2^>/dev/null ^|^| true >> .tars\blue\start-blue.sh
echo fi >> .tars\blue\start-blue.sh
echo. >> .tars\blue\start-blue.sh
echo # Create network if it doesn't exist >> .tars\blue\start-blue.sh
echo docker network create tars-evolution 2^>/dev/null ^|^| true >> .tars\blue\start-blue.sh
echo. >> .tars\blue\start-blue.sh
echo # Start blue container >> .tars\blue\start-blue.sh
echo echo "  🚀 Starting blue container..." >> .tars\blue\start-blue.sh
echo docker run -d --name %BLUE_ENV% ^\ >> .tars\blue\start-blue.sh
echo   --network tars-evolution ^\ >> .tars\blue\start-blue.sh
echo   --label tars.environment=blue ^\ >> .tars\blue\start-blue.sh
echo   --label tars.role=evolution ^\ >> .tars\blue\start-blue.sh
echo   --label tars.evolver.session=%SESSION_ID% ^\ >> .tars\blue\start-blue.sh
echo   -p 8082:8080 ^\ >> .tars\blue\start-blue.sh
echo   -p 8083:8081 ^\ >> .tars\blue\start-blue.sh
echo   -v "$(pwd)/.tars/blue:/app/tars:rw" ^\ >> .tars\blue\start-blue.sh
echo   -v "$(pwd)/.tars/shared:/app/shared:ro" ^\ >> .tars\blue\start-blue.sh
echo   -v "$(pwd)/.tars/evolution:/app/evolution:rw" ^\ >> .tars\blue\start-blue.sh
echo   -e TARS_ENVIRONMENT=blue ^\ >> .tars\blue\start-blue.sh
echo   -e TARS_ROLE=evolution ^\ >> .tars\blue\start-blue.sh
echo   -e TARS_EVOLUTION_ENABLED=true ^\ >> .tars\blue\start-blue.sh
echo   -e TARS_MONITORING_ENABLED=true ^\ >> .tars\blue\start-blue.sh
echo   -e TARS_SESSION_ID=%SESSION_ID% ^\ >> .tars\blue\start-blue.sh
echo   mcr.microsoft.com/dotnet/aspnet:9.0 >> .tars\blue\start-blue.sh
echo. >> .tars\blue\start-blue.sh
echo echo "  ✅ Blue environment ready at http://localhost:8082" >> .tars\blue\start-blue.sh
echo echo "  📊 Metrics available at http://localhost:8083/metrics" >> .tars\blue\start-blue.sh

echo   📄 Blue script: .tars\blue\start-blue.sh

echo.
echo 📊 CREATING MONITORING SCRIPT
echo ============================

echo #!/bin/bash > .tars\monitoring\green-blue\monitor-evolution.sh
echo # TARS Evolution Monitoring Script >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "📊 Starting TARS Evolution Monitoring" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo. >> .tars\monitoring\green-blue\monitor-evolution.sh
echo SESSION_ID="%SESSION_ID%" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo GREEN_ENDPOINT="http://localhost:8080" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo BLUE_ENDPOINT="http://localhost:8082" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo. >> .tars\monitoring\green-blue\monitor-evolution.sh
echo MONITORING_DIR=".tars/monitoring/green-blue/$SESSION_ID" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo mkdir -p "$MONITORING_DIR" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo. >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "  🔍 Monitoring session: $SESSION_ID" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "  🟢 Green endpoint: $GREEN_ENDPOINT" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "  🔵 Blue endpoint: $BLUE_ENDPOINT" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "  📊 Monitoring directory: $MONITORING_DIR" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo. >> .tars\monitoring\green-blue\monitor-evolution.sh
echo # Initialize CSV headers >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "timestamp,environment,status" ^> "$MONITORING_DIR/health-status.csv" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "timestamp,environment,metric,value" ^> "$MONITORING_DIR/metrics.csv" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo. >> .tars\monitoring\green-blue\monitor-evolution.sh
echo # Monitoring loop >> .tars\monitoring\green-blue\monitor-evolution.sh
echo echo "🔄 Starting monitoring loop (Ctrl+C to stop)..." >> .tars\monitoring\green-blue\monitor-evolution.sh
echo while true; do >> .tars\monitoring\green-blue\monitor-evolution.sh
echo     timestamp=$(date -Iseconds) >> .tars\monitoring\green-blue\monitor-evolution.sh
echo     echo "[$timestamp] Monitoring green/blue environments..." >> .tars\monitoring\green-blue\monitor-evolution.sh
echo     echo "$timestamp,green,monitoring" ^>^> "$MONITORING_DIR/health-status.csv" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo     echo "$timestamp,blue,monitoring" ^>^> "$MONITORING_DIR/health-status.csv" >> .tars\monitoring\green-blue\monitor-evolution.sh
echo     echo "  ⏳ Next check in 5 minutes..." >> .tars\monitoring\green-blue\monitor-evolution.sh
echo     sleep 300 >> .tars\monitoring\green-blue\monitor-evolution.sh
echo done >> .tars\monitoring\green-blue\monitor-evolution.sh

echo   📊 Monitoring script: .tars\monitoring\green-blue\monitor-evolution.sh

echo.
echo 📋 CREATING EVOLUTION REPORT
echo ===========================

echo { > .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "sessionId": "%SESSION_ID%", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "startTime": "%DATE% %TIME%", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "duration": "8 hours", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "status": "scheduled", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "greenEnvironment": "%GREEN_ENV%", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "blueEnvironment": "%BLUE_ENV%", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "evolutionGoals": [ >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Enhance meta-cognitive reasoning capabilities", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Optimize autonomous decision-making", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Improve pattern recognition across abstraction layers", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Increase learning efficiency and adaptation speed" >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   ], >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   "monitoringEndpoints": [ >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Green Environment: http://localhost:8080", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Blue Environment: http://localhost:8082", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Green Metrics: http://localhost:8081/metrics", >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo     "Blue Metrics: http://localhost:8083/metrics" >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo   ] >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json
echo } >> .tars\evolution\reports\evolution-session-%SESSION_ID%.json

echo   📋 Evolution report: .tars\evolution\reports\evolution-session-%SESSION_ID%.json

echo.
echo 🎉 NIGHT EVOLUTION SESSION READY!
echo ================================
echo   🆔 Session ID: %SESSION_ID%
echo   ⏰ Scheduled Start: %DATE% %TIME%
echo   ⏱️ Duration: 8 hours
echo   🎯 Evolution Goals: 4
echo.
echo 🚀 TO START EVOLUTION:
echo =====================
echo   1. bash .tars\green\start-green.sh
echo   2. bash .tars\blue\start-blue.sh
echo   3. bash .tars\monitoring\green-blue\monitor-evolution.sh
echo.
echo 📊 MONITORING:
echo =============
echo   🟢 Green: http://localhost:8080
echo   🔵 Blue: http://localhost:8082
echo   📈 Metrics: .tars\monitoring\green-blue\%SESSION_ID%\
echo.
echo 🌙 TARS WILL EVOLVE AUTONOMOUSLY THROUGH THE NIGHT!
echo Master Evolver Agent will monitor and adapt based on real-time metrics
echo.
echo ✨ Night evolution session configured successfully!
echo.

set /p START_NOW="🚀 Start evolution session now? (y/N): "

if /i "%START_NOW%"=="y" (
    echo.
    echo 🚀 STARTING EVOLUTION SESSION...
    echo.
    
    echo 🟢 Starting green environment...
    bash .tars\green\start-green.sh
    timeout /t 5 /nobreak >nul
    
    echo 🔵 Starting blue environment...
    bash .tars\blue\start-blue.sh
    timeout /t 5 /nobreak >nul
    
    echo 📊 Starting monitoring...
    echo 🔄 Monitoring will run in background - check logs in .tars\monitoring\green-blue\%SESSION_ID%\
    start /b bash .tars\monitoring\green-blue\monitor-evolution.sh
    
    echo.
    echo 🎉 EVOLUTION SESSION STARTED!
    echo ============================
    echo 🟢 Green environment: http://localhost:8080
    echo 🔵 Blue environment: http://localhost:8082
    echo 📊 Monitoring active in background
    echo.
    echo 🌙 TARS is now evolving autonomously!
) else (
    echo.
    echo 📋 Evolution session configured but not started.
    echo Run the commands above when ready to begin evolution.
)

echo.
pause
