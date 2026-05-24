@echo off
echo.
echo ========================================
echo   TARS SPECTRE CONSOLE WIDGETS DEMO
echo ========================================
echo.
echo This demo showcases spectacular Spectre.Console widgets:
echo   * Live Progress Bars
echo   * Interactive Tables  
echo   * Real-time Charts
echo   * Tree Views
echo   * Status Displays
echo   * Figlet Text
echo.

cd /d "%~dp0"

echo Building TARS CLI...
dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj --verbosity quiet

if %ERRORLEVEL% NEQ 0 (
    echo Build failed! Let's run a simulated demo instead...
    echo.
    goto :simulate_demo
)

echo Build successful! Running real Spectre.Console widgets demo...
echo.

REM Try to run the actual command
dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj -- swarm demo
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CLI execution failed, showing simulated widgets demo...
    goto :simulate_demo
)

goto :end

:simulate_demo
echo.
echo ========================================
echo   SIMULATED SPECTRE WIDGETS SHOWCASE
echo ========================================
echo.

echo [1] FIGLET TEXT DISPLAY:
echo.
echo  ████████╗ █████╗ ██████╗ ███████╗
echo  ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝
echo     ██║   ███████║██████╔╝███████╗
echo     ██║   ██╔══██║██╔══██╗╚════██║
echo     ██║   ██║  ██║██║  ██║███████║
echo     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
echo.
echo   Mixtral MoE Processing Engine
echo.

timeout /t 2 /nobreak > nul

echo [2] PROGRESS BARS - Live Data Processing:
echo.
echo GitHub Trending    [████████████████████] 100%% Complete
echo Hacker News        [██████████████████▒▒] 90%% Processing...  
echo Crypto Markets     [████████████▒▒▒▒▒▒▒▒] 75%% Analyzing...
echo Stack Overflow     [██████▒▒▒▒▒▒▒▒▒▒▒▒▒▒] 45%% Fetching...
echo Reddit Tech        [███▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒] 20%% Starting...
echo.

timeout /t 2 /nobreak > nul

echo [3] EXPERT STATUS TABLE:
echo.
echo ┌─────────────────┬─────────────┬────────────┬─────────────┐
echo │ Expert          │ Status      │ Confidence │ Tasks       │
echo ├─────────────────┼─────────────┼────────────┼─────────────┤
echo │ CodeGeneration  │ 🟢 Active   │ 0.92       │ 15 queued   │
echo │ CodeAnalysis    │ 🟢 Active   │ 0.88       │ 12 queued   │
echo │ Architecture    │ 🟡 Busy     │ 0.85       │ 8 queued    │
echo │ Testing         │ 🟢 Active   │ 0.91       │ 10 queued   │
echo │ Security        │ 🔴 Overload │ 0.79       │ 20 queued   │
echo │ Performance     │ 🟢 Active   │ 0.94       │ 7 queued    │
echo │ DevOps          │ 🟡 Busy     │ 0.87       │ 9 queued    │
echo │ Documentation   │ 🟢 Active   │ 0.83       │ 5 queued    │
echo └─────────────────┴─────────────┴────────────┴─────────────┘
echo.

timeout /t 2 /nobreak > nul

echo [4] TREE VIEW - System Architecture:
echo.
echo 🧠 TARS Mixtral MoE System
echo ├── 👥 Expert Network
echo │   ├── 🔧 CodeGeneration Expert - Active
echo │   ├── 🔍 CodeAnalysis Expert - Active  
echo │   ├── 🏗️ Architecture Expert - Busy
echo │   ├── 🧪 Testing Expert - Active
echo │   └── 🛡️ Security Expert - Overloaded
echo ├── 🧭 Intelligent Routing
echo │   ├── 📊 Query Analysis Engine
echo │   ├── 🎯 Expert Selection Algorithm
echo │   └── ⚖️ Load Balancing System
echo └── 📡 Live Data Sources
echo     ├── 🐙 GitHub API - Connected
echo     ├── 📰 Hacker News API - Connected
echo     ├── 💰 Crypto APIs - Connected
echo     └── ❓ Stack Overflow API - Connected
echo.

timeout /t 2 /nobreak > nul

echo [5] BAR CHART - Expert Workload Distribution:
echo.
echo CodeGeneration  ████████████████████████████ 25%%
echo CodeAnalysis   ████████████████████████     20%%
echo Architecture   ██████████████████           15%%
echo Testing        █████████████████████        18%%
echo Security       ██████████████               12%%
echo Performance    ████████████                 10%%
echo.

timeout /t 2 /nobreak > nul

echo [6] LIVE STATUS PANEL:
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║                    🚀 LIVE PROCESSING                    ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║ Current Task: Analyzing GitHub trending repositories     ║
echo ║ Expert: CodeAnalysis                                     ║
echo ║ Confidence: 0.94                                         ║
echo ║ Progress: ████████████████████████████████████ 85%%      ║
echo ║                                                          ║
echo ║ 🔍 Analyzing data patterns...                            ║
echo ║ 🧠 Routing to expert: CodeAnalysis                       ║
echo ║ ⚡ Processing with Mixtral MoE...                        ║
echo ║ 📊 Generating insights...                                ║
echo ║ ▶ Analysis in progress...                                ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

timeout /t 2 /nobreak > nul

echo [7] RESULTS PANEL:
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║                  🎯 AI ANALYSIS RESULTS                  ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║ Repository: awesome-rust-tools                           ║
echo ║ Expert Analysis: CodeAnalysis                            ║
echo ║                                                          ║
echo ║ Key Insights:                                            ║
echo ║ • 🚀 Emerging trend: Rust-based tools gaining momentum   ║
echo ║ • 📈 Performance improvements of 40%% over alternatives   ║
echo ║ • 🔧 Developer productivity focus with clean APIs        ║
echo ║ • 🌟 Strong community adoption and contribution          ║
echo ║ • 🛡️ Memory safety advantages clearly demonstrated       ║
echo ║                                                          ║
echo ║ Recommendation: High potential for enterprise adoption   ║
echo ║ Confidence Score: 0.94                                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

timeout /t 2 /nobreak > nul

echo [8] STATISTICS DASHBOARD:
echo.
echo ┌─────────────────────────────────────────────────────────┐
echo │                  📊 PERFORMANCE METRICS                 │
echo ├─────────────────────────────────────────────────────────┤
echo │ Total Queries Processed:     1,247                     │
echo │ Success Rate:                94.2%%                     │
echo │ Average Response Time:       1.3 seconds               │
echo │ Active Experts:              8/10                      │
echo │ Peak Throughput:             45 queries/minute         │
echo │ Cache Hit Rate:              78.5%%                     │
echo │ Error Rate:                  5.8%%                      │
echo │ Uptime:                      99.7%% (last 30 days)      │
echo └─────────────────────────────────────────────────────────┘
echo.

timeout /t 2 /nobreak > nul

echo [9] CALENDAR VIEW - Processing Activity:
echo.
echo              December 2024
echo     Su Mo Tu We Th Fr Sa
echo      1  2  3  4  5  6  7
echo      8  9 10 11 12 13 14
echo     15 16 17 18 19 20 21
echo     22 23 24 25 26 27 28
echo     29 30 31
echo.
echo     🟢 High Activity    🟡 Medium Activity    🔴 Peak Load
echo.

echo ========================================
echo        🎉 SPECTRE WIDGETS SHOWCASE
echo ========================================
echo.
echo Demonstrated Spectre.Console Features:
echo   ✓ Figlet Text with ASCII art
echo   ✓ Progress Bars with real-time updates
echo   ✓ Tables with borders and styling
echo   ✓ Tree Views for hierarchical data
echo   ✓ Bar Charts for data visualization
echo   ✓ Panels with borders and headers
echo   ✓ Status displays with live updates
echo   ✓ Calendar widgets for activity tracking
echo   ✓ Color coding and emoji support
echo   ✓ Layout management and positioning
echo.
echo 🚀 Ready to integrate with real TARS Mixtral MoE system!
echo.

:end
echo Press any key to exit...
pause > nul
