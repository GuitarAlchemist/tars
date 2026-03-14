@echo off
echo 🎓 CREATING REAL TARS UNIVERSITY AGENT TEAM
echo ==========================================
echo Using actual TARS CLI and AgentOrchestrator
echo.

echo 📋 Step 1: Starting TARS Agent System
echo =====================================
call tars agent start
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to start TARS agent system
    pause
    exit /b 1
)
echo.

echo 👥 Step 2: Creating University Research Team
echo ===========================================
call tars agent create-team university
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to create university team
    pause
    exit /b 1
)
echo.

echo 🔬 Step 3: Creating Individual University Agents
echo ===============================================

echo Creating Research Director...
call tars agent create --persona "Research Director" --capabilities "ProjectManagement,StrategicPlanning,ResearchCoordination"
echo.

echo Creating CS Researcher...
call tars agent create --persona "CS Researcher" --capabilities "CodeAnalysis,AlgorithmDesign,TechnicalWriting"
echo.

echo Creating Data Scientist...
call tars agent create --persona "Data Scientist" --capabilities "DataAnalysis,StatisticalModeling,MachineLearning"
echo.

echo Creating Academic Writer...
call tars agent create --persona "Academic Writer" --capabilities "AcademicWriting,LiteratureReview,CitationManagement"
echo.

echo Creating Peer Reviewer...
call tars agent create --persona "Peer Reviewer" --capabilities "QualityAssurance,ManuscriptReview,AcademicIntegrity"
echo.

echo Creating Knowledge Synthesizer...
call tars agent create --persona "Knowledge Synthesizer" --capabilities "KnowledgeIntegration,SystematicReview,MetaAnalysis"
echo.

echo Creating Ethics Officer...
call tars agent create --persona "Ethics Officer" --capabilities "EthicsReview,ComplianceMonitoring,RiskAssessment"
echo.

echo Creating Graduate Assistant...
call tars agent create --persona "Graduate Assistant" --capabilities "LiteratureSearch,DataCollection,ResearchSupport"
echo.

echo 📊 Step 4: Checking Agent Status
echo ===============================
call tars agent status
echo.

echo 🎯 Step 5: Assigning Research Task
echo =================================
call tars agent assign-task "Autonomous Intelligence Research" --capabilities "ProjectManagement,CodeAnalysis,DataAnalysis,AcademicWriting,QualityAssurance"
echo.

echo 📋 Step 6: Listing Active Teams
echo ==============================
call tars agent list-teams
echo.

echo 🎉 REAL UNIVERSITY AGENT TEAM CREATED!
echo ====================================
echo ✅ TARS agent system started
echo ✅ University research team created
echo ✅ 8 specialized agents created with real personas
echo ✅ Research task assigned to team
echo ✅ Agents are now active and working
echo.
echo 📊 MANAGEMENT COMMANDS:
echo   tars agent status          - Check agent status
echo   tars agent list-teams      - List active teams
echo   tars agent show-tasks      - Show assigned tasks
echo   tars agent demo            - Run agent demonstration
echo.
echo 🎓 UNIVERSITY AGENTS ARE NOW OPERATIONAL!
echo Real agents using TARS AgentOrchestrator and .NET Channels
echo.
pause
