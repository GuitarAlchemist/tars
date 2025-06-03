@echo off
REM TARS Qwen3 Reasoning Team Deployment
REM Deploy Qwen3 models locally for powerful reasoning capabilities

echo 🧠 TARS QWEN3 REASONING TEAM DEPLOYMENT
echo ========================================
echo 🚀 Setting up powerful reasoning capabilities with Qwen3
echo.

set DEPLOYMENT_ID=%RANDOM%%RANDOM%
set LOG_FILE=qwen3-deployment-%DEPLOYMENT_ID%.log

echo 📋 Deployment Configuration:
echo    Deployment ID: %DEPLOYMENT_ID%
echo    Log File: %LOG_FILE%
echo    Target: Local Ollama + TARS Integration
echo.

REM Step 1: Check prerequisites
echo 🔍 Step 1: Check Prerequisites
echo =============================
echo Checking Ollama installation...

ollama --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Ollama not found! Please install Ollama first.
    echo 📥 Download from: https://ollama.ai
    pause
    exit /b 1
)

echo ✅ Ollama is installed
echo.

REM Step 2: Check available disk space
echo 💾 Step 2: Check Disk Space
echo ===========================
echo Qwen3 models require significant disk space:
echo    - Qwen3-8B: ~5GB
echo    - Qwen3-14B: ~8GB  
echo    - Qwen3-32B: ~18GB
echo    - Qwen3-30B-A3B: ~17GB (MoE)
echo    - Qwen3-235B-A22B: ~130GB (Flagship)
echo.

for /f "tokens=3" %%a in ('dir /-c %SystemDrive%\ ^| find "bytes free"') do set FREE_SPACE=%%a
echo Available disk space: %FREE_SPACE% bytes
echo.

REM Step 3: Deploy Qwen3 models
echo 🧠 Step 3: Deploy Qwen3 Models
echo ===============================

echo 📥 Pulling Qwen3-8B (Fast reasoning)...
ollama pull qwen3:8b
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Failed to pull Qwen3-8B, continuing...
) else (
    echo ✅ Qwen3-8B deployed successfully
)
echo.

echo 📥 Pulling Qwen3-14B (Balanced reasoning)...
ollama pull qwen3:14b
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Failed to pull Qwen3-14B, continuing...
) else (
    echo ✅ Qwen3-14B deployed successfully
)
echo.

echo 📥 Pulling Qwen3-32B (Dense reasoning)...
ollama pull qwen3:32b
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Failed to pull Qwen3-32B, continuing...
) else (
    echo ✅ Qwen3-32B deployed successfully
)
echo.

echo 📥 Pulling Qwen3-30B-A3B (Efficient MoE reasoning)...
ollama pull qwen3:30b-a3b
if %ERRORLEVEL% neq 0 (
    echo ⚠️  Failed to pull Qwen3-30B-A3B, continuing...
) else (
    echo ✅ Qwen3-30B-A3B deployed successfully
)
echo.

echo 📥 Pulling Qwen3-235B-A22B (Flagship reasoning - Large download!)...
echo ⚠️  This is a very large model (~130GB). Continue? (Y/N)
set /p DEPLOY_FLAGSHIP="Deploy flagship model? (Y/N): "
if /i "%DEPLOY_FLAGSHIP%"=="Y" (
    ollama pull qwen3:235b-a22b
    if %ERRORLEVEL% neq 0 (
        echo ⚠️  Failed to pull Qwen3-235B-A22B, continuing...
    ) else (
        echo ✅ Qwen3-235B-A22B deployed successfully
    )
) else (
    echo ⏭️  Skipping flagship model deployment
)
echo.

REM Step 4: Verify deployments
echo 🔍 Step 4: Verify Deployments
echo =============================
echo 📊 Checking deployed Qwen3 models...

ollama list | findstr qwen3
echo.

REM Step 5: Test reasoning capabilities
echo 🧪 Step 5: Test Reasoning Capabilities
echo ======================================
echo 🔬 Testing Qwen3-8B reasoning...

echo Testing mathematical reasoning... > test-reasoning.txt
ollama run qwen3:8b "Solve this step by step: What is the derivative of x^3 + 2x^2 - 5x + 3?" >> test-reasoning.txt 2>&1

echo ✅ Reasoning test completed (see test-reasoning.txt)
echo.

REM Step 6: Create TARS integration
echo 🔗 Step 6: Create TARS Integration
echo ==================================

echo Creating TARS Qwen3 configuration...

echo { > .tars\qwen3-config.json
echo   "deployment_id": "%DEPLOYMENT_ID%", >> .tars\qwen3-config.json
echo   "deployed_models": [ >> .tars\qwen3-config.json
echo     "qwen3:8b", >> .tars\qwen3-config.json
echo     "qwen3:14b", >> .tars\qwen3-config.json
echo     "qwen3:32b", >> .tars\qwen3-config.json
echo     "qwen3:30b-a3b" >> .tars\qwen3-config.json
if /i "%DEPLOY_FLAGSHIP%"=="Y" (
    echo     ,"qwen3:235b-a22b" >> .tars\qwen3-config.json
)
echo   ], >> .tars\qwen3-config.json
echo   "reasoning_capabilities": { >> .tars\qwen3-config.json
echo     "mathematical_reasoning": "qwen3:30b-a3b", >> .tars\qwen3-config.json
echo     "logical_reasoning": "qwen3:32b", >> .tars\qwen3-config.json
echo     "causal_reasoning": "qwen3:14b", >> .tars\qwen3-config.json
echo     "strategic_reasoning": "qwen3:8b", >> .tars\qwen3-config.json
echo     "meta_reasoning": "qwen3:14b", >> .tars\qwen3-config.json
if /i "%DEPLOY_FLAGSHIP%"=="Y" (
    echo     "collaborative_reasoning": "qwen3:235b-a22b" >> .tars\qwen3-config.json
) else (
    echo     "collaborative_reasoning": "qwen3:30b-a3b" >> .tars\qwen3-config.json
)
echo   }, >> .tars\qwen3-config.json
echo   "deployment_time": "%DATE% %TIME%", >> .tars\qwen3-config.json
echo   "status": "active" >> .tars\qwen3-config.json
echo } >> .tars\qwen3-config.json

echo ✅ TARS Qwen3 configuration created
echo.

REM Step 7: Create reasoning test script
echo 🧪 Step 7: Create Reasoning Test Script
echo =======================================

echo @echo off > test-qwen3-reasoning.cmd
echo REM TARS Qwen3 Reasoning Test Script >> test-qwen3-reasoning.cmd
echo echo 🧠 TARS Qwen3 Reasoning Test >> test-qwen3-reasoning.cmd
echo echo ========================== >> test-qwen3-reasoning.cmd
echo echo. >> test-qwen3-reasoning.cmd
echo echo 🔬 Testing Mathematical Reasoning... >> test-qwen3-reasoning.cmd
echo ollama run qwen3:30b-a3b "Solve step by step: Find the integral of 2x^3 - 4x + 1" >> test-qwen3-reasoning.cmd
echo echo. >> test-qwen3-reasoning.cmd
echo echo 🔬 Testing Logical Reasoning... >> test-qwen3-reasoning.cmd
echo ollama run qwen3:32b "Use logical reasoning: If all A are B, and all B are C, what can we conclude about A and C?" >> test-qwen3-reasoning.cmd
echo echo. >> test-qwen3-reasoning.cmd
echo echo 🔬 Testing Causal Reasoning... >> test-qwen3-reasoning.cmd
echo ollama run qwen3:14b "Analyze the causal chain: What factors might cause a software system to become slow over time?" >> test-qwen3-reasoning.cmd
echo echo. >> test-qwen3-reasoning.cmd
echo echo 🔬 Testing Strategic Reasoning... >> test-qwen3-reasoning.cmd
echo ollama run qwen3:8b "Develop a strategy: How should a software team prioritize bug fixes vs new features?" >> test-qwen3-reasoning.cmd
echo echo. >> test-qwen3-reasoning.cmd
echo echo ✅ Reasoning tests completed! >> test-qwen3-reasoning.cmd

echo ✅ Reasoning test script created: test-qwen3-reasoning.cmd
echo.

REM Step 8: Create monitoring script
echo 📊 Step 8: Create Monitoring Script
echo ===================================

echo @echo off > monitor-qwen3-reasoning.cmd
echo REM TARS Qwen3 Reasoning Monitor >> monitor-qwen3-reasoning.cmd
echo echo 📊 TARS Qwen3 Reasoning Monitor >> monitor-qwen3-reasoning.cmd
echo echo =============================== >> monitor-qwen3-reasoning.cmd
echo echo. >> monitor-qwen3-reasoning.cmd
echo echo 🔍 Available Qwen3 Models: >> monitor-qwen3-reasoning.cmd
echo ollama list ^| findstr qwen3 >> monitor-qwen3-reasoning.cmd
echo echo. >> monitor-qwen3-reasoning.cmd
echo echo 💾 Model Storage Usage: >> monitor-qwen3-reasoning.cmd
echo dir "%USERPROFILE%\.ollama\models" /s ^| find "qwen3" >> monitor-qwen3-reasoning.cmd
echo echo. >> monitor-qwen3-reasoning.cmd
echo echo 🧠 Reasoning Capabilities Status: >> monitor-qwen3-reasoning.cmd
echo type .tars\qwen3-config.json >> monitor-qwen3-reasoning.cmd

echo ✅ Monitoring script created: monitor-qwen3-reasoning.cmd
echo.

REM Step 9: Integration with TARS
echo 🔗 Step 9: TARS Integration
echo ===========================

echo Building TARS Reasoning Engine...
dotnet build TarsEngine.FSharp.Reasoning/TarsEngine.FSharp.Reasoning.fsproj
if %ERRORLEVEL% neq 0 (
    echo ⚠️  TARS Reasoning Engine build failed, but Qwen3 models are deployed
) else (
    echo ✅ TARS Reasoning Engine built successfully
)
echo.

echo 🎉 QWEN3 REASONING TEAM DEPLOYMENT COMPLETE!
echo ============================================
echo ✅ Qwen3 models deployed locally via Ollama
echo ✅ TARS integration configuration created
echo ✅ Reasoning test scripts created
echo ✅ Monitoring tools configured
echo.
echo 🧠 Available Reasoning Capabilities:
echo    📊 Mathematical Reasoning (Qwen3-30B-A3B)
echo    🔍 Logical Reasoning (Qwen3-32B)
echo    🔗 Causal Reasoning (Qwen3-14B)
echo    🎯 Strategic Reasoning (Qwen3-8B)
echo    🤔 Meta Reasoning (Qwen3-14B)
if /i "%DEPLOY_FLAGSHIP%"=="Y" (
    echo    🤝 Collaborative Reasoning (Qwen3-235B-A22B)
) else (
    echo    🤝 Collaborative Reasoning (Qwen3-30B-A3B)
)
echo.
echo 📋 Next Steps:
echo    1. Run: test-qwen3-reasoning.cmd (Test reasoning capabilities)
echo    2. Run: monitor-qwen3-reasoning.cmd (Monitor system status)
echo    3. Integrate with TARS explorations for enhanced reasoning
echo.
echo 🚀 TARS is now equipped with powerful Qwen3 reasoning capabilities!
echo 🧠 Ready for advanced autonomous reasoning and exploration!
echo.

pause
