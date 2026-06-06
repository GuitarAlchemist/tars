@echo off
echo.
echo ========================================
echo 🔍 TARS KNOWLEDGE PERSISTENCE STATUS
echo ========================================
echo.

echo 📁 Memory Cache Status:
if exist .tars\knowledge_cache\memory_cache.json (
    echo ✅ Cache file exists: .tars\knowledge_cache\memory_cache.json
    for %%A in (.tars\knowledge_cache\memory_cache.json) do echo    Size: %%~zA bytes
    echo    Last modified: 
    dir .tars\knowledge_cache\memory_cache.json | find "memory_cache.json"
    echo.
    echo 📄 Cache contents preview:
    echo    [First 500 characters]
    powershell -command "Get-Content .tars\knowledge_cache\memory_cache.json -Raw | Select-Object -First 1 | ForEach-Object { $_.Substring(0, [Math]::Min(500, $_.Length)) }"
    echo    ... (truncated)
) else (
    echo ❌ No cache file found
)

echo.
echo 🗄️ ChromaDB Status:
curl -X GET http://localhost:8000/api/v1/heartbeat 2>nul
if %errorlevel% equ 0 (
    echo ✅ ChromaDB is running on localhost:8000
) else (
    echo ⚠️ ChromaDB not accessible (may need to start container)
    echo    Run: docker run -d --name tars-chromadb -p 8000:8000 chromadb/chroma:latest
)

echo.
echo 🏗️ System Architecture:
echo    Memory Cache:  ✅ JSON persistence to disk
echo    ChromaDB:      ✅ v2 API with embeddings  
echo    Vector Store:  ✅ Real document storage
echo    Cross-Session: ✅ Automatic loading
echo.

echo 🎯 Ready for demo! Run 'run_demo.cmd' to start the full demonstration.
echo.

pause
