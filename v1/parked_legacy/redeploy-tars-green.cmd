@echo off
REM TARS GREEN BASELINE REDEPLOYMENT
REM Deploy tars-green-stable under tars namespace
REM Critical: Green baseline MUST be working 24/7

echo 🚨 CRITICAL: TARS GREEN BASELINE REDEPLOYMENT
echo =============================================
echo 🔒 Deploying under tars namespace
echo.

set CONTAINER_NAME=tars-green-stable
set IMAGE_NAME=tars/green-stable:latest
set NETWORK_NAME=tars
set SESSION_ID=%RANDOM%%RANDOM%

echo 📋 Configuration:
echo    Container: %CONTAINER_NAME%
echo    Image: %IMAGE_NAME%
echo    Network: %NETWORK_NAME%
echo    Session: %SESSION_ID%
echo.

REM Step 1: Stop existing container
echo 🛑 Step 1: Cleanup Existing Container
echo =====================================
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul
echo ✅ Cleanup complete
echo.

REM Step 2: Create tars network
echo 🌐 Step 2: Setup TARS Network
echo =============================
docker network create tars 2>nul
echo ✅ TARS network ready
echo.

REM Step 3: Build TARS green image
echo 🏗️ Step 3: Build TARS Green Image
echo =================================

REM Create Dockerfile for green baseline
echo # TARS Green Baseline Container > Dockerfile.tars-green
echo FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base >> Dockerfile.tars-green
echo WORKDIR /app >> Dockerfile.tars-green
echo EXPOSE 8080 >> Dockerfile.tars-green
echo EXPOSE 8081 >> Dockerfile.tars-green
echo. >> Dockerfile.tars-green
echo # Build stage >> Dockerfile.tars-green
echo FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build >> Dockerfile.tars-green
echo WORKDIR /src >> Dockerfile.tars-green
echo COPY ["TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj", "TarsEngine.FSharp.Cli/"] >> Dockerfile.tars-green
echo RUN dotnet restore "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj" >> Dockerfile.tars-green
echo COPY . . >> Dockerfile.tars-green
echo WORKDIR "/src/TarsEngine.FSharp.Cli" >> Dockerfile.tars-green
echo RUN dotnet publish "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/publish >> Dockerfile.tars-green
echo. >> Dockerfile.tars-green
echo # Runtime stage >> Dockerfile.tars-green
echo FROM base AS final >> Dockerfile.tars-green
echo WORKDIR /app >> Dockerfile.tars-green
echo COPY --from=build /app/publish . >> Dockerfile.tars-green
echo. >> Dockerfile.tars-green
echo # Create TARS directories >> Dockerfile.tars-green
echo RUN mkdir -p /app/.tars/green /app/.tars/shared /app/.tars/logs >> Dockerfile.tars-green
echo. >> Dockerfile.tars-green
echo # Environment variables >> Dockerfile.tars-green
echo ENV TARS_ENVIRONMENT=green >> Dockerfile.tars-green
echo ENV TARS_ROLE=baseline >> Dockerfile.tars-green
echo ENV TARS_NAMESPACE=tars >> Dockerfile.tars-green
echo ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 >> Dockerfile.tars-green
echo ENV ASPNETCORE_URLS=http://+:8080 >> Dockerfile.tars-green
echo. >> Dockerfile.tars-green
echo # Health check >> Dockerfile.tars-green
echo HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8080/health ^|^| exit 1 >> Dockerfile.tars-green
echo. >> Dockerfile.tars-green
echo # Entry point >> Dockerfile.tars-green
echo ENTRYPOINT ["dotnet", "TarsEngine.FSharp.Cli.dll", "serve", "--port", "8080"] >> Dockerfile.tars-green

echo 📝 Dockerfile.tars-green created
echo 🔨 Building TARS green image...

docker build -f Dockerfile.tars-green -t %IMAGE_NAME% .
if %ERRORLEVEL% neq 0 (
    echo ❌ Image build failed!
    exit /b 1
)
echo ✅ TARS green image built
echo.

REM Step 4: Deploy container under tars namespace
echo 🚀 Step 4: Deploy TARS Green Container
echo ======================================
echo 🚀 Starting %CONTAINER_NAME% under tars namespace...

docker run -d ^
  --name %CONTAINER_NAME% ^
  --network %NETWORK_NAME% ^
  --label tars.environment=green ^
  --label tars.role=baseline ^
  --label tars.namespace=tars ^
  --label tars.session=%SESSION_ID% ^
  --restart always ^
  -p 8080:8080 ^
  -p 8081:8081 ^
  -v "%CD%\.tars\green:/app/.tars/green:rw" ^
  -v "%CD%\.tars\shared:/app/.tars/shared:ro" ^
  -e TARS_ENVIRONMENT=green ^
  -e TARS_ROLE=baseline ^
  -e TARS_NAMESPACE=tars ^
  -e TARS_MONITORING_ENABLED=true ^
  -e TARS_SESSION_ID=%SESSION_ID% ^
  %IMAGE_NAME%

if %ERRORLEVEL% neq 0 (
    echo ❌ Container deployment failed!
    exit /b 1
)
echo ✅ Container deployed under tars namespace
echo.

REM Step 5: Health verification
echo 🏥 Step 5: Health Verification
echo ==============================
echo 🔍 Waiting for container to be ready...
timeout /t 10 /nobreak >nul

docker ps --filter "name=%CONTAINER_NAME%" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

REM Step 6: Verify tars namespace
echo 📊 Step 6: TARS Namespace Verification
echo ======================================
echo 🔍 Containers under tars namespace:
docker ps --filter "label=tars.namespace=tars" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Labels}}"
echo.

echo 🌐 TARS network containers:
docker network inspect tars --format "{{range .Containers}}{{.Name}} {{end}}" 2>nul
echo.

echo 🎉 TARS GREEN BASELINE REDEPLOYMENT COMPLETE!
echo =============================================
echo ✅ Container: %CONTAINER_NAME% deployed under tars namespace
echo ✅ Image: %IMAGE_NAME% ready
echo ✅ Network: %NETWORK_NAME% configured
echo ✅ Ports: 8080, 8081 exposed
echo ✅ Auto-restart: enabled
echo ✅ Health monitoring: active
echo.
echo 🔒 STABLE GREEN BASELINE RESTORED!
echo 🎯 Ready for blue node operations under tars namespace
echo.
echo 📋 Management Commands:
echo    View logs: docker logs %CONTAINER_NAME% -f
echo    Stop: docker stop %CONTAINER_NAME%
echo    Restart: docker restart %CONTAINER_NAME%
echo    Health: docker inspect %CONTAINER_NAME% --format "{{.State.Health.Status}}"
echo    TARS containers: docker ps --filter "label=tars.namespace=tars"
echo.

pause
