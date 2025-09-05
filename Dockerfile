# TARS Unified Architecture - Production Docker Container
# Complete unified system with CUDA acceleration, caching, monitoring, and proof generation
# TARS_DOCKER_SIGNATURE: UNIFIED_ARCHITECTURE_V2

# Use .NET 9 runtime as base
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

# Use .NET 9 SDK for building
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy all project files for unified architecture
COPY ["TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj", "TarsEngine.FSharp.Cli/"]
COPY ["TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj", "TarsEngine.FSharp.Core/"]
COPY ["TarsEngine.FSharp.Tests/TarsEngine.FSharp.Tests.fsproj", "TarsEngine.FSharp.Tests/"]
COPY ["TarsEngine.CustomTransformers/TarsEngine.CustomTransformers.fsproj", "TarsEngine.CustomTransformers/"]
COPY ["Tars.Engine.Grammar/Tars.Engine.Grammar.fsproj", "Tars.Engine.Grammar/"]
COPY ["Tars.Engine.VectorStore/Tars.Engine.VectorStore.fsproj", "Tars.Engine.VectorStore/"]

# Restore dependencies for all projects
RUN dotnet restore "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

# Copy source code
COPY . .

# Build the unified architecture
WORKDIR "/src/TarsEngine.FSharp.Cli"
RUN dotnet build "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/build

# Run comprehensive tests to ensure quality
WORKDIR "/src"
RUN dotnet test "TarsEngine.FSharp.Tests/TarsEngine.FSharp.Tests.fsproj" -c Release --no-build --verbosity normal

# Publish the application
FROM build AS publish
WORKDIR "/src/TarsEngine.FSharp.Cli"
RUN dotnet publish "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/publish /p:UseAppHost=false

# Final runtime image
FROM base AS final
WORKDIR /app

# Install tools for unified architecture operations
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    nano \
    htop \
    procps \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime for GPU acceleration (optional)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy published application
COPY --from=publish /app/publish .

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create unified architecture directories
RUN mkdir -p /app/.tars/projects \
    && mkdir -p /app/.tars/metascripts \
    && mkdir -p /app/.tars/swarm \
    && mkdir -p /app/.tars/experiments \
    && mkdir -p /app/.tars/self-modification \
    && mkdir -p /app/data/cache \
    && mkdir -p /app/data/logs \
    && mkdir -p /app/data/config \
    && mkdir -p /app/data/proofs \
    && mkdir -p /app/data/monitoring \
    && mkdir -p /app/logs

# Copy TARS metascripts
COPY .tars/ /app/.tars/

# Environment variables for unified architecture
ENV TARS_ENVIRONMENT=Docker
ENV TARS_MODE=Unified
ENV TARS_SWARM_ENABLED=true
ENV TARS_SELF_MODIFICATION_ENABLED=true
ENV TARS_EXPERIMENTS_ENABLED=true
ENV TARS_CACHE_ENABLED=true
ENV TARS_MONITORING_ENABLED=true
ENV TARS_PROOF_GENERATION_ENABLED=true
ENV TARS_CUDA_ENABLED=true
ENV TARS_DATA_PATH=/app/data
ENV TARS_CACHE_PATH=/app/data/cache
ENV TARS_LOGS_PATH=/app/data/logs
ENV TARS_CONFIG_PATH=/app/data/config
ENV TARS_PROOFS_PATH=/app/data/proofs
ENV DOTNET_ENVIRONMENT=Production
ENV DOTNET_RUNNING_IN_CONTAINER=true

# Health check using unified diagnostics
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD dotnet TarsEngine.FSharp.Cli.dll diagnose --status || exit 1

# Default entrypoint - stable daemon mode
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
