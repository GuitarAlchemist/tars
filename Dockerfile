# TARS Autonomous System - Docker Container
# Enables safe self-modification, swarm deployment, and heavy experiments
# TARS_DOCKER_SIGNATURE: AUTONOMOUS_CONTAINER_V1

# Use .NET 9 runtime as base
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

# Use .NET 9 SDK for building
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy project files
COPY ["TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj", "TarsEngine.FSharp.Cli/"]

# Restore dependencies
RUN dotnet restore "TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj"

# Copy source code
COPY . .

# Build the application
WORKDIR "/src/TarsEngine.FSharp.Cli"
RUN dotnet build "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/build

# Publish the application
FROM build AS publish
RUN dotnet publish "TarsEngine.FSharp.Cli.fsproj" -c Release -o /app/publish /p:UseAppHost=false

# Final runtime image
FROM base AS final
WORKDIR /app

# Install additional tools for autonomous operations
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    nano \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy published application
COPY --from=publish /app/publish .

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create TARS directories
RUN mkdir -p /app/.tars/projects
RUN mkdir -p /app/.tars/metascripts
RUN mkdir -p /app/.tars/swarm
RUN mkdir -p /app/.tars/experiments
RUN mkdir -p /app/.tars/self-modification
RUN mkdir -p /app/logs

# Copy TARS metascripts
COPY .tars/ /app/.tars/

# Environment variables for TARS
ENV TARS_ENVIRONMENT=Docker
ENV TARS_MODE=Autonomous
ENV TARS_SWARM_ENABLED=true
ENV TARS_SELF_MODIFICATION_ENABLED=true
ENV TARS_EXPERIMENTS_ENABLED=true
ENV DOTNET_ENVIRONMENT=Production

# Health check (disabled for now since we don't have HTTP endpoints yet)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8080/health || exit 1

# Default entrypoint - stable daemon mode
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
