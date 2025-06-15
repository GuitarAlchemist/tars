# TARS Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy project files
COPY ["src/TarsEngine.FSharp.Core/TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj", "TarsEngine.FSharp.Core/"]
COPY ["src/TarsEngine.FSharp.Core/", "TarsEngine.FSharp.Core/"]

# Restore dependencies
RUN dotnet restore "TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj"

# Build application
RUN dotnet build "TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj" -c Release -o /app/build

# Publish application
RUN dotnet publish "TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj" -c Release -o /app/publish

# Runtime stage with CUDA support
FROM nvidia/cuda:v1.0.0-runtime-ubuntu22.04 AS runtime
WORKDIR /app

# Install .NET runtime
RUN apt-get update && apt-get install -y \\
    wget \\
    ca-certificates \\
    && wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb \\
    && dpkg -i packages-microsoft-prod.deb \\
    && apt-get update \\
    && apt-get install -y aspnetcore-runtime-9.0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy published application
COPY --from=build /app/publish .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=all
ENV TARS_3D_VISUALIZATION=true
ENV TARS_AUTO_IMPROVEMENT=true
ENV TARS_ENVIRONMENT=Staging (4, 2)
ENV TARS_FLUX_ENABLED=true

# Configure resource limits
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
ENV DOTNET_RUNNING_IN_CONTAINER=true
ENV ASPNETCORE_URLS=http://+:8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set resource requests and limits (for K8s)
LABEL io.kubernetes.container.requests.cpu="500m"
LABEL io.kubernetes.container.requests.memory="1Gi"
LABEL io.kubernetes.container.limits.cpu="2000m"
LABEL io.kubernetes.container.limits.memory="4Gi"
LABEL io.kubernetes.container.requests.nvidia.com/gpu="1"

# Run TARS
ENTRYPOINT ["dotnet", "TarsEngine.FSharp.Core.dll"]