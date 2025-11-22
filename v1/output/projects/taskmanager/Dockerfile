# TaskManager - Multi-stage Docker Build

# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy project files
COPY ["src/TaskManager.Api/TaskManager.Api.fsproj", "src/TaskManager.Api/"]
COPY ["src/TaskManager.Core/TaskManager.Core.fsproj", "src/TaskManager.Core/"]

# Restore dependencies
RUN dotnet restore "src/TaskManager.Api/TaskManager.Api.fsproj"

# Copy source code
COPY . .

# Build application
WORKDIR "/src/src/TaskManager.Api"
RUN dotnet build "TaskManager.Api.fsproj" -c Release -o /app/build

# Publish stage
FROM build AS publish
RUN dotnet publish "TaskManager.Api.fsproj" -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Copy published application
COPY --from=publish /app/publish .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:80/health || exit 1

# Expose port
EXPOSE 80

# Start application
ENTRYPOINT ["dotnet", "TaskManager.Api.dll"]
