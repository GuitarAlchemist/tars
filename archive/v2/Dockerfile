# TARS v2 - Multi-stage Docker Build
# Supports both CPU and CUDA inference

# ============================================
# Stage 1: Build Stage
# ============================================
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy project files
COPY *.sln ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY grammars/ ./grammars/
COPY docs/ ./docs/

# Restore and build
RUN dotnet restore
RUN dotnet build -c Release --no-restore
RUN dotnet publish src/Tars.Interface.Ui/Tars.Interface.Ui.fsproj -c Release -o /app/publish --no-build

# ============================================
# Stage 2: Runtime Stage (with Ollama)
# ============================================
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# Install curl and other utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy published app
COPY --from=build /app/publish .

# Create directories
RUN mkdir -p /models /root/.ollama

# Expose ports
# 5000 = TARS UI
# 11434 = Ollama API
EXPOSE 5000 11434

# Environment variables
ENV ASPNETCORE_URLS=http://+:5000
ENV DOTNET_RUNNING_IN_CONTAINER=true
ENV TARS_LLM_BACKEND=ollama

# Startup script
COPY scripts/docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
