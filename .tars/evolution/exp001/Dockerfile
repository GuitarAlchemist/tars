# TARS Evolution Container - Version 2.2.0
# Session: exp001
# Mode: experimental
# Created: 2025-05-31T23:07:01.831506

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build

# Container metadata
LABEL tars.version="2.2.0"
LABEL tars.evolution.session="exp001"
LABEL tars.evolution.mode="experimental"
LABEL tars.evolution.parent="2.1.0"

# Install evolution dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip nodejs npm git curl wget jq htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for TARS evolution
RUN pip3 install aiohttp asyncio requests numpy pandas

# Set up TARS evolution workspace
WORKDIR /tars

# Create evolution runtime image
FROM mcr.microsoft.com/dotnet/runtime:8.0

# Copy evolution metadata
LABEL tars.version="2.2.0"
LABEL tars.evolution.session="exp001"
LABEL tars.evolution.mode="experimental"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip nodejs npm git curl \
    && rm -rf /var/lib/apt/lists/*

# Set evolution environment
ENV TARS_VERSION="2.2.0"
ENV TARS_EVOLUTION_SESSION="exp001"
ENV TARS_EVOLUTION_MODE="experimental"
ENV TARS_PARENT_VERSION="2.1.0"
ENV TARS_CONTAINER_NAME="tars-evolution-v2.2.0-20250531-230701-exp001"
ENV TARS_DOCKER_ISOLATED="true"

# Create evolution directories
RUN mkdir -p /tars/evolution /tars/shared /tars/monitoring /tars/backups

# Copy evolution scripts
COPY evolution-scripts/ ./evolution-scripts/
RUN chmod +x ./evolution-scripts/*.sh

# Health check for evolution session
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start TARS evolution
ENTRYPOINT ["./evolution-scripts/start-evolution.sh"]
CMD ["--version", "2.2.0", "--session", "exp001"]
