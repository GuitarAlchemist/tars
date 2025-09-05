FROM python:3.11-slim

LABEL maintainer="TARS Team"
LABEL description="TARS Manager - Gordon Integration Tool"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    docker.io \
    docker-compose \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY tars-manager.py .
COPY tool-definition.json .

# Copy config directory
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV GORDON_API_URL=http://gordon:8997
ENV TARS_API_URL=http://tars-main:8080
ENV LOG_LEVEL=INFO
ENV ANALYSIS_INTERVAL=30

# Expose port
EXPOSE 8998

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8998/api/v1/health || exit 1

# Run the application
CMD ["python", "tars-manager.py"]
