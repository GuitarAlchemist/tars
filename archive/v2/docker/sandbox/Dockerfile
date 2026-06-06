FROM python:3.11-slim

# Create a non-root user
RUN useradd -m -s /bin/bash tars

# Set working directory
WORKDIR /home/tars/workspace

# Switch to non-root user
USER tars

# Define volume for mounting scripts
VOLUME /home/tars/workspace

# Default command
CMD ["python3", "--version"]
