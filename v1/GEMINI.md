# TARS v1 - Project Documentation

## Project Overview

TARS (The automated Reasoning System & AI Inference Engine) is a high-performance, F#-based AI inference engine with advanced capabilities for automated reasoning and self-improvement. The project's ambitious goal is to achieve "superintelligence" through a system of self-improving agents.

### Key Features:

*   **High-Performance Inference:** A production-ready, Ollama-compatible AI inference engine optimized for speed and low memory usage, with CUDA acceleration.
*   **Autonomous Reasoning:** A sophisticated multi-agent system that can execute complex tasks defined in structured "metascript" files.
*   **Self-Improvement:** TARS is designed to analyze and improve its own code, with a strong focus on quality and performance.
*   **Metascript DSL:** A domain-specific language for defining AI workflows, allowing for complex orchestration of the system's capabilities.
*   **F#-first Architecture:** The core of TARS is built with F#, leveraging functional programming for reliability and performance. Legacy components are in C#.

## Building and Running

### Docker (Recommended)

The TARS AI inference engine can be run in a Docker container with GPU support.

1.  **Build the Docker image:**
    ```bash
    docker build -f Dockerfile.ai -t tars-ai:latest .
    ```

2.  **Run the container:**
    ```bash
    docker run -d -p 11434:11434 --gpus most tars-ai:latest
    ```

3.  **Test the API:**
    ```bash
    curl -X POST http://localhost:11434/api/generate \
         -H "Content-Type: application/json" \
         -d '{"model":"tars-medium-7b","prompt":"Hello TARS!"}'
    ```

### Kubernetes

A Kubernetes deployment is also available.

```bash
kubectl apply -f k8s/tars-ai-deployment.yaml
```

### Local Development

1.  **Build the project:**
    ```bash
    dotnet build TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj
    ```

2.  **Run the API server:**
    ```bash
    dotnet run --project TarsEngine.FSharp.Cli -- --server --port 11434
    ```

3.  **Use the CLI:**
    The TARS CLI provides a wide range of commands for interacting with the system. Here are a few examples:

    *   **Start the interactive chatbot:**
        ```bash
        dotnet run --project TarsEngine.FSharp.Cli -- chatbot
        ```
    *   **Run a self-improvement cycle:**
        ```bash
        dotnet run --project TarsCli/TarsCli.csproj -- self-improve cycle path/to/project --max-files 10 --backup
        ```
    *   **Execute a metascript:**
        ```bash
        dotnet run --project TarsCli -- execute <script.tars>
        ```

## Development Conventions

*   **F#-first:** New development should be done in F#. C# is used for infrastructure and integrations.
*   **Conventional Commits:** The project uses the Conventional Commits specification. Commits made by the autonomous system are prefixed with `auto:`.
*   **High Quality Standards:** There is a "Zero Tolerance Policy" for placeholders or fake implementations. A minimum of 80% test coverage is required.
*   **Metascripts:** The `TARS_INSTRUCTION_FORMAT_SPECIFICATION.md` file defines a structured Markdown format for providing instructions to the TARS autonomous system. These `.tars.md` files are used to define tasks with clear objectives, workflows, and validation steps.

## Project Structure

The project has a complex structure with many sub-projects. The main components are:

*   **`src/`:** Contains the core F# projects, including `TarsEngine.FSharp.Core`, `TarsEngine.FSharp.Cli`, and `TarsEngine.FSharp.Metascript`.
*   **`parked_legacy/`:** A large directory containing archived C# projects for reference.
*   **`v2/`:** An empty directory intended for the next version of TARS.
*   **Dockerfiles:** Numerous Dockerfiles for building different components of the system.
*   **Documentation:** A large number of Markdown files documenting the project's architecture, features, and evolution.
