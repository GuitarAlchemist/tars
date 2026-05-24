# TarsSupervisor

A lightweight, local-first supervisor to automate development of your **TARS** project.
It runs an iterative loop (plan → validate → metrics → record) and can roll back to the last successful run.
Works with **.NET 8+** and **Ollama** (local LLM).

## Quick Start
```powershell
cd .\TarsSupervisor\src\TarsSupervisor.Cli
dotnet build
dotnet run -- init --project-root "C:\path\to\tars"
dotnet run -- plan
dotnet run -- iterate
dotnet run -- report
```
