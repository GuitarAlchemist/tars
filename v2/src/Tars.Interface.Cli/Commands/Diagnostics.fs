/// <summary>
/// TARS Diagnostics - System connectivity and configuration checks
/// </summary>
module Tars.Interface.Cli.Commands.Diagnostics

open System
open System.IO
open System.Diagnostics
open System.Net.Http
open System.Runtime.InteropServices
open System.Threading.Tasks
open Serilog
open Tars.Security
open Tars.Llm.OllamaClient

/// <summary>Diagnostic result for a single check</summary>
type DiagnosticResult =
    { Name: string
      Status: string
      Details: string
      IsOk: bool }

/// <summary>Console output helpers for diagnostics</summary>
module Console =
    let private supportsUnicode () =
        let asciiEnv = System.Environment.GetEnvironmentVariable("TARS_ASCII")
        let cp = System.Console.OutputEncoding.CodePage
        let unicodeCapable = cp = 65001 || cp = 1200 || cp = 28591

        let redirected =
            System.Console.IsOutputRedirected || System.Console.IsErrorRedirected

        unicodeCapable && System.String.IsNullOrWhiteSpace(asciiEnv) && not redirected

    let writeColor (color: ConsoleColor) (text: string) =
        let prev = System.Console.ForegroundColor
        System.Console.ForegroundColor <- color
        System.Console.Write(text)
        System.Console.ForegroundColor <- prev

    let header (text: string) =
        let useUnicode = supportsUnicode ()
        System.Console.WriteLine()

        if useUnicode then
            writeColor ConsoleColor.Cyan "╔════════════════════════════════════════════════════════════╗"
            System.Console.WriteLine()
            writeColor ConsoleColor.Cyan "║  "
            writeColor ConsoleColor.White text
            let padding = 58 - text.Length
            System.Console.Write(String.replicate padding " ")
            writeColor ConsoleColor.Cyan "║"
            System.Console.WriteLine()
            writeColor ConsoleColor.Cyan "╚════════════════════════════════════════════════════════════╝"
        else
            let line = "+----------------------------------------------------------+"
            writeColor ConsoleColor.Cyan line
            System.Console.WriteLine()
            writeColor ConsoleColor.Cyan "| "
            writeColor ConsoleColor.White text
            let padding = (line.Length - 3) - text.Length
            System.Console.Write(String.replicate padding " ")
            writeColor ConsoleColor.Cyan "|"
            System.Console.WriteLine()
            writeColor ConsoleColor.Cyan line

        System.Console.WriteLine()

    let checkOk (name: string) (details: string) =
        let prefix = if supportsUnicode () then "  ✓ " else "  [OK] "
        writeColor ConsoleColor.Green prefix
        writeColor ConsoleColor.White $"{name}: "
        writeColor ConsoleColor.Gray details
        System.Console.WriteLine()

    let checkFail (name: string) (details: string) =
        let prefix = if supportsUnicode () then "  ✗ " else "  [X] "
        writeColor ConsoleColor.Red prefix
        writeColor ConsoleColor.White $"{name}: "
        writeColor ConsoleColor.DarkGray details
        System.Console.WriteLine()

    let checkWarn (name: string) (details: string) =
        let prefix = if supportsUnicode () then "  ⚠ " else "  [!] "
        writeColor ConsoleColor.Yellow prefix
        writeColor ConsoleColor.White $"{name}: "
        writeColor ConsoleColor.Gray details
        System.Console.WriteLine()

    let info (text: string) =
        writeColor ConsoleColor.Gray $"    {text}"
        System.Console.WriteLine()

    let subHeader (text: string) =
        System.Console.WriteLine()

        let line =
            if supportsUnicode () then
                $"  ─── {text} ───"
            else
                $"  --- {text} ---"

        writeColor ConsoleColor.DarkCyan line
        System.Console.WriteLine()

/// <summary>Check if a URL is localhost</summary>
let private isLocalhost (url: string) =
    url.Contains("localhost") || url.Contains("127.0.0.1") || url.Contains("::1")

/// <summary>Check Ollama connectivity</summary>
let private checkOllama () =
    task {
        let ollamaUrl =
            match CredentialVault.getSecret "OLLAMA_BASE_URL" with
            | Ok url -> url
            | Error _ -> "http://localhost:11434"

        let location = if isLocalhost ollamaUrl then "Local" else "Remote"
        Console.info $"Checking Ollama at {ollamaUrl}..."

        try
            use http = new HttpClient(Timeout = TimeSpan.FromSeconds(5.0))
            let baseUri = Uri(ollamaUrl)

            // Check connectivity
            let sw = System.Diagnostics.Stopwatch.StartNew()
            let! models = getTagsAsync http baseUri
            sw.Stop()

            return
                { Name = $"Ollama ({location})"
                  Status = "Connected"
                  Details = $"{ollamaUrl} - {models.Length} models, {sw.ElapsedMilliseconds}ms latency"
                  IsOk = true },
                Some models
        with
        | :? TaskCanceledException ->
            return
                { Name = $"Ollama ({location})"
                  Status = "Timeout"
                  Details = $"{ollamaUrl} - Connection timed out after 5s (is Ollama running?)"
                  IsOk = false },
                None
        | :? HttpRequestException as ex ->
            return
                { Name = $"Ollama ({location})"
                  Status = "Unreachable"
                  Details = $"{ollamaUrl} - {ex.Message}"
                  IsOk = false },
                None
        | ex ->
            return
                { Name = $"Ollama ({location})"
                  Status = "Error"
                  Details = $"{ollamaUrl} - {ex.GetType().Name}: {ex.Message}"
                  IsOk = false },
                None
    }

/// <summary>Check environment configuration</summary>
let private checkEnvConfig () =
    let checks =
        [ ("OLLAMA_BASE_URL", CredentialVault.getSecret "OLLAMA_BASE_URL")
          ("DEFAULT_OLLAMA_MODEL", CredentialVault.getSecret "DEFAULT_OLLAMA_MODEL")
          ("OPENWEBUI_EMAIL", CredentialVault.getSecret "OPENWEBUI_EMAIL")
          ("OPENWEBUI_PASSWORD", CredentialVault.getSecret "OPENWEBUI_PASSWORD") ]

    checks
    |> List.map (fun (key, result) ->
        match result with
        | Ok value ->
            let masked =
                if value.Length > 4 then
                    value.Substring(0, 4) + "****"
                else
                    "****"

            { Name = key
              Status = "Set"
              Details = masked
              IsOk = true }
        | Error _ ->
            { Name = key
              Status = "Not Set"
              Details = "(using defaults)"
              IsOk = false })

/// <summary>Get system information</summary>
let private getSystemInfo () =
    let osDesc = RuntimeInformation.OSDescription
    let osArch = RuntimeInformation.OSArchitecture.ToString()
    let procArch = RuntimeInformation.ProcessArchitecture.ToString()
    let dotnetVersion = RuntimeInformation.FrameworkDescription
    let procCount = Environment.ProcessorCount

    [ { Name = "OS"
        Status = "Info"
        Details = $"{osDesc} ({osArch})"
        IsOk = true }
      { Name = ".NET Runtime"
        Status = "Info"
        Details = dotnetVersion
        IsOk = true }
      { Name = "Process Arch"
        Status = "Info"
        Details = procArch
        IsOk = true }
      { Name = "CPU Cores"
        Status = "Info"
        Details = $"{procCount} logical processors"
        IsOk = true } ]

/// <summary>Get memory information</summary>
let private getMemoryInfo () =
    let proc = System.Diagnostics.Process.GetCurrentProcess()
    let workingSetMB = proc.WorkingSet64 / (1024L * 1024L)
    let privateMemMB = proc.PrivateMemorySize64 / (1024L * 1024L)
    let gcTotalMB = GC.GetTotalMemory(false) / (1024L * 1024L)

    let status =
        if workingSetMB < 500L then "OK"
        else if workingSetMB < 1000L then "High"
        else "Critical"

    let isOk = workingSetMB < 1000L

    [ { Name = "Working Set"
        Status = status
        Details = $"{workingSetMB} MB"
        IsOk = isOk }
      { Name = "Private Memory"
        Status = "Info"
        Details = $"{privateMemMB} MB"
        IsOk = true }
      { Name = "GC Heap"
        Status = "Info"
        Details = $"{gcTotalMB} MB"
        IsOk = true } ]

/// <summary>Detect GPU/CUDA availability</summary>
let private getGpuInfo () =
    let results = ResizeArray<DiagnosticResult>()

    // Check for NVIDIA GPU via nvidia-smi
    try
        let psi =
            System.Diagnostics.ProcessStartInfo(
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
            )

        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true

        use proc = System.Diagnostics.Process.Start(psi)
        let output = proc.StandardOutput.ReadToEnd()
        proc.WaitForExit()

        if proc.ExitCode = 0 && not (String.IsNullOrWhiteSpace(output)) then
            let lines = output.Trim().Split([| '\n' |], StringSplitOptions.RemoveEmptyEntries)

            for line in lines do
                let parts = line.Split([| ',' |]) |> Array.map (fun s -> s.Trim())

                if parts.Length >= 3 then
                    let gpuName = parts.[0]
                    let memMB = parts.[1]
                    let driver = parts.[2]

                    results.Add(
                        { Name = "NVIDIA GPU"
                          Status = "Detected"
                          Details = $"{gpuName}, {memMB} MB VRAM, Driver {driver}"
                          IsOk = true }
                    )

            // Check CUDA version
            let psiCuda =
                System.Diagnostics.ProcessStartInfo("nvidia-smi", "--query-gpu=compute_cap --format=csv,noheader")

            psiCuda.RedirectStandardOutput <- true
            psiCuda.UseShellExecute <- false
            psiCuda.CreateNoWindow <- true
            use procCuda = System.Diagnostics.Process.Start(psiCuda)
            let cudaOutput = procCuda.StandardOutput.ReadToEnd().Trim()
            procCuda.WaitForExit()

            if not (String.IsNullOrWhiteSpace(cudaOutput)) then
                results.Add(
                    { Name = "CUDA Compute"
                      Status = "Available"
                      Details = $"Capability {cudaOutput}"
                      IsOk = true }
                )
        else
            results.Add(
                { Name = "NVIDIA GPU"
                  Status = "Not Found"
                  Details = "nvidia-smi not available or no GPU"
                  IsOk = true }
            )
    with _ ->
        results.Add(
            { Name = "NVIDIA GPU"
              Status = "Not Found"
              Details = "nvidia-smi not installed"
              IsOk = true }
        )

    // Check for ROCm (AMD GPU) - Windows/Linux
    try
        let rocmPath =
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                Environment.GetEnvironmentVariable("HIP_PATH")
            else
                "/opt/rocm"

        if not (String.IsNullOrEmpty(rocmPath)) && Directory.Exists(rocmPath) then
            results.Add(
                { Name = "AMD ROCm"
                  Status = "Detected"
                  Details = rocmPath
                  IsOk = true }
            )
    with _ ->
        ()

    // Check environment variables for GPU frameworks
    let cudaHome = Environment.GetEnvironmentVariable("CUDA_HOME")
    let cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH")

    if not (String.IsNullOrEmpty(cudaHome)) then
        results.Add(
            { Name = "CUDA_HOME"
              Status = "Set"
              Details = cudaHome
              IsOk = true }
        )
    elif not (String.IsNullOrEmpty(cudaPath)) then
        results.Add(
            { Name = "CUDA_PATH"
              Status = "Set"
              Details = cudaPath
              IsOk = true }
        )

    if results.Count = 0 then
        results.Add(
            { Name = "GPU Acceleration"
              Status = "Not Available"
              Details = "No GPU detected - using CPU"
              IsOk = true }
        )

    results |> Seq.toList

/// <summary>Get disk space for workspace</summary>
let private getDiskInfo () =
    try
        let currentDir = Environment.CurrentDirectory
        let root = Path.GetPathRoot(currentDir)
        let drive = DriveInfo(root)

        let freeGB = drive.AvailableFreeSpace / (1024L * 1024L * 1024L)
        let totalGB = drive.TotalSize / (1024L * 1024L * 1024L)

        let usedPercent =
            100.0 - (float drive.AvailableFreeSpace / float drive.TotalSize * 100.0)

        let status =
            if freeGB > 10L then "OK"
            else if freeGB > 5L then "Low"
            else "Critical"

        let isOk = freeGB > 5L

        [ { Name = $"Disk ({root.TrimEnd('\\')})"
            Status = status
            Details = $"{freeGB} GB free of {totalGB} GB ({usedPercent:F1}%% used)"
            IsOk = isOk } ]
    with _ ->
        [ { Name = "Disk"
            Status = "Unknown"
            Details = "Could not read disk info"
            IsOk = true } ]

/// <summary>Check network connectivity to common endpoints</summary>
let private checkNetworkConnectivity (verbose: bool) =
    task {
        use http = new HttpClient(Timeout = TimeSpan.FromSeconds(5.0))

        let endpoints =
            [ ("GitHub API", "https://api.github.com", true) // Always check
              ("OpenAI API", "https://api.openai.com", true) // Always check
              ("Anthropic API", "https://api.anthropic.com", true) // Always check
              ("HuggingFace", "https://huggingface.co", true) // Always check
              ("Ollama Hub", "https://ollama.com", verbose) // Verbose only
              ("PyPI", "https://pypi.org", verbose) // Verbose only
              ("npm Registry", "https://registry.npmjs.org", verbose) // Verbose only
              ("NuGet", "https://api.nuget.org/v3/index.json", verbose) ] // Verbose only

        let endpointsToCheck =
            endpoints |> List.filter (fun (_, _, shouldCheck) -> shouldCheck)

        Console.info $"Checking {endpointsToCheck.Length} endpoints (5s timeout each)..."

        let! results =
            endpointsToCheck
            |> List.map (fun (name, url, _) ->
                task {
                    try
                        let sw = System.Diagnostics.Stopwatch.StartNew()
                        let! resp = http.GetAsync(url)
                        sw.Stop()
                        let code = int resp.StatusCode
                        // Any response (even 401/403) means the endpoint is reachable
                        return
                            { Name = name
                              Status = "Reachable"
                              Details = $"{url} ({sw.ElapsedMilliseconds}ms, HTTP {code})"
                              IsOk = true }
                    with
                    | :? TaskCanceledException ->
                        return
                            { Name = name
                              Status = "Timeout"
                              Details = $"{url} - Timed out after 5s"
                              IsOk = false }
                    | :? HttpRequestException as ex ->
                        let msg =
                            if ex.Message.Length > 50 then
                                ex.Message.Substring(0, 50) + "..."
                            else
                                ex.Message

                        return
                            { Name = name
                              Status = "Unreachable"
                              Details = $"{url} - {msg}"
                              IsOk = false }
                    | ex ->
                        let msg =
                            if ex.Message.Length > 40 then
                                ex.Message.Substring(0, 40) + "..."
                            else
                                ex.Message

                        return
                            { Name = name
                              Status = "Error"
                              Details = $"{url} - {ex.GetType().Name}: {msg}"
                              IsOk = false }
                })
            |> Task.WhenAll

        return results |> Array.toList
    }

/// <summary>Run a shell process and capture stdout with a timeout.</summary>
let private tryRunProcess (exe: string) (args: string) (timeoutMs: int) =
    try
        let psi = ProcessStartInfo(exe, args)
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true
        use proc = new Process()
        proc.StartInfo <- psi
        let started = proc.Start()

        if not started then
            Error "failed to start"
        else if proc.WaitForExit(timeoutMs) then
            let output = proc.StandardOutput.ReadToEnd()
            let err = proc.StandardError.ReadToEnd()

            if proc.ExitCode = 0 then
                Ok output
            else
                Error(if String.IsNullOrWhiteSpace(err) then output else err)
        else
            try
                proc.Kill(true)
            with _ ->
                ()

            Error "timeout"
    with ex ->
        Error ex.Message

/// <summary>Check Docker availability.</summary>
let private checkDocker () =
    match tryRunProcess "docker" "--version" 5000 with
    | Ok output ->
        { Name = "Docker Engine"
          Status = "Available"
          Details = output.Trim()
          IsOk = true }
    | Error err ->
        { Name = "Docker Engine"
          Status = "Unavailable"
          Details = err
          IsOk = false }

/// <summary>List running containers (names + status).</summary>
let private listDockerContainers () =
    match tryRunProcess "docker" "ps --format \"{{.Names}}|{{.Status}}\"" 5000 with
    | Ok output ->
        let lines =
            output.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.toList

        let containers =
            lines
            |> List.choose (fun line ->
                match line.Split('|') with
                | [| name; status |] -> Some(name.Trim(), status.Trim())
                | _ -> None)

        Ok containers
    | Error err -> Error err

/// <summary>Check for presence of key TARS infra containers.</summary>
let private checkTarsInfraContainers (containers: (string * string) list) =
    let required = [ "graphiti"; "chroma"; "postgres"; "redis"; "ollama" ]

    required
    |> List.map (fun name ->
        match
            containers
            |> List.tryFind (fun (n, _) -> n.IndexOf(name, StringComparison.OrdinalIgnoreCase) >= 0)
        with
        | Some(_, status) ->
            { Name = $"Container: {name}"
              Status = "Running"
              Details = status
              IsOk = true }
        | None ->
            { Name = $"Container: {name}"
              Status = "Missing"
              Details = "Not found in docker ps"
              IsOk = false })

/// <summary>Get TARS-specific information</summary>
let private getTarsInfo () =
    let workspaceDir = Environment.CurrentDirectory
    let hasTarsConfig = File.Exists(Path.Combine(workspaceDir, "tars.json"))
    let hasSecretsFile = File.Exists(Path.Combine(workspaceDir, "secrets.json"))
    let v2Dir = Path.Combine(workspaceDir, "v2")
    let hasV2 = Directory.Exists(v2Dir)

    [ { Name = "Workspace"
        Status = "Info"
        Details = workspaceDir
        IsOk = true }
      { Name = "TARS v2 Directory"
        Status = (if hasV2 then "Found" else "Missing")
        Details = v2Dir
        IsOk = hasV2 }
      { Name = "Config (tars.json)"
        Status = (if hasTarsConfig then "Found" else "Not Found")
        Details = "(optional)"
        IsOk = true }
      { Name = "Secrets File"
        Status = (if hasSecretsFile then "Found" else "Not Found")
        Details = "(optional - use env vars instead)"
        IsOk = true } ]

/// <summary>Run all diagnostics</summary>
/// <param name="logger">Serilog logger</param>
/// <param name="verbose">Show additional diagnostic details</param>
let runWithVerbose (logger: ILogger) (verbose: bool) =
    task {
        Console.header "TARS System Diagnostics"

        if verbose then
            Console.info "(Verbose mode - showing additional details)"

        // System info
        Console.subHeader "System Information"
        let sysInfo = getSystemInfo ()

        for r in sysInfo do
            Console.checkOk r.Name r.Details

        // GPU info
        Console.subHeader "GPU / Acceleration"
        let gpuInfo = getGpuInfo ()

        for r in gpuInfo do
            match r.Status with
            | "Detected"
            | "Available"
            | "Set" -> Console.checkOk r.Name r.Details
            | "Not Found"
            | "Not Available" -> Console.checkWarn r.Name r.Details
            | _ -> Console.checkOk r.Name r.Details

        // Memory info
        Console.subHeader "Memory Usage"
        let memInfo = getMemoryInfo ()

        for r in memInfo do
            if r.IsOk then
                Console.checkOk r.Name r.Details
            else
                Console.checkWarn r.Name r.Details

        // Disk info
        Console.subHeader "Disk Space"
        let diskInfo = getDiskInfo ()

        for r in diskInfo do
            if r.IsOk then
                Console.checkOk r.Name r.Details
            else
                Console.checkWarn r.Name r.Details

        // TARS info
        Console.subHeader "TARS Configuration"
        let tarsInfo = getTarsInfo ()

        for r in tarsInfo do
            if r.IsOk then
                Console.checkOk r.Name r.Details
            else
                Console.checkWarn r.Name r.Details

        // Docker / Infra
        Console.subHeader "Infra (Docker / Containers)"
        let dockerResult = checkDocker ()

        if dockerResult.IsOk then
            Console.checkOk dockerResult.Name dockerResult.Details
        else
            Console.checkWarn dockerResult.Name dockerResult.Details

        let containerResults =
            match listDockerContainers () with
            | Ok containers ->
                if containers.Length = 0 then
                    Console.info "No running containers detected."
                    []
                else
                    Console.info $"Containers ({containers.Length}):"

                    let toShow =
                        if verbose then
                            containers
                        else
                            containers |> List.truncate 10

                    toShow |> List.iter (fun (n, s) -> Console.info $"  • {n} [{s}]")

                    if not verbose && containers.Length > 10 then
                        Console.info $"  ... and {containers.Length - 10} more (use --verbose to see all)"

                    checkTarsInfraContainers containers
            | Error err ->
                Console.checkWarn "Docker ps" err
                []

        for r in containerResults do
            if r.IsOk then
                Console.checkOk r.Name r.Details
            else
                Console.checkWarn r.Name r.Details

        // Environment checks
        Console.subHeader "Environment Variables"
        let envResults = checkEnvConfig ()

        for r in envResults do
            if r.IsOk then
                Console.checkOk r.Name r.Details
            else
                Console.checkWarn r.Name r.Details

        // Ollama connectivity
        Console.subHeader "LLM Connectivity"
        let! (ollamaResult, models) = checkOllama ()

        if ollamaResult.IsOk then
            Console.checkOk ollamaResult.Name ollamaResult.Details

            match models with
            | Some m when m.Length > 0 ->
                Console.info "Available models:"
                let modelsToShow = if verbose then m else m |> List.truncate 10

                for model in modelsToShow do
                    Console.info $"  • {model}"

                if not verbose && m.Length > 10 then
                    Console.info $"  ... and {m.Length - 10} more (use --verbose to see all)"
            | _ -> ()
        else
            Console.checkFail ollamaResult.Name ollamaResult.Details

        // Network connectivity
        Console.subHeader "External Connectivity"
        let! netResults = checkNetworkConnectivity verbose

        for r in netResults do
            if r.IsOk then
                Console.checkOk r.Name r.Details
            else
                Console.checkWarn r.Name r.Details

        // Summary
        Console.subHeader "Summary"
        let mutable issues = 0

        if not ollamaResult.IsOk then
            issues <- issues + 1

        let failedNet = netResults |> List.filter (fun r -> not r.IsOk) |> List.length
        issues <- issues + failedNet
        let lowDisk = diskInfo |> List.exists (fun r -> not r.IsOk)

        if lowDisk then
            issues <- issues + 1

        let highMem = memInfo |> List.exists (fun r -> not r.IsOk)

        if highMem then
            issues <- issues + 1

        let noGpu =
            gpuInfo
            |> List.forall (fun r -> r.Status = "Not Found" || r.Status = "Not Available")

        if not dockerResult.IsOk then
            issues <- issues + 1

        let missingInfra =
            containerResults |> List.filter (fun r -> not r.IsOk) |> List.length

        issues <- issues + missingInfra

        if issues = 0 then
            Console.checkOk "System Status" "All checks passed ✨"

            if noGpu then
                Console.info "Note: No GPU detected - LLM inference will use CPU"
        elif issues <= 2 then
            Console.checkWarn "System Status" $"{issues} minor issue(s) detected"
        else
            Console.checkFail "System Status" $"{issues} issues detected - review above"

        System.Console.WriteLine()
        return 0
    }

/// <summary>Generate TARS Architecture Diagram</summary>
let private getArchitectureDiagram () =
    let ascii =
        """
       +----------------+       +----------------+
       |   User / CLI   | ----> | Tars.Interface |
       +----------------+       +-------+--------+
                                        |
                                        v
       +--------------------------------------------------+
       |                   Tars.Kernel                    |
       |  (Agent Registry, Message Bus, Tool Execution)   |
       +------------------------+-------------------------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
  +--------------+      +--------------+      +--------------+
  | Tars.Cortex  |      | Tars.Evolution|     |   Tars.Llm   |
  | (Cognitive)  |      | (Self-Imp.)   |     | (Connectors) |
  +------+-------+      +-------+------+      +-------+------+
         |                      |                     |
         v                      v                     v
  +--------------+      +--------------+      +--------------+
  |  Knowledge   |      |   Metascript  |     | External APIs|
  |    Graph     |      |   (Workflow)  |     | (Ollama/AI)  |
  +--------------+      +--------------+      +--------------+
        """

    let mermaid =
        """graph TD
    User[User / CLI] --> Interface[Tars.Interface.Cli]
    Interface --> Kernel[Tars.Kernel]
    Interface --> Evolution[Tars.Evolution]
    
    subgraph Core System
        Kernel --> Core[Tars.Core]
        Kernel --> Security[Tars.Security]
        Kernel --> Sandbox[Tars.Sandbox]
        Kernel --> Llm[Tars.Llm]
    end
    
    subgraph Cognitive Layer
        Evolution --> Cortex[Tars.Cortex]
        Cortex --> Knowledge[Knowledge Graph]
        Cortex --> Memory[Semantic Memory]
    end
    
    subgraph External
        Llm --> Ollama
        Llm --> OpenAI
        Llm --> vLLM
    end"""

    ascii, mermaid

/// <summary>Run diagnostics with architecture output</summary>
let runWithArch (logger: ILogger) =
    task {
        Console.header "TARS Architecture"

        let ascii, mermaid = getArchitectureDiagram ()

        Console.writeColor ConsoleColor.Cyan ascii
        System.Console.WriteLine()

        let outputFile = "tars_architecture.md"

        let mdContent =
            sprintf "# TARS Architecture\n\n```mermaid\n%s\n```\n\n## ASCII Representation\n```\n%s\n```" mermaid ascii

        do! File.WriteAllTextAsync(outputFile, mdContent)

        Console.checkOk "Architecture" "Displayed above"
        Console.checkOk "Export" $"Saved to {outputFile}"

        return 0
    }

/// <summary>Run all diagnostics (non-verbose)</summary>
let run (logger: ILogger) = runWithVerbose logger false
