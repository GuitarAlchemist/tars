namespace Tars.Tools.Standard

open System
open System.IO
open System.Net.Http
open System.Diagnostics
open Tars.Tools

module SystemTools =

    let private httpClient = new HttpClient()

    [<TarsToolAttribute("get_env",
                        "Gets the value of an environment variable. Input: variable name (e.g., 'PATH', 'DOTNET_ROOT')")>]
    let getEnv (varName: string) =
        task {
            let name = varName.Trim()
            printfn "🔧 GET ENV: %s" name

            let value = Environment.GetEnvironmentVariable(name)

            if isNull value then
                let suggestions =
                    [ "PATH"; "DOTNET_ROOT"; "HOME"; "USERPROFILE"; "TEMP"; "COMPUTERNAME" ]
                    |> List.filter (fun v -> Environment.GetEnvironmentVariable(v) <> null)
                    |> String.concat ", "

                return sprintf "Environment variable '%s' not found.\n\nAvailable common variables: %s" name suggestions
            else
                let preview =
                    if value.Length > 200 then
                        value.Substring(0, 200) + "..."
                    else
                        value

                return sprintf "ENV[%s] = %s" name preview
        }

    [<TarsToolAttribute("list_env", "Lists common environment variables. No input required.")>]
    let listEnv (_: string) =
        task {
            printfn "📋 LISTING ENVIRONMENT VARIABLES"

            let commonVars =
                [ "PATH"
                  "DOTNET_ROOT"
                  "HOME"
                  "USERPROFILE"
                  "TEMP"
                  "TMP"
                  "COMPUTERNAME"
                  "USERNAME"
                  "OS"
                  "PROCESSOR_ARCHITECTURE" ]

            let envList =
                commonVars
                |> List.choose (fun name ->
                    let value = Environment.GetEnvironmentVariable(name)

                    if isNull value then
                        None
                    else
                        let preview =
                            if value.Length > 50 then
                                value.Substring(0, 50) + "..."
                            else
                                value

                        Some(sprintf "  %s = %s" name preview))
                |> String.concat "\n"

            return sprintf "Common Environment Variables:\n%s\n\nUse get_env for full values." envList
        }

    [<TarsToolAttribute("run_shell",
                        "Runs a shell command safely. Input JSON: { \"command\": \"dotnet --version\", \"timeout\": 30 }")>]
    let runShell (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let command = root.GetProperty("command").GetString()

                let mutable timeoutProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let timeout =
                    if root.TryGetProperty("timeout", &timeoutProp) then
                        timeoutProp.GetInt32()
                    else
                        30

                // Safety checks
                let dangerous = [ "rm -rf"; "del /s"; "format"; "shutdown"; "reboot" ]
                let isDangerous = dangerous |> List.exists (fun d -> command.ToLower().Contains(d))

                if isDangerous then
                    return sprintf "⚠️ Command blocked for safety: %s" command
                else
                    printfn "🖥️ RUNNING: %s (timeout: %ds)" command timeout

                    let psi = ProcessStartInfo()

                    psi.FileName <-
                        if Environment.OSVersion.Platform = PlatformID.Win32NT then
                            "cmd.exe"
                        else
                            "/bin/sh"

                    psi.Arguments <-
                        if Environment.OSVersion.Platform = PlatformID.Win32NT then
                            sprintf "/c %s" command
                        else
                            sprintf "-c \"%s\"" command

                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.UseShellExecute <- false
                    psi.CreateNoWindow <- true

                    let proc = Process.Start(psi)
                    let completed = proc.WaitForExit(timeout * 1000)

                    if completed then
                        let stdout = proc.StandardOutput.ReadToEnd()
                        let stderr = proc.StandardError.ReadToEnd()
                        let output = if stdout.Length > 0 then stdout else stderr

                        let preview =
                            if output.Length > 1000 then
                                output.Substring(0, 1000) + "..."
                            else
                                output

                        return sprintf "Exit code: %d\n\n%s" proc.ExitCode preview
                    else
                        proc.Kill()
                        return sprintf "Command timed out after %d seconds" timeout
            with ex ->
                return "run_shell error: " + ex.Message
        }

    [<TarsToolAttribute("http_get", "Fetches content from a URL. Input: URL to fetch")>]
    let httpGet (url: string) =
        task {
            let uri = url.Trim()
            printfn "🌐 HTTP GET: %s" uri

            try
                if not (uri.StartsWith("http://") || uri.StartsWith("https://")) then
                    return "URL must start with http:// or https://"
                else
                    let! response = httpClient.GetAsync(uri)

                    if response.IsSuccessStatusCode then
                        let! content = response.Content.ReadAsStringAsync()

                        let preview =
                            if content.Length > 2000 then
                                content.Substring(0, 2000) + "..."
                            else
                                content

                        return sprintf "Status: %d %s\n\n%s" (int response.StatusCode) (response.ReasonPhrase) preview
                    else
                        return sprintf "HTTP Error: %d %s" (int response.StatusCode) (response.ReasonPhrase)
            with ex ->
                return "http_get error: " + ex.Message
        }

    [<TarsToolAttribute("get_system_info", "Gets system and runtime information. No input required.")>]
    let getSystemInfo (_: string) =
        task {
            printfn "💻 GETTING SYSTEM INFO"

            let info =
                "System Information:\n"
                + sprintf "  OS: %s\n" (Environment.OSVersion.ToString())
                + sprintf "  Machine: %s\n" Environment.MachineName
                + sprintf "  Processors: %d\n" Environment.ProcessorCount
                + sprintf "  .NET Version: %s\n" (Environment.Version.ToString())
                + sprintf "  64-bit OS: %b\n" Environment.Is64BitOperatingSystem
                + sprintf "  64-bit Process: %b\n" Environment.Is64BitProcess
                + sprintf "  Current Directory: %s\n" Environment.CurrentDirectory
                + sprintf "  User: %s\n" Environment.UserName

            return info
        }

    [<TarsToolAttribute("get_time", "Gets the current date and time. No input required.")>]
    let getTime (_: string) =
        task {
            let now = DateTime.Now
            let utc = DateTime.UtcNow

            return
                sprintf
                    "Current Time:\n  Local: %s\n  UTC: %s\n  Unix: %d"
                    (now.ToString("yyyy-MM-dd HH:mm:ss"))
                    (utc.ToString("yyyy-MM-dd HH:mm:ss"))
                    (DateTimeOffset.UtcNow.ToUnixTimeSeconds())
        }

    [<TarsToolAttribute("get_working_dir", "Gets the current working directory. No input required.")>]
    let getWorkingDir (_: string) =
        task {
            let cwd = Environment.CurrentDirectory

            let files =
                try
                    Directory.GetFiles(cwd)
                    |> Array.take (min 10 (Directory.GetFiles(cwd).Length))
                    |> Array.map Path.GetFileName
                    |> String.concat ", "
                with _ ->
                    "unable to list"

            return sprintf "Working Directory: %s\n\nRecent files: %s" cwd files
        }

    [<TarsToolAttribute("set_working_dir", "Changes the current working directory. Input: new directory path")>]
    let setWorkingDir (path: string) =
        task {
            let newPath = path.Trim()
            printfn "📂 CHANGING DIR: %s" newPath

            try
                let fullPath = Path.GetFullPath(newPath)

                if Directory.Exists(fullPath) then
                    Environment.CurrentDirectory <- fullPath
                    return sprintf "Changed directory to: %s" fullPath
                else
                    return sprintf "Directory not found: %s" fullPath
            with ex ->
                return "set_working_dir error: " + ex.Message
        }
