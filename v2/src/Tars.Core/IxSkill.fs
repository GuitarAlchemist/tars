namespace Tars.Core

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Threading.Tasks

/// Generic client for the sibling `ix` repo's skill CLI.
///
/// ix is a Rust workspace whose `ix-skill` crate builds a binary named `ix`.
/// Skills are invoked as `ix run <skill> --input-file <json> --format json`,
/// resolved by registry name (e.g. `optimize`, `grammar.search`, `grammar.weights`).
///
/// A prebuilt binary under the repo's `target/{release,debug}/` is preferred;
/// otherwise we drive it via `cargo run -p ix-skill`. This module is the single
/// seam every TARS layer uses to reach ix, so the subprocess handling (and its
/// pipe-draining hazards) lives in exactly one place.
module IxSkill =

    type Config =
        { /// Path to the cargo executable (used only when no prebuilt binary exists).
          CargoPath: string
          /// Timeout for skill invocations.
          Timeout: TimeSpan
          /// The ix repo root — cargo and `target/` live here.
          RepoDir: string option }

    let defaultConfig =
        { CargoPath = "cargo"
          Timeout = TimeSpan.FromSeconds 30.0
          RepoDir = None }

    /// Discover the sibling ix repo at the conventional ~/source/repos/ix path.
    let discover () : Config option =
        let candidate =
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "source", "repos", "ix")
        if Directory.Exists candidate then
            Some { defaultConfig with RepoDir = Some candidate }
        else None

    /// Locate a prebuilt `ix` binary (the ix-skill crate's `[[bin]]`) under target/.
    let private prebuiltExe (config: Config) : string option =
        match config.RepoDir with
        | Some dir ->
            [ "release"; "debug" ]
            |> List.collect (fun profile ->
                [ Path.Combine(dir, "target", profile, "ix.exe")
                  Path.Combine(dir, "target", profile, "ix") ])
            |> List.tryFind File.Exists
        | None -> None

    /// Resolve how to invoke ix-skill: (fileName, argPrefix).
    let private invocation (config: Config) : string * string =
        match prebuiltExe config with
        | Some exe -> exe, ""
        | None -> config.CargoPath, "run -q -p ix-skill -- "

    let private applyWorkingDir (config: Config) (psi: ProcessStartInfo) =
        match config.RepoDir with
        | Some dir -> psi.WorkingDirectory <- dir
        | None -> ()

    /// True when ix can be reached (prebuilt binary present, or cargo resolves it).
    let isAvailable (config: Config) : bool =
        try
            let fileName, prefix = invocation config
            let psi = ProcessStartInfo()
            psi.FileName <- fileName
            psi.Arguments <- prefix + "list skills"
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true
            applyWorkingDir config psi

            use proc = Process.Start(psi)
            // Drain both streams before waiting: `list skills` emits enough JSON to
            // fill the OS pipe buffer, which would otherwise deadlock the child.
            proc.StandardOutput.ReadToEnd() |> ignore
            proc.StandardError.ReadToEnd() |> ignore
            proc.WaitForExit(30000) |> ignore
            proc.HasExited && proc.ExitCode = 0
        with _ -> false

    /// True when ix exposes the real `pipeline` surface (not the historical stub).
    /// Probes `pipeline schema`, which is read-only and draws its skill enum from
    /// the live registry — exit 0 means a rebuilt binary with the executor wired in.
    /// Callers degrade to a serial path when this is false (ADR 0001, D7).
    let pipelineAvailable (config: Config) : bool =
        try
            let fileName, prefix = invocation config
            let psi = ProcessStartInfo()
            psi.FileName <- fileName
            psi.Arguments <- prefix + "pipeline schema --format json"
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true
            applyWorkingDir config psi

            use proc = Process.Start(psi)
            let out = proc.StandardOutput.ReadToEnd()
            proc.StandardError.ReadToEnd() |> ignore
            proc.WaitForExit(30000) |> ignore
            // A stub prints a notice and/or exits non-zero; the real schema is a JSON object.
            proc.HasExited && proc.ExitCode = 0 && out.Contains("\"")
        with _ -> false

    /// Execute an `ix.yaml` pipeline file, returning raw JSON stdout.
    /// `paramFiles` are (name, path) pairs bound as `--param name=@path`; each
    /// file must hold a JSON value (a JSON-encoded string for raw text). The
    /// stdout shape is `{ "stages": { "<id>": { "output": ... } } }`.
    let runPipelineJson
        (config: Config)
        (yamlPath: string)
        (paramFiles: (string * string) list)
        : Task<Result<string, string>> =
        task {
            try
                let fileName, prefix = invocation config
                let paramArgs =
                    paramFiles
                    |> List.map (fun (name, path) -> sprintf "--param %s=@\"%s\"" name path)
                    |> String.concat " "
                let psi = ProcessStartInfo()
                psi.FileName <- fileName
                psi.Arguments <-
                    sprintf "%spipeline run --file \"%s\" %s --format json" prefix yamlPath paramArgs
                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.CreateNoWindow <- true
                applyWorkingDir config psi

                use proc = new Process()
                proc.StartInfo <- psi

                let stdout = StringBuilder()
                let stderr = StringBuilder()
                proc.OutputDataReceived.Add(fun e ->
                    if not (isNull e.Data) then stdout.AppendLine(e.Data) |> ignore)
                proc.ErrorDataReceived.Add(fun e ->
                    if not (isNull e.Data) then stderr.AppendLine(e.Data) |> ignore)

                proc.Start() |> ignore
                proc.BeginOutputReadLine()
                proc.BeginErrorReadLine()

                let! completed =
                    Task.Run(fun () -> proc.WaitForExit(int config.Timeout.TotalMilliseconds))

                if not completed then
                    try proc.Kill() with _ -> ()
                    return Error "ix pipeline timed out"
                elif proc.ExitCode <> 0 then
                    return Error (stderr.ToString().Trim())
                else
                    return Ok (stdout.ToString().Trim())
            with ex ->
                return Error (sprintf "ix pipeline error: %s" ex.Message)
        }

    /// Run an ix skill with a JSON input document, returning raw JSON stdout.
    /// Input is passed via a temp file to avoid shell-quoting hazards on Windows.
    let runSkillJson
        (config: Config)
        (skill: string)
        (inputJson: string)
        : Task<Result<string, string>> =
        task {
            let tmp =
                Path.Combine(Path.GetTempPath(), sprintf "tars_ix_%s.json" (Guid.NewGuid().ToString("N")))
            try
                try
                    File.WriteAllText(tmp, inputJson)

                    let fileName, prefix = invocation config
                    let psi = ProcessStartInfo()
                    psi.FileName <- fileName
                    psi.Arguments <-
                        sprintf "%srun %s --input-file \"%s\" --format json" prefix skill tmp
                    psi.UseShellExecute <- false
                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.CreateNoWindow <- true
                    applyWorkingDir config psi

                    use proc = new Process()
                    proc.StartInfo <- psi

                    let stdout = StringBuilder()
                    let stderr = StringBuilder()

                    proc.OutputDataReceived.Add(fun e ->
                        if not (isNull e.Data) then stdout.AppendLine(e.Data) |> ignore)
                    proc.ErrorDataReceived.Add(fun e ->
                        if not (isNull e.Data) then stderr.AppendLine(e.Data) |> ignore)

                    proc.Start() |> ignore
                    proc.BeginOutputReadLine()
                    proc.BeginErrorReadLine()

                    let! completed =
                        Task.Run(fun () ->
                            proc.WaitForExit(int config.Timeout.TotalMilliseconds))

                    if not completed then
                        try proc.Kill() with _ -> ()
                        return Error "ix timed out"
                    elif proc.ExitCode <> 0 then
                        return Error (stderr.ToString().Trim())
                    else
                        return Ok (stdout.ToString().Trim())
                with ex ->
                    return Error (sprintf "ix error: %s" ex.Message)
            finally
                try File.Delete tmp with _ -> ()
        }
