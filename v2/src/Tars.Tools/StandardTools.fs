namespace Tars.Tools.Standard

open System
open System.IO
open System.Threading.Tasks
open Tars.Tools
open Tars.Sandbox

module StandardTools =

    let private dockerClient = lazy (DockerClient.createClient ())

    [<TarsToolAttribute("run_command", "Executes a shell command in a secure sandbox. Input: command string.")>]
    let runCommand (command: string) =
        task {
            let client = dockerClient.Value
            // Use /bin/bash -c to execute the command string
            let cmdList = [ "/bin/bash"; "-c"; command ]

            // Use the tars-sandbox image
            let! result = DockerClient.runContainer client "tars-sandbox:latest" cmdList

            match result with
            | Ok(stdout, stderr, exitCode) ->
                if exitCode = 0L then
                    return stdout.Trim()
                else
                    return sprintf "Error (Exit Code %d): %s\nStdout: %s" exitCode (stderr.Trim()) (stdout.Trim())
            | Error e -> return sprintf "Sandbox Error: %s" e
        }

    [<TarsToolAttribute("read_file", "Reads a text file (UTF-8). Input JSON: { \"path\": \"relative/or/absolute\" }")>]
    let readFile (args: string) =
        task {
            try
                let path =
                    try
                        let doc = System.Text.Json.JsonDocument.Parse(args)
                        let root = doc.RootElement
                        let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>
                        if root.TryGetProperty("path", &prop) then
                            prop.GetString()
                        else
                            args
                    with _ ->
                        args

                let fullPath = Path.GetFullPath(path)
                if not (File.Exists fullPath) then
                    return $"File not found: {fullPath}"
                else
                    let text = File.ReadAllText(fullPath)
                    if text.Length > 64000 then
                        return text.Substring(0, 64000) + "... [truncated]"
                    else
                        return text
            with ex ->
                return $"read_file error: {ex.Message}"
        }

    [<TarsToolAttribute("list_dir", "Lists files and folders. Input JSON: { \"path\": \"dir\" }")>]
    let listDir (args: string) =
        task {
            try
                let path =
                    try
                        let doc = System.Text.Json.JsonDocument.Parse(args)
                        let root = doc.RootElement
                        let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>
                        if root.TryGetProperty("path", &prop) then
                            prop.GetString()
                        else
                            args
                    with _ ->
                        args

                let fullPath = Path.GetFullPath(path)
                if not (Directory.Exists fullPath) then
                    return $"Directory not found: {fullPath}"
                else
                    let entries =
                        Directory.EnumerateFileSystemEntries(fullPath)
                        |> Seq.take 200
                        |> Seq.map (fun p -> Path.GetFileName p)
                        |> String.concat "\n"

                    return entries
            with ex ->
                return $"list_dir error: {ex.Message}"
        }
