namespace Tars.Tools.Standard

open System
open System.IO
open System.Threading.Tasks
open System.Net.Http
open System.Text.RegularExpressions
open Tars.Tools
open Tars.Sandbox

module StandardTools =

    let private dockerClient = lazy (DockerClient.createClient ())
    let private httpClient = lazy (new HttpClient())

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

    [<TarsToolAttribute("html_to_text", "Converts HTML to plain text. Input JSON: { \"html\": \"...\" }")>]
    let htmlToText (args: string) =
        task {
            try
                let html =
                    try
                        let doc = System.Text.Json.JsonDocument.Parse(args)
                        let root = doc.RootElement
                        let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>
                        if root.TryGetProperty("html", &prop) then
                            prop.GetString()
                        else
                            args
                    with _ ->
                        args

                if String.IsNullOrWhiteSpace html then
                    return "html_to_text error: missing html"
                else
                    // naive strip tags
                    let text = Regex.Replace(html, "<.*?>", " ")
                    let decoded = System.Net.WebUtility.HtmlDecode(text)
                    // collapse whitespace
                    let cleaned = Regex.Replace(decoded, "\\s+", " ").Trim()
                    return cleaned
            with ex ->
                return $"html_to_text error: {ex.Message}"
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

    [<TarsToolAttribute("web_fetch", "Fetches a URL over HTTP(S) and returns the text content. Input JSON: { \"url\": \"https://...\" }")>]
    let webFetch (args: string) =
        task {
            try
                let url =
                    try
                        let doc = System.Text.Json.JsonDocument.Parse(args)
                        let root = doc.RootElement
                        let mutable prop = Unchecked.defaultof<System.Text.Json.JsonElement>
                        if root.TryGetProperty("url", &prop) then
                            prop.GetString()
                        else
                            args
                    with _ ->
                        args

                if String.IsNullOrWhiteSpace url then
                    return "web_fetch error: missing url"
                else
                    let! resp = httpClient.Value.GetAsync(url)
                    resp.EnsureSuccessStatusCode() |> ignore
                    let! content = resp.Content.ReadAsStringAsync()
                    let trimmed =
                        if content.Length > 64000 then
                            content.Substring(0, 64000) + "... [truncated]"
                        else
                            content
                    return trimmed
            with ex ->
                return $"web_fetch error: {ex.Message}"
        }
