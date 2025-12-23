namespace Tars.Tools.Standard

open System
open System.Diagnostics
open System.Collections.Generic
open Tars.Tools

module AutomationTools =

    /// Structured action log
    let private actionLog = ResizeArray<DateTime * string * string>()

    [<TarsToolAttribute("log_action",
                        "Logs a structured action for tracking. Input JSON: { \"action\": \"started_task\", \"details\": \"Building module X\" }")>]
    let logAction (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let action = root.GetProperty("action").GetString()
                let mutable detailsProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let details =
                    if root.TryGetProperty("details", &detailsProp) then
                        detailsProp.GetString()
                    else
                        ""

                let timestamp = DateTime.Now
                actionLog.Add((timestamp, action, details))

                printfn "📝 LOG: [%s] %s - %s" (timestamp.ToString("HH:mm:ss")) action details

                return sprintf "Logged: [%s] %s" (timestamp.ToString("yyyy-MM-dd HH:mm:ss")) action
            with ex ->
                return "log_action error: " + ex.Message
        }

    [<TarsToolAttribute("get_action_log", "Gets the action log history. Input: optional filter (action name)")>]
    let getActionLog (filter: string) =
        task {
            printfn "📋 GETTING ACTION LOG"

            let filtered =
                if String.IsNullOrWhiteSpace(filter) then
                    actionLog |> Seq.toList
                else
                    actionLog
                    |> Seq.filter (fun (_, action, _) -> action.Contains(filter))
                    |> Seq.toList

            if filtered.Length = 0 then
                return "No actions logged yet. Use log_action to record actions."
            else
                let entries =
                    filtered
                    |> List.map (fun (ts, action, details) ->
                        sprintf "  [%s] %s: %s" (ts.ToString("HH:mm:ss")) action details)
                    |> String.concat "\n"

                return $"Action Log (%d{filtered.Length} entries):\n%s{entries}"
        }

    [<TarsToolAttribute("measure_time",
                        "Starts or stops a timer for performance measurement. Input JSON: { \"timer\": \"name\", \"action\": \"start|stop\" }")>]
    let measureTime (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let timer = root.GetProperty("timer").GetString()
                let action = root.GetProperty("action").GetString().ToLower()

                // Store timer starts in a static dictionary
                // For simplicity, we'll just return the current time
                let now = DateTime.Now

                printfn $"⏱️ TIMER %s{timer}: %s{action}"

                if action = "start" then
                    return sprintf "Timer '%s' started at %s" timer (now.ToString("HH:mm:ss.fff"))
                elif action = "stop" then
                    return
                        sprintf
                            "Timer '%s' stopped at %s (use log to track start time)"
                            timer
                            (now.ToString("HH:mm:ss.fff"))
                else
                    return $"Unknown action: %s{action}. Use 'start' or 'stop'"
            with ex ->
                return "measure_time error: " + ex.Message
        }

    [<TarsToolAttribute("sleep_ms", "Pauses execution for specified milliseconds. Input: milliseconds (max 5000)")>]
    let sleepMs (msInput: string) =
        task {
            try
                let ms = Int32.Parse(msInput.Trim())
                let actualMs = Math.Min(ms, 5000) // Cap at 5 seconds

                printfn $"💤 SLEEPING: %d{actualMs} ms"
                do! System.Threading.Tasks.Task.Delay(actualMs)

                return $"Slept for %d{actualMs} ms"
            with ex ->
                return "sleep_ms error: " + ex.Message
        }

    [<TarsToolAttribute("generate_id", "Generates a unique identifier. Input: optional prefix")>]
    let generateId (prefix: string) =
        task {
            let p =
                if String.IsNullOrWhiteSpace(prefix) then
                    "id"
                else
                    prefix.Trim()

            let guid = Guid.NewGuid().ToString("N").Substring(0, 8)
            let id = $"%s{p}_%s{guid}"

            printfn $"🔑 GENERATED ID: %s{id}"
            return id
        }

    [<TarsToolAttribute("format_json", "Pretty-prints JSON. Input: JSON string")>]
    let formatJson (json: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(json)
                let options = System.Text.Json.JsonSerializerOptions()
                options.WriteIndented <- true
                let formatted = System.Text.Json.JsonSerializer.Serialize(doc.RootElement, options)

                return $"Formatted JSON:\n%s{formatted}"
            with ex ->
                return "format_json error: " + ex.Message
        }

    [<TarsToolAttribute("hash_text", "Computes SHA256 hash of text. Input: text to hash")>]
    let hashText (text: string) =
        task {
            use sha = System.Security.Cryptography.SHA256.Create()
            let bytes = System.Text.Encoding.UTF8.GetBytes(text)
            let hash = sha.ComputeHash(bytes)
            let hashString = BitConverter.ToString(hash).Replace("-", "").ToLower()

            return $"SHA256: %s{hashString}"
        }

    [<TarsToolAttribute("base64_encode", "Encodes text to Base64. Input: text to encode")>]
    let base64Encode (text: string) =
        task {
            let bytes = System.Text.Encoding.UTF8.GetBytes(text)
            let encoded = Convert.ToBase64String(bytes)

            return $"Base64: %s{encoded}"
        }

    [<TarsToolAttribute("base64_decode", "Decodes Base64 to text. Input: Base64 string")>]
    let base64Decode (encoded: string) =
        task {
            try
                let bytes = Convert.FromBase64String(encoded.Trim())
                let decoded = System.Text.Encoding.UTF8.GetString(bytes)

                return $"Decoded: %s{decoded}"
            with ex ->
                return "base64_decode error: " + ex.Message
        }
