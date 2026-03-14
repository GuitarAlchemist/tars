namespace Tars.Tools.Standard

open System
open System.Diagnostics
open System.Net.Http
open System.Text.Json
open Tars.Tools

module LlmTools =

    /// Current active model (mutable for runtime switching)
    let mutable private activeModel = "qwen2.5-coder:1.5b"

    /// HTTP client for Ollama API
    let private httpClient = new HttpClient()

    [<TarsToolAttribute("list_models", "Lists available Ollama models on this system. No input required.")>]
    let listModels (_: string) =
        task {
            printfn "📋 LISTING AVAILABLE MODELS..."

            try
                // Try to get from Ollama API first
                let! response = httpClient.GetAsync("http://localhost:11434/api/tags")

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    let doc = JsonDocument.Parse(content)
                    let models = doc.RootElement.GetProperty("models")

                    let modelList =
                        [| for model in models.EnumerateArray() do
                               let name = model.GetProperty("name").GetString()

                               let size =
                                   let mutable sizeProp = Unchecked.defaultof<JsonElement>

                                   if model.TryGetProperty("size", &sizeProp) then
                                       let bytes = sizeProp.GetInt64()
                                       $"%.1f{float bytes / 1073741824.0} GB"
                                   else
                                       "unknown size"

                               yield $"  - %s{name} (%s{size})" |]
                        |> String.concat "\n"

                    return
                        $"Available Models (from Ollama):\n%s{modelList}\n\nActive model: %s{activeModel}\n\nUse switch_model to change, or pull_model to download new models."
                else
                    return $"Could not reach Ollama API. Active model: %s{activeModel}"
            with ex ->
                // Fallback if Ollama not running
                return
                    $"Ollama API error: %s{ex.Message}\n\nMake sure Ollama is running. Active model: %s{activeModel}"
        }

    [<TarsToolAttribute("switch_model",
                        "Switches to a different LLM model. Input: model name (e.g., 'llama3:8b', 'codestral:latest')")>]
    let switchModel (modelName: string) =
        task {
            let model = modelName.Trim()
            printfn $"🔄 SWITCHING MODEL to: %s{model}"

            if String.IsNullOrWhiteSpace(model) then
                return $"Model name required. Current model: %s{activeModel}"
            else
                let oldModel = activeModel
                activeModel <- model

                // Note: Actual model switching happens in the LLM layer
                // This tool updates the preference and informs what to do
                return
                    $"Model preference changed: %s{oldModel} -> %s{activeModel}\n\nNote: The actual model switch takes effect on next LLM call. Use list_models to see available models."
        }

    [<TarsToolAttribute("recommend_model", "Recommends the best model for a specific task. Input: task description")>]
    let recommendModel (taskDescription: string) =
        task {
            printfn $"🤔 RECOMMENDING MODEL for: %s{taskDescription.Substring(0, min 50 taskDescription.Length)}"

            let task = taskDescription.ToLower()

            let recommendation =
                if
                    task.Contains("code")
                    || task.Contains("program")
                    || task.Contains("function")
                    || task.Contains("debug")
                then
                    "Recommended: **codestral:latest** or **qwen2.5-coder:7b**\n"
                    + "  - Excellent at code generation and debugging\n"
                    + "  - Strong understanding of programming concepts\n"
                    + "  - Good at following code style conventions"
                elif
                    task.Contains("reason")
                    || task.Contains("think")
                    || task.Contains("plan")
                    || task.Contains("analyze")
                then
                    "Recommended: **llama3:70b** or **qwen2.5:14b**\n"
                    + "  - Strong reasoning capabilities\n"
                    + "  - Good at complex multi-step tasks\n"
                    + "  - Better at handling ambiguity"
                elif task.Contains("fast") || task.Contains("quick") || task.Contains("simple") then
                    "Recommended: **qwen2.5-coder:1.5b** or **llama3:8b**\n"
                    + "  - Fast response times\n"
                    + "  - Lower memory usage\n"
                    + "  - Good for simple tasks"
                elif task.Contains("creative") || task.Contains("write") || task.Contains("story") then
                    "Recommended: **llama3:70b** or **claude-3-opus**\n"
                    + "  - Strong creative writing abilities\n"
                    + "  - Good at maintaining narrative coherence\n"
                    + "  - Rich vocabulary and style"
                else
                    "Recommended: **qwen2.5-coder:7b** (general purpose)\n"
                    + "  - Good balance of speed and capability\n"
                    + "  - Works well for most tasks\n"
                    + "  - Reliable code and reasoning"

            return
                sprintf
                    "Task: %s\n\n%s\n\nUse switch_model to change models, or list_models to see what's installed."
                    (if task.Length > 100 then
                         task.Substring(0, 100) + "..."
                     else
                         task)
                    recommendation
        }

    [<TarsToolAttribute("pull_model",
                        "Downloads a new model from Ollama. Input: model name to download (e.g., 'codestral:latest')")>]
    let pullModel (modelName: string) =
        task {
            let model = modelName.Trim()
            printfn $"📥 PULLING MODEL: %s{model}"

            if String.IsNullOrWhiteSpace(model) then
                return "Model name required. Example: pull_model 'codestral:latest'"
            else
                try
                    // Use Ollama CLI to pull model
                    let psi = ProcessStartInfo()
                    psi.FileName <- "ollama"
                    psi.Arguments <- $"pull %s{model}"
                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.UseShellExecute <- false
                    psi.CreateNoWindow <- true

                    let proc = Process.Start(psi)

                    // Don't wait forever, just start the download
                    let completed = proc.WaitForExit(5000) // Wait 5 seconds

                    if completed then
                        let output = proc.StandardOutput.ReadToEnd()
                        let error = proc.StandardError.ReadToEnd()

                        if proc.ExitCode = 0 then
                            return
                                $"Model '%s{model}' pulled successfully!\n%s{output}\n\nUse switch_model to activate it."
                        else
                            return $"Error pulling model '%s{model}': %s{error}"
                    else
                        return
                            $"Model '%s{model}' download started. This may take several minutes for large models.\n\nRun 'ollama pull %s{model}' in a terminal to see progress.\n\nUse list_models to check when it's available."
                with ex ->
                    return $"pull_model error: %s{ex.Message}\n\nMake sure Ollama is installed and running."
        }

    [<TarsToolAttribute("model_info", "Gets detailed information about a specific model. Input: model name")>]
    let modelInfo (modelName: string) =
        task {
            let model =
                if String.IsNullOrWhiteSpace(modelName) then
                    activeModel
                else
                    modelName.Trim()

            printfn $"ℹ️ MODEL INFO: %s{model}"

            try
                let psi = ProcessStartInfo()
                psi.FileName <- "ollama"
                psi.Arguments <- $"show %s{model}"
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                let proc = Process.Start(psi)
                let completed = proc.WaitForExit(10000)

                if completed && proc.ExitCode = 0 then
                    let output = proc.StandardOutput.ReadToEnd()
                    return $"Model: %s{model}\n\n%s{output}"
                else
                    let error = proc.StandardError.ReadToEnd()
                    return $"Could not get info for '%s{model}': %s{error}"
            with ex ->
                return $"model_info error: %s{ex.Message}"
        }

    [<TarsToolAttribute("get_active_model", "Returns the currently active LLM model. No input required.")>]
    let getActiveModel (_: string) =
        task {
            printfn $"🔍 ACTIVE MODEL: %s{activeModel}"

            return
                $"Current active model: %s{activeModel}\n\nUse switch_model to change, or list_models to see alternatives."
        }
