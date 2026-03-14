namespace Tars.Connectors

module Llm =
    let generate (modelConfig: string) (prompt: string) =
        let parts = modelConfig.Split(':', 2)

        if parts.Length = 2 then
            let provider = parts[0].ToLowerInvariant()
            let modelName = parts[1]

            match provider with
            | "ollama" -> Ollama.generate modelName prompt
            | "lmstudio" -> OpenAiCompatible.generate "http://localhost:1234" modelName prompt
            | "llamacpp" -> OpenAiCompatible.generate "http://localhost:8080" modelName prompt
            | "openwebui" ->
                OpenWebUi.generate "https://aialpha.bar-scouts.com/" modelName [ { Role = "user"; Content = prompt } ]
            | _ ->
                // Fallback to Ollama if provider prefix is unknown but present (e.g. might be part of model name)
                // But usually we want explicit providers.
                // Let's assume if it looks like "provider:model", we try to match, otherwise default.
                // If we don't recognize the provider, we probably shouldn't strip the prefix if it's actually part of the model name.
                // But for safety, let's just pass the whole thing to Ollama as before.
                Ollama.generate modelConfig prompt
        else
            // No provider prefix, default to Ollama
            Ollama.generate modelConfig prompt
