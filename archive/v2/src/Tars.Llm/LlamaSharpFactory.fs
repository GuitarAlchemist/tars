namespace Tars.Llm

open System.Collections.Concurrent

/// <summary>
/// Factory to manage LlamaSharp instances to avoid reloading models.
/// </summary>
module LlamaSharpFactory =
    
    let private cache = ConcurrentDictionary<string, LlamaSharpService>()
    
    let getService (config: LlmServiceConfig) (apiKey: string option) (modelPath: string) =
        cache.GetOrAdd(modelPath, fun path ->
            new LlamaSharpService(config, path))
