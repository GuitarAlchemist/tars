namespace Tars.Core

open System
open System.IO

/// <summary>
/// Configuration for LLM services
/// </summary>
type LlmSettings =
    { Provider: string
      Model: string
      LlamaSharpModelPath: string option
      EmbeddingModel: string
      BaseUrl: string option
      LlamaCppUrl: string option
      ApiKey: string option
      ContextWindow: int
      Temperature: float
      ReasoningModel: string option
      CodingModel: string option
      FastModel: string option }

/// <summary>
/// Configuration for Memory and Storage
/// </summary>
type MemorySettings =
    { DataDirectory: string
      VectorStorePath: string
      KnowledgeBasePath: string
      EpisodeDbPath: string
      GraphitiUrl: string option
      PostgresConnectionString: string option
      FusekiUrl: string option
      FusekiAuth: string option }

/// <summary>
/// Configuration for Evolution Engine
/// </summary>
type EvolutionSettings =
    { DefaultBudget: float
      MaxIterations: int
      AutoSave: bool
      TraceEnabled: bool }

/// <summary>
/// Pre-LLM pipeline configuration
/// </summary>
type PreLlmSettings =
    { UseIntentClassifier: bool
      DefaultPolicies: string list }

/// Root Configuration object for TARS
/// </summary>
type TarsConfig =
    {
        Llm: LlmSettings
        Memory: MemorySettings
        Evolution: EvolutionSettings
        PreLlm: PreLlmSettings
        /// Overlays for file paths (Original -> Variant)
        VariantOverlays: Map<string, string>
    }

/// <summary>
/// Default configuration values
/// </summary>
module ConfigurationDefaults =

    let getTarsHome () =
        let home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
        Path.Combine(home, ".tars")

    let DefaultLlm =
        { Provider = "Ollama"
          Model = "qwen2.5-coder:7b"
          LlamaSharpModelPath = None
          EmbeddingModel = "nomic-embed-text"
          BaseUrl = Some "http://localhost:11434"
          LlamaCppUrl = None
          ApiKey = None
          ContextWindow = 32768
          Temperature = 0.7
          ReasoningModel = Some "deepseek-r1:8b"
          CodingModel = Some "qwen2.5-coder:7b"
          FastModel = Some "mistral:7b" }

    let DefaultEvolution =
        { DefaultBudget = 100.0
          MaxIterations = 10
          AutoSave = true
          TraceEnabled = false }

    let DefaultPreLlm =
        { UseIntentClassifier = true
          DefaultPolicies = [ "no_destructive_commands" ] }

    let createDefault () =
        let home = getTarsHome ()

        { Llm = DefaultLlm
          Memory =
            { DataDirectory = home
              VectorStorePath = Path.Combine(home, "memory.db")
              KnowledgeBasePath = Path.Combine(home, "knowledge")
              EpisodeDbPath = Path.Combine(home, "episodes.db")
              GraphitiUrl = None
              PostgresConnectionString = None
              FusekiUrl = Some "http://localhost:3030/tars_v2"
              FusekiAuth = None }
          Evolution = DefaultEvolution
          PreLlm = DefaultPreLlm
          VariantOverlays = Map.empty }
