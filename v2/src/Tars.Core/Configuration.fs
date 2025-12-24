namespace Tars.Core

open System
open System.IO

/// <summary>
/// Configuration for LLM services
/// </summary>
type LlmSettings =
    { Provider: string
      Model: string
      EmbeddingModel: string
      BaseUrl: string option
      LlamaCppUrl: string option
      ApiKey: string option
      ContextWindow: int
      Temperature: float }

/// <summary>
/// Configuration for Memory and Storage
/// </summary>
type MemorySettings =
    { DataDirectory: string
      VectorStorePath: string
      KnowledgeBasePath: string
      EpisodeDbPath: string
      GraphitiUrl: string option
      PostgresConnectionString: string option }

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
    { Llm: LlmSettings
      Memory: MemorySettings
      Evolution: EvolutionSettings
      PreLlm: PreLlmSettings }

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
          EmbeddingModel = "nomic-embed-text"
          BaseUrl = Some "http://localhost:11434"
          LlamaCppUrl = None
          ApiKey = None
          ContextWindow = 32768
          Temperature = 0.7 }

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
              PostgresConnectionString = None }
          Evolution = DefaultEvolution
          PreLlm = DefaultPreLlm }
