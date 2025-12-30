namespace Tars.Kernel

open System
open Tars.Core

/// Factory for creating agent instances
module AgentFactory =

    /// Create a new agent with default state
    let create (id: Guid) name version model systemPrompt tools capabilities =
        { Id = AgentId id
          Name = name
          Version = version
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = model
          SystemPrompt = systemPrompt
          Tools = tools
          Capabilities = capabilities
          State = Idle
          Memory = []
          Fitness = 0.5
          Drives = { Accuracy = 0.5; Speed = 0.5; Creativity = 0.5; Safety = 0.5 }
          Constitution = AgentConstitution.Create(AgentId id, GeneralReasoning) }

/// Factory for creating the kernel context
module KernelBootstrap =
    open Tars.Core
    open Tars.Llm
    open Tars.Llm.LlmService
    open SemanticMemory

    type KernelContext = { SemanticMemory: ISemanticMemory }

    let createKernel (storageRoot: string) (embedder: Embedder) (llm: ILlmService) =
        let semMemConfig = { StorageRoot = storageRoot; TopK = 8 }

        let semMem = SemanticMemoryService(semMemConfig, embedder, llm) :> ISemanticMemory

        { SemanticMemory = semMem }
