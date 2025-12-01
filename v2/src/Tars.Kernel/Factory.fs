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
          Memory = [] }

/// Factory for creating the kernel context
module KernelBootstrap =
    open Tars.Core
    open SemanticMemory

    let createKernel (storageRoot: string) (embedder: Embedder) =
        let semMemConfig = {
            StorageRoot = storageRoot
            TopK = 8
        }
        
        let semMem = SemanticMemoryService(semMemConfig, embedder) :> ISemanticMemory
        
        {
            SemanticMemory = semMem
        }
