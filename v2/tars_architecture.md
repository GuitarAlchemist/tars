# TARS Architecture

```mermaid
graph TD
    User[User / CLI] --> Interface[Tars.Interface.Cli]
    Interface --> Kernel[Tars.Kernel]
    Interface --> Evolution[Tars.Evolution]
    
    subgraph Core System
        Kernel --> Core[Tars.Core]
        Kernel --> Security[Tars.Security]
        Kernel --> Sandbox[Tars.Sandbox]
        Kernel --> Llm[Tars.Llm]
    end
    
    subgraph Cognitive Layer
        Evolution --> Cortex[Tars.Cortex]
        Cortex --> Knowledge[Knowledge Graph]
        Cortex --> Memory[Semantic Memory]
    end
    
    subgraph External
        Llm --> Ollama
        Llm --> OpenAI
        Llm --> vLLM
    end
```

## ASCII Representation
```

       +----------------+       +----------------+
       |   User / CLI   | ----> | Tars.Interface |
       +----------------+       +-------+--------+
                                        |
                                        v
       +--------------------------------------------------+
       |                   Tars.Kernel                    |
       |  (Agent Registry, Message Bus, Tool Execution)   |
       +------------------------+-------------------------+
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
  +--------------+      +--------------+      +--------------+
  | Tars.Cortex  |      | Tars.Evolution|     |   Tars.Llm   |
  | (Cognitive)  |      | (Self-Imp.)   |     | (Connectors) |
  +------+-------+      +-------+------+      +-------+------+
         |                      |                     |
         v                      v                     v
  +--------------+      +--------------+      +--------------+
  |  Knowledge   |      |   Metascript  |     | External APIs|
  |    Graph     |      |   (Workflow)  |     | (Ollama/AI)  |
  +--------------+      +--------------+      +--------------+
        
```