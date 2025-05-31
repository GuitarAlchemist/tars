# RAG Test Scenarios for TARS Development

## 🎯 **PRACTICAL TEST SCENARIOS**

These scenarios demonstrate how RAG can provide real value for TARS developers by answering common development questions.

## 📋 **SCENARIO 1: "How do I implement a new command?"**

### **Test Query**
```
"How do I implement a new command in TARS? Show me the pattern and required interfaces."
```

### **Expected RAG Response**
The RAG system should retrieve and synthesize information from:
- `TarsEngine.FSharp.Cli/Commands/Types.fs` - ICommand interface
- `TarsEngine.FSharp.Cli/Commands/CommandBase.fs` - Base implementation patterns
- Example command implementations (e.g., `AutonomousCommand.fs`, `RagCommand.fs`)
- `TarsEngine.FSharp.Cli/Commands/CommandRegistry.fs` - Registration patterns

### **Expected Response Content**
```
To implement a new command in TARS:

1. **Create Command File**: Create a new .fs file in TarsEngine.FSharp.Cli/Commands/
2. **Implement ICommand Interface**: 
   - Name: string - Command name for CLI
   - Description: string - Help text description
   - Usage: string - Usage pattern
   - Examples: string list - Usage examples
   - ValidateOptions: CommandOptions -> bool
   - ExecuteAsync: CommandOptions -> Task<CommandResult>

3. **Example Implementation**:
```fsharp
type MyCommand() =
    interface ICommand with
        member _.Name = "mycommand"
        member _.Description = "Description of my command"
        member _.Usage = "tars mycommand [options]"
        member _.Examples = ["tars mycommand --help"]
        member _.ValidateOptions(options) = true
        member _.ExecuteAsync(options) = 
            Task.Run(fun () -> CommandResult.success("Command executed"))
```

4. **Register Command**: Add to CommandRegistry.RegisterDefaultCommands()

**Sources**: Types.fs, CommandBase.fs, AutonomousCommand.fs
```

### **Quality Criteria**
- [ ] Includes complete ICommand interface definition
- [ ] Shows concrete implementation example
- [ ] Explains registration process
- [ ] Provides accurate source attribution
- [ ] Actionable and immediately usable

## 📋 **SCENARIO 2: "What commands are available in TARS?"**

### **Test Query**
```
"What commands are available in TARS? List them with descriptions and examples."
```

### **Expected RAG Response**
The RAG system should retrieve information from:
- All `*Command.fs` files in TarsEngine.FSharp.Cli/Commands/
- Command descriptions and examples from each implementation
- Main README.md for high-level command overview

### **Expected Response Content**
```
TARS Available Commands:

**Core Commands:**
• autonomous - Comprehensive 28-metric superautonomous measurement
  Example: tars autonomous measure

• intelligence - AI intelligence analysis and measurement  
  Example: tars intelligence measure --detailed

• consciousness - Consciousness system analysis
  Example: tars consciousness analyze

• knowledge - Knowledge extraction and management
  Example: tars knowledge extract --source docs/

**AI & RAG Commands:**
• mcp - Model Context Protocol integration
  Example: tars mcp models

• rag - Retrieval-Augmented Generation with ChromaDB
  Example: tars rag test-services

**Development Commands:**
• run - Run F# scripts or applications
  Example: tars run script.fsx

• ml - Machine learning operations
  Example: tars ml train --model classifier

• demo - Run demos of TARS features
  Example: tars demo all

**Analysis Commands:**
• self-analyze - Analyze TARS codebase
  Example: tars self-analyze --pattern "*.fs"

• self-rewrite - Autonomous code improvement
  Example: tars self-rewrite --target Program.fs

**Sources**: AutonomousCommand.fs, IntelligenceCommand.fs, RagCommand.fs, etc.
```

### **Quality Criteria**
- [ ] Lists all available commands
- [ ] Includes accurate descriptions
- [ ] Provides realistic usage examples
- [ ] Groups commands logically
- [ ] Cites source files correctly

## 📋 **SCENARIO 3: "How does the F# architecture work in TARS?"**

### **Test Query**
```
"Explain the F# architecture in TARS. How are the engine and CLI separated?"
```

### **Expected RAG Response**
The RAG system should retrieve information from:
- `.tars/docs/architecture.md` - System architecture overview
- `.tars/docs/fsharp-migration.md` - F# implementation details
- Project files showing separation (TarsEngine.FSharp.Cli vs TarsEngine.FSharp.RAG)
- README.md project structure section

### **Expected Response Content**
```
TARS F# Architecture:

**Separation of Concerns:**
The TARS F# implementation follows a clean architecture with clear separation:

1. **TarsEngine.FSharp.RAG** - Core Engine
   - Contains business logic and services
   - RAG services (ChromaVectorStore, OllamaEmbeddingService)
   - Type definitions and interfaces
   - No CLI dependencies

2. **TarsEngine.FSharp.Cli** - User Interface
   - Command implementations
   - CLI argument parsing
   - User interaction and formatting
   - Depends on engine projects

**Key Patterns:**
• **Service Interfaces**: Clean abstractions (IVectorStore, IEmbeddingService)
• **Dependency Injection**: Services injected into commands
• **Type Safety**: Strong F# typing prevents runtime errors
• **Functional Programming**: Immutable data and pure functions

**Benefits:**
• **Testability**: Engine can be tested independently
• **Reusability**: Engine services can be used in other contexts
• **Maintainability**: Clear boundaries between concerns
• **Type Safety**: Compile-time guarantees

**Sources**: architecture.md, fsharp-migration.md, project structure
```

### **Quality Criteria**
- [ ] Explains engine/CLI separation clearly
- [ ] Describes key architectural patterns
- [ ] Explains benefits of the approach
- [ ] Provides concrete examples
- [ ] References architectural documentation

## 📋 **SCENARIO 4: "How do I add a new service to the RAG engine?"**

### **Test Query**
```
"How do I add a new service to the TARS RAG engine? Show me the interface and implementation pattern."
```

### **Expected RAG Response**
The RAG system should retrieve information from:
- RAG service interfaces (IVectorStore.fs, IEmbeddingService.fs, IRagService.fs)
- Service implementations (ChromaVectorStore.fs, OllamaEmbeddingService.fs)
- Dependency injection patterns from ServiceCollectionExtensions.fs

### **Expected Response Content**
```
Adding a New Service to TARS RAG Engine:

**1. Define Service Interface**:
Create interface in TarsEngine.FSharp.RAG/RAG/Services/
```fsharp
type IMyService =
    abstract member DoSomethingAsync: input:string -> Task<string>
    abstract member TestConnectionAsync: unit -> Task<bool>
```

**2. Implement Service**:
```fsharp
type MyService(config: MyServiceConfig, httpClient: HttpClient, logger: ILogger<MyService>) =
    interface IMyService with
        member _.DoSomethingAsync(input: string) =
            Task.Run(fun () -> 
                // Implementation here
                input.ToUpperCase())
        
        member _.TestConnectionAsync() =
            Task.Run(fun () -> true)
```

**3. Add to Project File**:
Update TarsEngine.FSharp.RAG.fsproj:
```xml
<Compile Include="RAG/Services/IMyService.fs" />
<Compile Include="RAG/Services/MyService.fs" />
```

**4. Register in DI**:
Add to ServiceCollectionExtensions.fs

**Pattern**: Follow existing services like ChromaVectorStore and OllamaEmbeddingService

**Sources**: IVectorStore.fs, ChromaVectorStore.fs, ServiceCollectionExtensions.fs
```

### **Quality Criteria**
- [ ] Shows complete interface definition
- [ ] Provides concrete implementation example
- [ ] Explains project file updates
- [ ] Covers dependency injection registration
- [ ] References existing service patterns

## 📋 **SCENARIO 5: "How do I use the RAG system for document search?"**

### **Test Query**
```
"How do I use the TARS RAG system to search documents? Show me the API and usage patterns."
```

### **Expected RAG Response**
The RAG system should retrieve information from:
- RAG service interfaces and implementations
- RagCommand.fs usage examples
- RAG type definitions from Types.fs

### **Expected Response Content**
```
Using TARS RAG System for Document Search:

**1. Start Required Services**:
```bash
tars rag start-chromadb  # Start ChromaDB automatically
ollama serve             # Start Ollama
ollama pull llama3       # Pull embedding model
```

**2. Test Service Connectivity**:
```bash
tars rag test-services   # Verify all services are running
```

**3. Ingest Documents** (programmatic):
```fsharp
let ragService = // Get from DI
let documents = [
    { Id = "doc1"; Content = "Document content"; Metadata = Map.empty }
]
let! result = ragService.IngestDocumentsAsync(documents)
```

**4. Search Documents**:
```fsharp
let! searchResults = ragService.SearchAsync("search query", maxResults = 5)
for result in searchResults do
    printfn "Score: %f, Content: %s" result.Score result.Document.Content
```

**5. CLI Usage**:
```bash
tars rag list-models     # Check available models
tars rag test-tars       # Test with TARS content
```

**Sources**: IRagService.fs, RagService.fs, RagCommand.fs, Types.fs
```

### **Quality Criteria**
- [ ] Shows complete workflow from setup to search
- [ ] Includes both CLI and programmatic usage
- [ ] Provides realistic code examples
- [ ] Explains service dependencies
- [ ] References relevant source files

## 🎯 **TEST SCENARIO EXECUTION PLAN**

### **Automated Testing Approach**
1. **Ingest TARS Knowledge**: Load all identified knowledge sources
2. **Execute Test Queries**: Run each scenario query against RAG system
3. **Evaluate Responses**: Check response quality against criteria
4. **Measure Performance**: Track response time and relevance scores
5. **Report Results**: Generate comprehensive test report

### **Success Metrics**
- **Relevance Score**: >0.8 for retrieved documents
- **Response Completeness**: All expected content areas covered
- **Source Attribution**: Accurate file references
- **Actionability**: Responses enable immediate action
- **Response Time**: <5 seconds per query

### **Quality Evaluation Criteria**
- [ ] **Accuracy**: Information is factually correct
- [ ] **Completeness**: Covers all aspects of the question
- [ ] **Actionability**: Provides concrete, usable guidance
- [ ] **Source Attribution**: Cites relevant source files
- [ ] **Code Examples**: Includes working code snippets
- [ ] **Context Awareness**: Understands TARS-specific patterns

## 🚀 **IMPLEMENTATION PRIORITY**

### **Phase 1: Core Scenarios (Implement First)**
1. Scenario 1: "How do I implement a new command?"
2. Scenario 2: "What commands are available?"

### **Phase 2: Architecture Scenarios**
3. Scenario 3: "How does the F# architecture work?"
4. Scenario 4: "How do I add a new service?"

### **Phase 3: Advanced Usage**
5. Scenario 5: "How do I use the RAG system?"

These test scenarios provide concrete, measurable ways to validate that the TARS RAG system delivers real value to developers working on the TARS project.
