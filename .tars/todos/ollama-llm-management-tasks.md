# Ollama LLM Management Tasks for F# RAG Engine

## 🎯 **MAIN OBJECTIVE**
Steal and convert Ollama LLM management functionality from `tars_fucked_up` to F# and integrate into the RAG command.

## 📋 **TASK BREAKDOWN**

### **Phase 1: Analysis and Planning (5 tasks)**

#### Task 1.1: Analyze Existing Ollama Services
- [ ] Review `TarsCli/Services/OllamaService.cs` from fucked up TARS
- [ ] Review `TarsCli/Services/DockerModelRunnerService.cs` 
- [ ] Document API endpoints used for model management
- [ ] Document data structures and response types
- [ ] Create mapping of C# to F# conversion needs

#### Task 1.2: Analyze Existing CLI Commands
- [ ] Review `TarsCli/Commands/LlmCommand.cs` structure
- [ ] Review `TarsCli/Commands/DockerModelRunnerCommand.cs` structure
- [ ] Document command patterns and argument handling
- [ ] Document console output formatting
- [ ] Create F# command structure plan

#### Task 1.3: Identify Core Functionality to Steal
- [ ] List Models (`/api/tags` endpoint)
- [ ] Pull Models (`/api/pull` endpoint)
- [ ] Model Info (model metadata and status)
- [ ] Model Status (availability checking)
- [ ] Error handling patterns

#### Task 1.4: Design F# Integration Strategy
- [ ] Plan integration with existing RAG command
- [ ] Design F# type definitions for Ollama responses
- [ ] Plan service layer architecture
- [ ] Design command argument structure
- [ ] Plan error handling and logging

#### Task 1.5: Create Implementation Plan
- [ ] Define file structure for new F# services
- [ ] Plan testing strategy
- [ ] Define success criteria for each feature
- [ ] Create implementation order priority
- [ ] Document dependencies and prerequisites

### **Phase 2: F# Type Definitions (3 tasks)**

#### Task 2.1: Create Ollama API Types
- [ ] Create `OllamaModel` record type
- [ ] Create `OllamaTagsResponse` record type
- [ ] Create `OllamaPullRequest` record type
- [ ] Create `OllamaPullResponse` record type
- [ ] Add JSON serialization attributes

#### Task 2.2: Create Model Management Types
- [ ] Create `ModelInfo` record type
- [ ] Create `ModelStatus` discriminated union
- [ ] Create `ModelCategory` discriminated union
- [ ] Create `PullProgress` record type
- [ ] Create error types for model operations

#### Task 2.3: Create Service Interface Types
- [ ] Create `IOllamaModelService` interface
- [ ] Define async method signatures
- [ ] Add proper F# async/task patterns
- [ ] Include logging and error handling
- [ ] Add configuration types

### **Phase 3: F# Service Implementation (4 tasks)**

#### Task 3.1: Implement Basic Ollama HTTP Client
- [ ] Create `OllamaHttpClient` class
- [ ] Implement connection testing
- [ ] Implement basic HTTP operations
- [ ] Add timeout and retry logic
- [ ] Add proper error handling

#### Task 3.2: Implement Model Listing Service
- [ ] Implement `ListModelsAsync` method
- [ ] Parse `/api/tags` response
- [ ] Handle empty model lists
- [ ] Add model categorization logic
- [ ] Add recommended models fallback

#### Task 3.3: Implement Model Pulling Service
- [ ] Implement `PullModelAsync` method
- [ ] Handle `/api/pull` endpoint
- [ ] Add progress tracking (if possible)
- [ ] Handle long-running operations
- [ ] Add pull status verification

#### Task 3.4: Implement Model Info Service
- [ ] Implement `GetModelInfoAsync` method
- [ ] Combine model metadata
- [ ] Add model size and context info
- [ ] Add model availability checking
- [ ] Format model descriptions

### **Phase 4: CLI Command Integration (5 tasks)**

#### Task 4.1: Enhance RAG Command Structure
- [ ] Add new subcommands to RAG command
- [ ] Update help text and examples
- [ ] Add argument parsing for model names
- [ ] Update command validation
- [ ] Add command aliases

#### Task 4.2: Implement List Models Command
- [ ] Add `list-models` subcommand
- [ ] Format model output in tables
- [ ] Add model status indicators
- [ ] Show recommended models when empty
- [ ] Add filtering options

#### Task 4.3: Implement Pull Model Command
- [ ] Add `pull <model>` subcommand
- [ ] Add progress indicators
- [ ] Handle pull failures gracefully
- [ ] Verify successful pulls
- [ ] Add pull recommendations

#### Task 4.4: Implement Model Info Command
- [ ] Add `model-info <model>` subcommand
- [ ] Display comprehensive model information
- [ ] Show model availability status
- [ ] Add context length and capabilities
- [ ] Format output professionally

#### Task 4.5: Integrate with Existing RAG Tests
- [ ] Update `test-services` to show model status
- [ ] Add model availability to service tests
- [ ] Update setup instructions with model pulling
- [ ] Add model recommendations to help
- [ ] Update error messages with model suggestions

### **Phase 5: Testing and Validation (4 tasks)**

#### Task 5.1: Unit Testing
- [ ] Test Ollama API type serialization
- [ ] Test service method implementations
- [ ] Test error handling scenarios
- [ ] Test timeout and retry logic
- [ ] Mock HTTP responses for testing

#### Task 5.2: Integration Testing
- [ ] Test with real Ollama instance
- [ ] Test model listing with various states
- [ ] Test model pulling end-to-end
- [ ] Test error scenarios (offline, invalid models)
- [ ] Test command line interface

#### Task 5.3: User Experience Testing
- [ ] Test help text and examples
- [ ] Test error messages and guidance
- [ ] Test output formatting and readability
- [ ] Test command discoverability
- [ ] Test workflow completeness

#### Task 5.4: Performance Testing
- [ ] Test large model list handling
- [ ] Test long-running pull operations
- [ ] Test concurrent operations
- [ ] Test memory usage
- [ ] Test timeout handling

### **Phase 6: Documentation and Polish (3 tasks)**

#### Task 6.1: Update Documentation
- [ ] Update RAG command help text
- [ ] Add model management examples
- [ ] Update setup instructions
- [ ] Add troubleshooting guide
- [ ] Document recommended models

#### Task 6.2: Add Professional Polish
- [ ] Improve console output formatting
- [ ] Add progress indicators and spinners
- [ ] Add colored output for status
- [ ] Improve error messages
- [ ] Add success confirmations

#### Task 6.3: Final Integration
- [ ] Ensure all commands work together
- [ ] Test complete RAG workflow with models
- [ ] Verify build and deployment
- [ ] Update project documentation
- [ ] Create usage examples

## 🎯 **SUCCESS CRITERIA**

### **Minimum Viable Product (MVP)**
- [ ] `tars rag list-models` - Shows available Ollama models
- [ ] `tars rag pull <model>` - Downloads Ollama models
- [ ] `tars rag model-info <model>` - Shows model information
- [ ] Integration with existing RAG test commands
- [ ] Professional error handling and user guidance

### **Full Feature Set**
- [ ] All MVP features working perfectly
- [ ] Progress indicators for long operations
- [ ] Comprehensive model recommendations
- [ ] Integration with RAG pipeline testing
- [ ] Professional CLI experience matching other commands

## 📊 **IMPLEMENTATION PRIORITY**

### **High Priority (Implement First)**
1. Task 2.1: Create Ollama API Types
2. Task 3.2: Implement Model Listing Service
3. Task 4.2: Implement List Models Command
4. Task 5.2: Integration Testing

### **Medium Priority (Implement Second)**
1. Task 3.3: Implement Model Pulling Service
2. Task 4.3: Implement Pull Model Command
3. Task 4.4: Implement Model Info Command
4. Task 4.5: Integrate with Existing RAG Tests

### **Low Priority (Polish and Enhancement)**
1. Task 6.1: Update Documentation
2. Task 6.2: Add Professional Polish
3. Task 5.4: Performance Testing
4. Task 6.3: Final Integration

## 🚀 **NEXT IMMEDIATE ACTIONS**

1. **Start with Task 2.1**: Create Ollama API Types in F#
2. **Then Task 3.2**: Implement Model Listing Service
3. **Then Task 4.2**: Implement List Models Command
4. **Test incrementally** after each task

This decomposition ensures we can implement the Ollama LLM management functionality incrementally, testing each piece as we go, and building up to a complete professional solution.
