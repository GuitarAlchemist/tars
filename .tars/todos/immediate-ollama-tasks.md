# Immediate Ollama LLM Management Tasks

## 🎯 **IMMEDIATE FOCUS: MVP Implementation**

Let's implement the core Ollama LLM management functionality in small, testable increments.

## 📋 **NEXT 4 TASKS (Start Here)**

### **Task 1: Create Ollama API Types (30 minutes)**
**File**: `TarsEngine.FSharp.RAG/RAG/Services/OllamaTypes.fs`

**Subtasks**:
- [ ] Create `OllamaModel` record type
- [ ] Create `OllamaTagsResponse` record type  
- [ ] Create `OllamaPullRequest` record type
- [ ] Add JSON serialization attributes
- [ ] Test compilation

**Success Criteria**: Types compile and can be used in service implementations

### **Task 2: Implement Model Listing Service (45 minutes)**
**File**: `TarsEngine.FSharp.RAG/RAG/Services/OllamaModelService.fs`

**Subtasks**:
- [ ] Create `OllamaModelService` class
- [ ] Implement `ListModelsAsync` method
- [ ] Parse `/api/tags` response using new types
- [ ] Handle empty responses gracefully
- [ ] Add basic error handling

**Success Criteria**: Can successfully list models from running Ollama instance

### **Task 3: Add List Models to RAG Command (30 minutes)**
**File**: `TarsEngine.FSharp.Cli/Commands/RagCommand.fs`

**Subtasks**:
- [ ] Add `list-models` subcommand
- [ ] Integrate with OllamaModelService
- [ ] Format output in readable table
- [ ] Add to help text and examples
- [ ] Handle service errors gracefully

**Success Criteria**: `tars rag list-models` command works and shows available models

### **Task 4: Test and Validate (15 minutes)**

**Subtasks**:
- [ ] Test with running Ollama instance
- [ ] Test with offline Ollama
- [ ] Test with no models installed
- [ ] Verify error messages are helpful
- [ ] Test help text is accurate

**Success Criteria**: Command works reliably in all scenarios

## 🚀 **IMPLEMENTATION ORDER**

1. **Start with Task 1** - Create the F# types (foundation)
2. **Then Task 2** - Implement the service (core functionality)  
3. **Then Task 3** - Add to CLI command (user interface)
4. **Finally Task 4** - Test everything works

## 📊 **ESTIMATED TIME**
- **Total**: ~2 hours for MVP
- **Each task**: 15-45 minutes
- **Testing**: 15 minutes per task

## 🎯 **AFTER MVP COMPLETION**

Once these 4 tasks are done, we'll have:
- ✅ Working `tars rag list-models` command
- ✅ F# service architecture for Ollama
- ✅ Foundation for adding more commands

**Next features to add**:
1. `tars rag pull <model>` command
2. `tars rag model-info <model>` command  
3. Integration with RAG testing

## 💡 **IMPLEMENTATION NOTES**

- **Keep it simple**: Start with basic functionality, add polish later
- **Test incrementally**: Test each task before moving to the next
- **Reuse patterns**: Follow the same patterns as existing RAG commands
- **Handle errors gracefully**: Always provide helpful error messages
- **Document as you go**: Update help text and examples

Ready to start with **Task 1: Create Ollama API Types**? 🚀
