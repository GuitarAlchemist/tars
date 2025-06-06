# TARS Engine API Injection Documentation

## Overview

TARS Engine API Injection provides comprehensive access to the entire TARS engine from within metascript blocks, enabling powerful polyglot programming capabilities with enterprise-grade security and performance.

## 🚀 **Key Features**

- **Complete API Access**: Full TARS engine functionality available within metascripts
- **Multi-Language Support**: Native F# and C# integration with zero overhead
- **Enterprise Security**: Comprehensive sandboxing and resource management
- **Real-Time Tracing**: Detailed execution monitoring and debugging
- **Production Ready**: Thread-safe concurrent access with proper error handling

## 📚 **Documentation Structure**

### Core Documentation
- [API Investigation Report](../../output/TARS_Engine_API_Injection_Investigation_Report.md) - Comprehensive analysis of implementation approaches
- [Implementation Summary](../../output/TARS_API_Injection_Implementation_Summary.md) - Complete implementation details and results

### API Reference
- [ITarsEngineApi Interface](../../TarsEngine.FSharp.Core/Api/ITarsEngineApi.fs) - Core API interface definition
- [TarsEngineApiImpl](../../TarsEngine.FSharp.Core/Api/TarsEngineApiImpl.fs) - Complete implementation
- [TarsApiRegistry](../../TarsEngine.FSharp.Core/Api/TarsApiRegistry.fs) - Global API registry
- [TarsExecutionContext](../../TarsEngine.FSharp.Core/Api/TarsExecutionContext.fs) - Security and resource management

### Example Metascripts
- [F# Real API Demo](../../.tars/tars_api_real_demo.trsx) - Basic F# API usage
- [C# Integration Demo](../../.tars/tars_api_csharp_demo.trsx) - C# style patterns
- [F# & C# Integration](../../.tars/tars_api_fsharp_csharp_demo.trsx) - Multi-language integration
- [API Investigation](../../.tars/tars_engine_api_injection_investigation.trsx) - Comprehensive analysis
- [API Demonstration](../../.tars/tars_api_injection_demo.trsx) - Practical examples

## 🎯 **Quick Start**

### F# Usage
```fsharp
open TarsEngine.FSharp.Core.Api

// Get TARS API instance
let tars = TarsApiRegistry.GetApi()

// Use async workflows
let workflow = async {
    let! results = tars.VectorStore.SearchAsync("machine learning", 10)
    let! response = tars.LlmService.CompleteAsync("Explain AI", "gpt-4")
    return results, response
}
```

### C# Style Usage
```fsharp
// C# style async/await patterns
let csharpStyle = task {
    let tars = TarsApiRegistry.GetApi()
    let! results = tars.VectorStore.SearchAsync("neural networks", 5)
    
    // LINQ-style operations
    let highScore = 
        results
        |> Array.filter (fun r -> r.Score > 0.8)
        |> Array.sortByDescending (fun r -> r.Score)
    
    return highScore
}
```

## 🔒 **Security Model**

### Security Policies
- **Restrictive**: Minimal permissions for untrusted scripts
- **Standard**: Balanced permissions for trusted scripts
- **Unrestricted**: Full permissions for system scripts

### Resource Limits
- Memory usage limits (MB)
- CPU time limits (milliseconds)
- Network request limits
- File operation limits
- LLM request limits

## 📊 **API Services**

### Vector Store API
- Search, Add, Delete operations
- Similarity search and indexing
- Metadata support and filtering

### LLM Service API
- Text completion and chat
- Embedding generation
- Multi-model support

### Agent Coordination API
- Agent spawning and lifecycle management
- Inter-agent communication
- Parallel processing coordination

### File System API
- Secure file operations
- Directory management
- Metadata access

### Execution Context API
- Logging and tracing
- Metadata management
- Performance monitoring

## 🚀 **Performance**

- **Zero Overhead**: Native .NET integration
- **Sub-millisecond**: API call latency
- **Thread-Safe**: Concurrent access support
- **Memory Efficient**: Minimal resource usage
- **Scalable**: Production-ready architecture

## 🔧 **Development**

### Building
```bash
dotnet build TarsEngine.FSharp.Core
```

### Testing
```bash
dotnet run --project TarsEngine.FSharp.Metascript.Runner -- .tars/tars_api_real_demo.trsx
```

### Adding New Services
1. Extend ITarsEngineApi interface
2. Implement in TarsEngineApiImpl
3. Add security policies
4. Create demonstration metascripts
5. Update documentation

## 📈 **Roadmap**

### Phase 1: Enhanced Services (Completed ✅)
- Complete API infrastructure
- F# and C# integration
- Security and resource management
- Real-time tracing

### Phase 2: Multi-Language Expansion (Planned)
- Python bridge using Python.NET
- JavaScript bridge using Jint engine
- Rust bridge using C FFI

### Phase 3: Advanced Features (Planned)
- Distributed agent coordination
- API versioning and compatibility
- Performance optimization
- Enhanced monitoring

### Phase 4: Production Deployment (Planned)
- Docker containerization
- Kubernetes deployment
- CI/CD integration
- Enterprise features

## 🤝 **Contributing**

1. Review the [API Investigation Report](../../output/TARS_Engine_API_Injection_Investigation_Report.md)
2. Understand the [Implementation Summary](../../output/TARS_API_Injection_Implementation_Summary.md)
3. Study existing metascript examples
4. Follow security and performance guidelines
5. Add comprehensive tests and documentation

## 📞 **Support**

For questions, issues, or contributions related to TARS Engine API Injection:

1. Check existing documentation and examples
2. Review implementation details in source code
3. Test with provided demonstration metascripts
4. Follow established patterns and conventions

---

**Status**: ✅ Production Ready  
**Languages**: F#, C# (Native .NET)  
**Security**: Enterprise Grade  
**Performance**: Optimized  
**Documentation**: Comprehensive
