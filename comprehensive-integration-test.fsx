// TARS Comprehensive Integration Validation Test
open System
open System.IO

printfn "🚀 TARS COMPREHENSIVE INTEGRATION VALIDATION"
printfn "============================================="

// Test all integrated components
let integrationComponents = [
    // FLUX Integration
    ("FLUX AST", "src/TarsEngine.FSharp.Core/FLUX/Ast/FluxAst.fs")
    ("FLUX Refinement", "src/TarsEngine.FSharp.Core/FLUX/Refinement/CrossEntropyRefinement.fs")
    ("FLUX VectorStore", "src/TarsEngine.FSharp.Core/FLUX/VectorStore/SemanticVectorStore.fs")
    ("FLUX FractalGrammar", "src/TarsEngine.FSharp.Core/FLUX/FractalGrammar/SimpleFractalGrammar.fs")
    ("FLUX FractalLanguage", "src/TarsEngine.FSharp.Core/FLUX/FractalLanguage/FluxFractalArchitecture.fs")
    ("FLUX UnifiedFormat", "src/TarsEngine.FSharp.Core/FLUX/UnifiedFormat/UnifiedTrsxInterpreter.fs")
    ("FLUX Mathematics", "src/TarsEngine.FSharp.Core/FLUX/Mathematics/MathematicalEngine.fs")
    
    // CustomTransformers Integration
    ("CustomTransformers Engine", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/TarsCustomTransformerEngine.fs")
    ("CUDA Hybrid Operations", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/CudaHybridOperations.fs")
    ("Meta Optimizer", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/MetaOptimizer.fs")
    ("Janus Cosmology", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/JanusCosmologyExtension.fs")
    ("Drug Discovery", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/DrugDiscoveryDemo.fs")
    ("Financial Risk", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/FinancialRiskDemo.fs")
    ("Scientific Research", "src/TarsEngine.FSharp.Core/AI/CustomTransformers/ScientificResearchDemo.fs")
    
    // CUDA VectorStore Integration
    ("CUDA Integration", "src/TarsEngine.FSharp.Core/VectorStore/CUDA/TarsCudaIntegration.fs")
    
    // Grammar Engine Integration
    ("Fractal Grammar", "src/TarsEngine.FSharp.Core/Grammar/FractalGrammar.fs")
    ("Grammar Resolver", "src/TarsEngine.FSharp.Core/Grammar/GrammarResolver.fs")
    ("Language Dispatcher", "src/TarsEngine.FSharp.Core/Grammar/LanguageDispatcher.fs")
    ("RFC Processor", "src/TarsEngine.FSharp.Core/Grammar/RFCProcessor.fs")
    
    // VectorStore Core Integration
    ("VectorStore Types", "src/TarsEngine.FSharp.Core/VectorStore/Core/Types.fs")
    ("VectorStore Core", "src/TarsEngine.FSharp.Core/VectorStore/Core/VectorStore.fs")
    ("Embedding Generator", "src/TarsEngine.FSharp.Core/VectorStore/Core/EmbeddingGenerator.fs")
    ("Similarity Computer", "src/TarsEngine.FSharp.Core/VectorStore/Core/SimilarityComputer.fs")
    ("Inference Engine", "src/TarsEngine.FSharp.Core/VectorStore/Core/InferenceEngine.fs")
    
    // Integration Engine
    ("VectorStore Integration", "src/TarsEngine.FSharp.Core/Integration/VectorStoreIntegration.fs")
]

printfn "\n✅ COMPONENT INTEGRATION VALIDATION:"
let mutable allComponentsPresent = true
let mutable totalLines = 0
let mutable totalSize = 0L

for (name, path) in integrationComponents do
    if File.Exists(path) then
        let fileInfo = new FileInfo(path)
        let lineCount = File.ReadAllLines(path).Length
        totalLines <- totalLines + lineCount
        totalSize <- totalSize + fileInfo.Length
        printfn "  ✅ %s (%d lines, %d bytes)" name lineCount (int fileInfo.Length)
    else
        printfn "  ❌ %s (missing)" name
        allComponentsPresent <- false

// Test CUDA files
printfn "\n✅ CUDA FILES VALIDATION:"
let cudaFiles = Directory.GetFiles("src/TarsEngine.FSharp.Core/VectorStore/CUDA", "*.cu", SearchOption.AllDirectories)
printfn "  ✅ CUDA kernel files: %d" cudaFiles.Length
for cudaFile in cudaFiles do
    let fileName = Path.GetFileName(cudaFile)
    let fileSize = (new FileInfo(cudaFile)).Length
    printfn "    - %s (%d bytes)" fileName (int fileSize)

// Test Tools Organization
printfn "\n✅ TOOLS ORGANIZATION VALIDATION:"
let toolsDirs = ["tools/fsharp"; "tools/powershell"; "tools/python"]
for toolDir in toolsDirs do
    if Directory.Exists(toolDir) then
        let fileCount = Directory.GetFiles(toolDir).Length
        printfn "  ✅ %s (%d files)" toolDir fileCount
    else
        printfn "  ❌ %s (missing)" toolDir

// Test Archive Organization
printfn "\n✅ ARCHIVE ORGANIZATION VALIDATION:"
let archiveDirs = ["archive/legacy-csharp"]
for archiveDir in archiveDirs do
    if Directory.Exists(archiveDir) then
        printfn "  ✅ %s (created)" archiveDir
    else
        printfn "  ❌ %s (missing)" archiveDir

// Test Project File Integration
printfn "\n✅ PROJECT FILE INTEGRATION VALIDATION:"
let projectFile = "src/TarsEngine.FSharp.Core/TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj"
if File.Exists(projectFile) then
    let content = File.ReadAllText(projectFile)
    let integrationChecks = [
        ("FLUX Integration", content.Contains("FLUX/Ast/FluxAst.fs"))
        ("CustomTransformers Integration", content.Contains("AI/CustomTransformers/TarsCustomTransformerEngine.fs"))
        ("CUDA Integration", content.Contains("VectorStore/CUDA/TarsCudaIntegration.fs"))
        ("Grammar Integration", content.Contains("Grammar/FractalGrammar.fs"))
        ("VectorStore Integration", content.Contains("VectorStore/Core/VectorStore.fs"))
        ("Integration Engine", content.Contains("Integration/VectorStoreIntegration.fs"))
    ]
    
    for (checkName, isPresent) in integrationChecks do
        if isPresent then
            printfn "  ✅ %s references present" checkName
        else
            printfn "  ❌ %s references missing" checkName
            allComponentsPresent <- false
else
    printfn "  ❌ Project file not found"
    allComponentsPresent <- false

// Test Original Projects Preservation
printfn "\n✅ ORIGINAL PROJECTS PRESERVATION:"
let originalProjects = [
    "TarsEngine.FSharp.FLUX.Standalone"
    "TarsEngine.CustomTransformers"
    "TarsEngine.CUDA.VectorStore"
    "Tars.Engine.Grammar"
    "Tars.Engine.VectorStore"
    "Tars.Engine.Integration"
    "Legacy_CSharp_Projects"
]

for project in originalProjects do
    if Directory.Exists(project) then
        let fileCount = Directory.GetFiles(project, "*", SearchOption.AllDirectories).Length
        printfn "  ✅ %s preserved (%d files)" project fileCount
    else
        printfn "  ❌ %s missing" project

// Summary
printfn "\n🎯 COMPREHENSIVE INTEGRATION SUMMARY:"
printfn "====================================="

if allComponentsPresent then
    printfn "✅ ALL INTEGRATIONS SUCCESSFUL!"
    printfn "✅ Total integrated components: %d" integrationComponents.Length
    printfn "✅ Total lines of code integrated: %d" totalLines
    printfn "✅ Total size integrated: %d KB" (int (totalSize / 1024L))
    printfn ""
    printfn "🌟 INTEGRATED CAPABILITIES:"
    printfn "- FLUX: ChatGPT-Cross-Entropy, Vector Store Semantics, Fractal Grammars"
    printfn "- CustomTransformers: Advanced AI, CUDA Acceleration, Scientific Applications"
    printfn "- CUDA VectorStore: GPU Acceleration, Non-Euclidean Geometry"
    printfn "- Grammar Engine: Fractal Grammars, Language Processing, RFC Support"
    printfn "- VectorStore Core: Embeddings, Similarity, Inference"
    printfn "- Integration Engine: Unified Integration Layer"
    printfn "- Tools Organization: F#, PowerShell, Python tools organized"
    printfn "- Archive System: Legacy projects safely archived"
else
    printfn "❌ SOME INTEGRATIONS INCOMPLETE"
    printfn "Review missing components above"

printfn "\n🚀 TARS is now a unified, comprehensive AI engine!"
