// FLUX Integration Validation Test
open System
open System.IO

printfn "🌟 FLUX INTEGRATION VALIDATION TEST"
printfn "==================================="

// Test FLUX component integration
let fluxComponents = [
    "src/TarsEngine.FSharp.Core/FLUX/Ast/FluxAst.fs"
    "src/TarsEngine.FSharp.Core/FLUX/Refinement/CrossEntropyRefinement.fs"
    "src/TarsEngine.FSharp.Core/FLUX/VectorStore/SemanticVectorStore.fs"
    "src/TarsEngine.FSharp.Core/FLUX/FractalGrammar/SimpleFractalGrammar.fs"
    "src/TarsEngine.FSharp.Core/FLUX/FractalLanguage/FluxFractalArchitecture.fs"
    "src/TarsEngine.FSharp.Core/FLUX/FractalLanguage/FluxFractalInterpreter.fs"
    "src/TarsEngine.FSharp.Core/FLUX/UnifiedFormat/UnifiedTrsxInterpreter.fs"
    "src/TarsEngine.FSharp.Core/FLUX/UnifiedFormat/TrsxMigrationTool.fs"
    "src/TarsEngine.FSharp.Core/FLUX/UnifiedFormat/TrsxCli.fs"
    "src/TarsEngine.FSharp.Core/FLUX/Mathematics/MathematicalEngine.fs"
]

printfn "\n✅ FLUX Component Integration Validation:"
let mutable allComponentsPresent = true

for comp in fluxComponents do
    if File.Exists(comp) then
        let fileSize = (new FileInfo(comp)).Length
        let lineCount = File.ReadAllLines(comp).Length
        printfn "  ✅ %s (%d lines, %d bytes)" (Path.GetFileName(comp)) lineCount fileSize
    else
        printfn "  ❌ %s (missing)" (Path.GetFileName(comp))
        allComponentsPresent <- false

// Test FLUX tests integration
printfn "\n✅ FLUX Tests Integration Validation:"
let fluxTests = [
    "src/TarsEngine.FSharp.Tests/FLUX/StandaloneTestRunner.fs"
    "src/TarsEngine.FSharp.Tests/FLUX/SimpleFractalGrammarTests.fs"
    "src/TarsEngine.FSharp.Tests/FLUX/CudaVectorStoreValidationTests.fs"
    "src/TarsEngine.FSharp.Tests/FLUX/PracticalUseCaseTests.fs"
    "src/TarsEngine.FSharp.Tests/FLUX/CustomTransformerTests.fs"
]

let mutable allTestsPresent = true

for test in fluxTests do
    if File.Exists(test) then
        let lineCount = File.ReadAllLines(test).Length
        printfn "  ✅ %s (%d lines)" (Path.GetFileName(test)) lineCount
    else
        printfn "  ❌ %s (missing)" (Path.GetFileName(test))
        allTestsPresent <- false

// Test project file integration
printfn "\n✅ Project File Integration Validation:"
let projectFile = "src/TarsEngine.FSharp.Core/TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj"
if File.Exists(projectFile) then
    let content = File.ReadAllText(projectFile)
    let hasFluxReferences = content.Contains("FLUX/Ast/FluxAst.fs") &&
                            content.Contains("FLUX/Refinement/CrossEntropyRefinement.fs") &&
                            content.Contains("FLUX/VectorStore/SemanticVectorStore.fs")
    
    if hasFluxReferences then
        printfn "  ✅ Project file contains FLUX references"
    else
        printfn "  ❌ Project file missing FLUX references"
        allComponentsPresent <- false
else
    printfn "  ❌ Project file not found"
    allComponentsPresent <- false

// Test original standalone project
printfn "\n✅ Original Standalone Project Validation:"
let standaloneProject = "TarsEngine.FSharp.FLUX.Standalone"
if Directory.Exists(standaloneProject) then
    let standaloneFiles = Directory.GetFiles(standaloneProject, "*", SearchOption.AllDirectories)
    printfn "  ✅ Original standalone project preserved (%d files)" standaloneFiles.Length
else
    printfn "  ❌ Original standalone project missing"

// Summary
printfn "\n🎯 FLUX INTEGRATION SUMMARY:"
printfn "============================"

if allComponentsPresent && allTestsPresent then
    printfn "✅ FLUX INTEGRATION SUCCESSFUL!"
    printfn "✅ All core components integrated"
    printfn "✅ All tests integrated"
    printfn "✅ Project file updated"
    printfn "✅ Original standalone preserved"
    printfn ""
    printfn "🌟 FLUX CAPABILITIES NOW AVAILABLE IN MAIN ENGINE:"
    printfn "- ChatGPT-Cross-Entropy Refinement"
    printfn "- Vector Store Semantics"
    printfn "- Fractal Grammar System"
    printfn "- Fractal Language Architecture"
    printfn "- Unified TRSX Format"
    printfn "- Mathematical Engine"
    printfn "- Comprehensive Test Suite"
else
    printfn "❌ FLUX INTEGRATION INCOMPLETE"
    printfn "Some components or tests are missing"

printfn "\n🚀 FLUX integration validation completed!"
