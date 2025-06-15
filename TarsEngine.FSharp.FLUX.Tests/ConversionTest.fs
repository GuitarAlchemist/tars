namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.Conversion.TarsToFluxConverter

/// Tests for TARS to FLUX conversion
module ConversionTest =
    
    [<Fact>]
    let ``Can convert simple TARS metascript to FLUX`` () =
        // Arrange
        let tarsContent = """DESCRIBE {
    name: "Simple Test"
    version: "1.0"
    description: "A simple test"
}

CONFIG {
    model: "llama3"
    temperature: 0.7
}

VARIABLE greeting {
    value: "Hello, FLUX!"
}

ACTION {
    type: "log"
    message: "${greeting}"
}

FSHARP {
    let x = 42
    printfn "The answer is %d" x
}"""
        
        let tempInput = Path.GetTempFileName()
        let tempOutput = Path.ChangeExtension(tempInput, ".flux")
        
        try
            // Act
            File.WriteAllText(tempInput, tarsContent)
            let result = convertTarsToFlux tempInput tempOutput
            
            // Assert
            result.Success |> should equal true
            result.Errors |> should be Empty
            File.Exists(tempOutput) |> should equal true
            
            let fluxContent = File.ReadAllText(tempOutput)
            fluxContent |> should contain "META {"
            fluxContent |> should contain "config:"
            fluxContent |> should contain "VARIABLE greeting"
            fluxContent |> should contain "FSHARP {"
            fluxContent |> should contain "REASONING {"
            
            printfn "✅ Conversion successful!"
            printfn "Original TARS content: %d characters" tarsContent.Length
            printfn "Converted FLUX content: %d characters" fluxContent.Length
            
        finally
            // Cleanup
            if File.Exists(tempInput) then File.Delete(tempInput)
            if File.Exists(tempOutput) then File.Delete(tempOutput)
    
    [<Fact>]
    let ``Can convert TARS metascripts from .tars directory`` () =
        // Arrange
        let tarsDir = Path.Combine(__SOURCE_DIRECTORY__, "..", ".tars", "metascripts")
        let outputDir = Path.Combine(__SOURCE_DIRECTORY__, "converted-flux")
        
        if Directory.Exists(outputDir) then
            Directory.Delete(outputDir, true)
        
        // Act
        let results = convertDirectoryTarsToFlux tarsDir outputDir
        
        // Assert
        results |> should not' (be Empty)
        
        let successful = results |> List.filter (fun r -> r.Success)
        let failed = results |> List.filter (fun r -> not r.Success)
        
        printfn "🔄 TARS to FLUX Conversion Results:"
        printfn "=================================="
        printfn "Total files: %d" results.Length
        printfn "Successful: %d" successful.Length
        printfn "Failed: %d" failed.Length
        printfn "Success rate: %.1f%%" (float successful.Length / float results.Length * 100.0)
        
        if failed.Length > 0 then
            printfn ""
            printfn "❌ Failed conversions:"
            failed |> List.iter (fun r ->
                printfn "  - %s: %s" (Path.GetFileName(r.OriginalFile)) (String.concat "; " r.Errors))
        
        if successful.Length > 0 then
            printfn ""
            printfn "✅ Successful conversions:"
            successful |> List.take (min 5 successful.Length) |> List.iter (fun r ->
                printfn "  - %s -> %s" (Path.GetFileName(r.OriginalFile)) (Path.GetFileName(r.OutputFile)))
            
            if successful.Length > 5 then
                printfn "  ... and %d more" (successful.Length - 5)
        
        // Generate conversion report
        let report = generateConversionReport results
        let reportPath = Path.Combine(outputDir, "conversion-report.md")
        File.WriteAllText(reportPath, report)
        
        printfn ""
        printfn "📊 Conversion report saved to: %s" reportPath
        
        // At least some conversions should succeed
        successful.Length |> should be (greaterThan 0)
        
        // Verify output directory exists and has files
        Directory.Exists(outputDir) |> should equal true
        Directory.GetFiles(outputDir, "*.flux", SearchOption.AllDirectories).Length |> should be (greaterThan 0)
    
    [<Fact>]
    let ``Converted FLUX metascripts can be executed`` () =
        async {
            // Arrange - Create a simple TARS metascript and convert it
            let tarsContent = """DESCRIBE {
    name: "Executable Test"
    version: "1.0"
    description: "Test that converted FLUX can be executed"
}

FSHARP {
    printfn "🚀 Converted TARS metascript executing as FLUX!"
    printfn "This demonstrates successful TARS -> FLUX conversion"
    let result = 2 + 2
    printfn "Calculation result: %d" result
}"""
            
            let tempInput = Path.GetTempFileName()
            let tempOutput = Path.ChangeExtension(tempInput, ".flux")
            
            try
                File.WriteAllText(tempInput, tarsContent)
                let conversionResult = convertTarsToFlux tempInput tempOutput
                
                // Verify conversion succeeded
                conversionResult.Success |> should equal true
                
                // Act - Execute the converted FLUX metascript
                let engine = TarsEngine.FSharp.FLUX.FluxEngine.FluxEngine()
                let! executionResult = engine.ExecuteFile(tempOutput) |> Async.AwaitTask
                
                // Assert
                executionResult |> should not' (equal null)
                executionResult.Success |> should equal true
                executionResult.BlocksExecuted |> should be (greaterThan 0)
                
                printfn "🎉 TARS -> FLUX Conversion and Execution Success!"
                printfn "=================================="
                printfn "✅ TARS metascript converted to FLUX"
                printfn "✅ FLUX metascript executed successfully"
                printfn "✅ Blocks executed: %d" executionResult.BlocksExecuted
                printfn "✅ Execution time: %A" executionResult.ExecutionTime
                
            finally
                if File.Exists(tempInput) then File.Delete(tempInput)
                if File.Exists(tempOutput) then File.Delete(tempOutput)
        }
