namespace TarsEngine.FSharp.FLUX.Tests

open System
open TarsEngine.FSharp.FLUX.FractalGrammar.SimpleFractalGrammar

/// Tests for Simple Fractal Grammar System
module SimpleFractalGrammarTests =

    /// Test result type
    type TestResult = {
        TestName: string
        Success: bool
        Message: string
        ExecutionTime: TimeSpan
    }

    /// Run a test with error handling
    let runTest testName testFunc =
        let startTime = DateTime.UtcNow
        try
            testFunc()
            {
                TestName = testName
                Success = true
                Message = "Test passed"
                ExecutionTime = DateTime.UtcNow - startTime
            }
        with
        | ex ->
            {
                TestName = testName
                Success = false
                Message = sprintf "Test failed: %s" ex.Message
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Test fractal engine transformations
    let testFractalTransformations() =
        [
            runTest "Scale Transformation - Expand" (fun () ->
                let engine = SimpleFractalEngine()
                let result = engine.ApplyTransformation("test", Scale 2.0)
                if result.Length <= "test".Length then
                    failwith "Scale expansion should increase pattern length"
            )

            runTest "Scale Transformation - Contract" (fun () ->
                let engine = SimpleFractalEngine()
                let result = engine.ApplyTransformation("test pattern", Scale 0.5)
                if result.Length >= "test pattern".Length then
                    failwith "Scale contraction should decrease pattern length"
            )

            runTest "Rotate Transformation - 180 degrees" (fun () ->
                let engine = SimpleFractalEngine()
                let original = "abc"
                let result = engine.ApplyTransformation(original, Rotate 180.0)
                if result <> "cba" then
                    failwith "180-degree rotation should reverse the pattern"
            )

            runTest "Recursive Transformation" (fun () ->
                let engine = SimpleFractalEngine()
                let result = engine.ApplyTransformation("base", Recursive 2)
                if not (result.Contains("base")) then
                    failwith "Recursive transformation should contain original pattern"
                if result.Length <= "base".Length then
                    failwith "Recursive transformation should expand pattern"
            )
        ]

    /// Test fractal rule creation
    let testFractalRules() =
        [
            runTest "Sierpinski Rule Creation" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateSierpinskiRule()
                if rule.Name <> "Sierpinski Triangle" then
                    failwith "Sierpinski rule should have correct name"
                if rule.Dimension <> 1.585 then
                    failwith "Sierpinski rule should have correct dimension"
                if rule.BasePattern <> "△" then
                    failwith "Sierpinski rule should have correct base pattern"
            )

            runTest "Koch Rule Creation" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateKochRule()
                if rule.Name <> "Koch Snowflake" then
                    failwith "Koch rule should have correct name"
                if Math.Abs(rule.Dimension - 1.261) > 0.001 then
                    failwith "Koch rule should have correct dimension"
                if rule.BasePattern <> "─" then
                    failwith "Koch rule should have correct base pattern"
            )

            runTest "Dragon Rule Creation" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateDragonRule()
                if rule.Name <> "Dragon Curve" then
                    failwith "Dragon rule should have correct name"
                if rule.Dimension <> 2.0 then
                    failwith "Dragon rule should have correct dimension"
                if rule.BasePattern <> "F" then
                    failwith "Dragon rule should have correct base pattern"
            )
        ]

    /// Test fractal generation
    let testFractalGeneration() =
        [
            runTest "Generate Sierpinski Fractal" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateSierpinskiRule()
                let result = engine.GenerateFractal(rule)
                
                if not result.Success then
                    failwith "Sierpinski generation should succeed"
                if result.Iterations <= 0 then
                    failwith "Should perform at least one iteration"
                if String.IsNullOrEmpty(result.GeneratedPattern) then
                    failwith "Should generate non-empty pattern"
                if result.FractalDimension <> 1.585 then
                    failwith "Should preserve fractal dimension"
            )

            runTest "Generate Koch Fractal" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateKochRule()
                let result = engine.GenerateFractal(rule)
                
                if not result.Success then
                    failwith "Koch generation should succeed"
                if result.Iterations <= 0 then
                    failwith "Should perform at least one iteration"
                if String.IsNullOrEmpty(result.GeneratedPattern) then
                    failwith "Should generate non-empty pattern"
            )

            runTest "Generate Dragon Fractal" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateDragonRule()
                let result = engine.GenerateFractal(rule)
                
                if not result.Success then
                    failwith "Dragon generation should succeed"
                if result.Iterations <= 0 then
                    failwith "Should perform at least one iteration"
                if String.IsNullOrEmpty(result.GeneratedPattern) then
                    failwith "Should generate non-empty pattern"
                if result.FractalDimension <> 2.0 then
                    failwith "Should preserve fractal dimension"
            )

            runTest "Generate Custom Fractal" (fun () ->
                let engine = SimpleFractalEngine()
                let customRule = {
                    Name = "Test Fractal"
                    BasePattern = "X"
                    RecursivePattern = Some "X Y X"
                    MaxDepth = 3
                    Transformations = [Scale 0.8]
                    Dimension = 1.2
                }
                let result = engine.GenerateFractal(customRule)
                
                if not result.Success then
                    failwith "Custom fractal generation should succeed"
                if result.FractalDimension <> 1.2 then
                    failwith "Should preserve custom dimension"
            )
        ]

    /// Test fractal service
    let testFractalService() =
        [
            runTest "Generate All Examples" (fun () ->
                let service = SimpleFractalService()
                let examples = service.GenerateExamples()
                
                if examples.Length <> 3 then
                    failwith "Should generate exactly 3 examples"
                
                let (sierpinskiName, sierpinskiResult) = examples.[0]
                let (kochName, kochResult) = examples.[1]
                let (dragonName, dragonResult) = examples.[2]
                
                if sierpinskiName <> "Sierpinski" then
                    failwith "First example should be Sierpinski"
                if kochName <> "Koch" then
                    failwith "Second example should be Koch"
                if dragonName <> "Dragon" then
                    failwith "Third example should be Dragon"
                
                if not sierpinskiResult.Success || not kochResult.Success || not dragonResult.Success then
                    failwith "All examples should succeed"
            )

            runTest "Create Custom Rule" (fun () ->
                let service = SimpleFractalService()
                let rule = service.CreateCustomRule("Test", "A", "A B A", 4, 1.5)
                
                if rule.Name <> "Test" then
                    failwith "Custom rule should have correct name"
                if rule.BasePattern <> "A" then
                    failwith "Custom rule should have correct base pattern"
                if rule.RecursivePattern <> Some "A B A" then
                    failwith "Custom rule should have correct recursive pattern"
                if rule.MaxDepth <> 4 then
                    failwith "Custom rule should have correct max depth"
                if rule.Dimension <> 1.5 then
                    failwith "Custom rule should have correct dimension"
            )

            runTest "Generate Custom Fractal" (fun () ->
                let service = SimpleFractalService()
                let result = service.GenerateCustomFractal("Custom", "🌟", "🌟 ✨ 🌟", 3, 1.3)
                
                if not result.Success then
                    failwith "Custom fractal generation should succeed"
                if result.FractalDimension <> 1.3 then
                    failwith "Should have correct dimension"
                if not (result.GeneratedPattern.Contains("🌟")) then
                    failwith "Should contain base pattern"
            )

            runTest "Get Fractal Statistics" (fun () ->
                let service = SimpleFractalService()
                let examples = service.GenerateExamples()
                let results = examples |> List.map snd
                let stats = service.GetFractalStatistics(results)
                
                if not (stats.ContainsKey("total_fractals")) then
                    failwith "Statistics should contain total_fractals"
                if not (stats.ContainsKey("successful_fractals")) then
                    failwith "Statistics should contain successful_fractals"
                if not (stats.ContainsKey("average_dimension")) then
                    failwith "Statistics should contain average_dimension"
                
                let totalFractals = stats.["total_fractals"] :?> int
                if totalFractals <> 3 then
                    failwith "Should have 3 total fractals"
            )
        ]

    /// Test fractal analysis
    let testFractalAnalysis() =
        [
            runTest "Analyze Fractal Result" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateSierpinskiRule()
                let result = engine.GenerateFractal(rule)
                let analysis = engine.AnalyzeFractal(result)
                
                if not (analysis.ContainsKey("pattern_length")) then
                    failwith "Analysis should contain pattern_length"
                if not (analysis.ContainsKey("iterations")) then
                    failwith "Analysis should contain iterations"
                if not (analysis.ContainsKey("fractal_dimension")) then
                    failwith "Analysis should contain fractal_dimension"
                if not (analysis.ContainsKey("execution_time_ms")) then
                    failwith "Analysis should contain execution_time_ms"
                if not (analysis.ContainsKey("complexity_score")) then
                    failwith "Analysis should contain complexity_score"
                if not (analysis.ContainsKey("success")) then
                    failwith "Analysis should contain success"
                
                let success = analysis.["success"] :?> bool
                if not success then
                    failwith "Analysis should show success"
                
                let dimension = analysis.["fractal_dimension"] :?> float
                if dimension <> 1.585 then
                    failwith "Analysis should show correct dimension"
            )
        ]

    /// Test fractal visualization
    let testFractalVisualization() =
        [
            runTest "ASCII Art Conversion" (fun () ->
                let ascii = FractalVisualization.toAsciiArt "Hello World Test" 5
                if String.IsNullOrEmpty(ascii) then
                    failwith "ASCII art should not be empty"
                if not (ascii.Contains("Hello")) then
                    failwith "ASCII art should contain original text"
            )

            runTest "SVG Generation" (fun () ->
                let svg = FractalVisualization.toSvg "ABC" 100 100
                if not (svg.Contains("<svg")) then
                    failwith "Should generate valid SVG"
                if not (svg.Contains("</svg>")) then
                    failwith "Should have closing SVG tag"
                if not (svg.Contains("A")) then
                    failwith "Should contain pattern characters"
            )

            runTest "Tree Visualization" (fun () ->
                let engine = SimpleFractalEngine()
                let rule = engine.CreateSierpinskiRule()
                let result = engine.GenerateFractal(rule)
                let tree = FractalVisualization.toTreeVisualization(result)
                
                if String.IsNullOrEmpty(tree) then
                    failwith "Tree visualization should not be empty"
                if not (tree.Contains("Fractal Tree")) then
                    failwith "Should contain tree title"
                if not (tree.Contains("Dimension")) then
                    failwith "Should contain dimension info"
                if not (tree.Contains("Iterations")) then
                    failwith "Should contain iterations info"
            )
        ]

    /// Run all fractal grammar tests
    let runAllTests() =
        printfn "🧪 TARS Simple Fractal Grammar Tests"
        printfn "====================================="
        printfn ""
        
        let mutable allResults = []
        
        // Transformation tests
        printfn "🔄 Testing Fractal Transformations..."
        let transformationResults = testFractalTransformations()
        allResults <- allResults @ transformationResults
        
        // Rule creation tests
        printfn "📐 Testing Fractal Rules..."
        let ruleResults = testFractalRules()
        allResults <- allResults @ ruleResults
        
        // Generation tests
        printfn "🌀 Testing Fractal Generation..."
        let generationResults = testFractalGeneration()
        allResults <- allResults @ generationResults
        
        // Service tests
        printfn "🔧 Testing Fractal Service..."
        let serviceResults = testFractalService()
        allResults <- allResults @ serviceResults
        
        // Analysis tests
        printfn "📊 Testing Fractal Analysis..."
        let analysisResults = testFractalAnalysis()
        allResults <- allResults @ analysisResults
        
        // Visualization tests
        printfn "🎨 Testing Fractal Visualization..."
        let visualizationResults = testFractalVisualization()
        allResults <- allResults @ visualizationResults
        
        // Report results
        printfn ""
        printfn "📊 TEST RESULTS"
        printfn "==============="
        
        let passed = allResults |> List.filter (fun r -> r.Success) |> List.length
        let failed = allResults |> List.filter (fun r -> not r.Success) |> List.length
        let totalTime = allResults |> List.map (fun r -> r.ExecutionTime.TotalMilliseconds) |> List.sum
        
        for result in allResults do
            let status = if result.Success then "✅ PASS" else "❌ FAIL"
            let time = sprintf "%.1fms" result.ExecutionTime.TotalMilliseconds
            printfn "%s | %s | %s | %s" status result.TestName time result.Message
        
        printfn ""
        printfn "Summary: %d passed, %d failed, %.1fms total" passed failed totalTime
        printfn ""
        
        if failed = 0 then
            printfn "🎉 ALL FRACTAL GRAMMAR TESTS PASSED!"
            printfn "✅ Simple Fractal Grammar System is working correctly!"
        else
            printfn "⚠️  Some tests failed. Please review the implementation."
        
        (passed, failed)
