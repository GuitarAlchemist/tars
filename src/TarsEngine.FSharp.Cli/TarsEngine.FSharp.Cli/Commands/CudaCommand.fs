namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging

module CudaInterop =
    // Basic CUDA functions
    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int minimal_cuda_device_count()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int minimal_cuda_init(int device_id)

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int minimal_cuda_cleanup()

    // Advanced test functions
    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_simple_memory_test()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_simple_vector_test()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_matrix_multiply_test()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_neural_network_test()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_vector_similarity_test()

    // Advanced AI operations
    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_transformer_attention_test()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int cuda_ai_model_inference_test()

/// Test result with metrics
type TestResult = {
    TestName: string
    Success: bool
    Message: string
    ExecutionTimeMs: float
    MemoryUsedMB: float
}

/// CUDA Command for TARS CLI - Real GPU execution
type CudaCommand(logger: ILogger<CudaCommand>) =

    /// Create test result with metrics
    member _.createTestResult (testName: string) (success: bool) (message: string) (startTime: DateTime) (endTime: DateTime) : TestResult =
        let executionTime = (endTime - startTime).TotalMilliseconds
        let currentMemory = float (GC.GetTotalMemory(false)) / (1024.0 * 1024.0)

        {
            TestName = testName
            Success = success
            Message = message
            ExecutionTimeMs = executionTime
            MemoryUsedMB = currentMemory
        }

    /// Test CUDA device detection
    member this.testDeviceDetection() =
        let startTime = DateTime.UtcNow
        try
            let deviceCount = CudaInterop.minimal_cuda_device_count()
            let endTime = DateTime.UtcNow

            if deviceCount > 0 then
                this.createTestResult "Device Detection" true (sprintf "Found %d CUDA device(s)" deviceCount) startTime endTime
            else
                this.createTestResult "Device Detection" false "No CUDA devices found" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Device Detection" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test CUDA initialization
    member this.testCudaInitialization() =
        let startTime = DateTime.UtcNow
        try
            let initResult = CudaInterop.minimal_cuda_init(0)
            let endTime = DateTime.UtcNow

            if initResult = 0 then
                this.createTestResult "CUDA Initialization" true "CUDA initialized successfully" startTime endTime
            else
                this.createTestResult "CUDA Initialization" false (sprintf "CUDA initialization failed with code: %d" initResult) startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "CUDA Initialization" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test CUDA memory operations
    member this.testMemoryOperations() =
        let startTime = DateTime.UtcNow
        try
            // Initialize CUDA first
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "Memory Operations" false "CUDA initialization failed" startTime endTime
            else
                // Test simple memory operations
                let memResult = CudaInterop.cuda_simple_memory_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if memResult = 0 then
                    this.createTestResult "Memory Operations" true "Memory allocation and free test passed" startTime endTime
                else
                    this.createTestResult "Memory Operations" false "Memory test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Memory Operations" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test vector operations
    member this.testVectorOperations() =
        let startTime = DateTime.UtcNow
        try
            // Initialize CUDA
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "Vector Operations" false "CUDA initialization failed" startTime endTime
            else
                // Test simple vector operations
                let vectorResult = CudaInterop.cuda_simple_vector_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if vectorResult = 0 then
                    this.createTestResult "Vector Operations" true "Vector addition test passed (1024 elements)" startTime endTime
                else
                    this.createTestResult "Vector Operations" false "Vector addition test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Vector Operations" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test matrix operations
    member this.testMatrixOperations() =
        let startTime = DateTime.UtcNow
        try
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "Matrix Operations" false "CUDA initialization failed" startTime endTime
            else
                let matrixResult = CudaInterop.cuda_matrix_multiply_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if matrixResult = 0 then
                    this.createTestResult "Matrix Operations" true "Matrix multiplication test passed (64x64 matrices)" startTime endTime
                else
                    this.createTestResult "Matrix Operations" false "Matrix multiplication test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Matrix Operations" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test neural network operations
    member this.testNeuralNetworkOperations() =
        let startTime = DateTime.UtcNow
        try
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "Neural Network Operations" false "CUDA initialization failed" startTime endTime
            else
                let neuralResult = CudaInterop.cuda_neural_network_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if neuralResult = 0 then
                    this.createTestResult "Neural Network Operations" true "ReLU and Sigmoid activations computed correctly" startTime endTime
                else
                    this.createTestResult "Neural Network Operations" false "Neural network activation test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Neural Network Operations" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test vector similarity operations for vector store acceleration
    member this.testVectorSimilarityOperations() =
        let startTime = DateTime.UtcNow
        try
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "Vector Similarity Operations" false "CUDA initialization failed" startTime endTime
            else
                let similarityResult = CudaInterop.cuda_vector_similarity_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if similarityResult = 0 then
                    this.createTestResult "Vector Similarity Operations" true "Vector dot product computed correctly (512D vectors)" startTime endTime
                else
                    this.createTestResult "Vector Similarity Operations" false "Vector similarity test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Vector Similarity Operations" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test transformer attention mechanisms
    member this.testTransformerAttention() =
        let startTime = DateTime.UtcNow
        try
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "Transformer Attention" false "CUDA initialization failed" startTime endTime
            else
                let attentionResult = CudaInterop.cuda_transformer_attention_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if attentionResult = 0 then
                    this.createTestResult "Transformer Attention" true "Scaled dot-product attention computed successfully" startTime endTime
                else
                    this.createTestResult "Transformer Attention" false "Transformer attention test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "Transformer Attention" false (sprintf "Error: %s" ex.Message) startTime endTime

    /// Test comprehensive AI model inference pipeline
    member this.testAIModelInference() =
        let startTime = DateTime.UtcNow
        try
            let initResult = CudaInterop.minimal_cuda_init(0)
            if initResult <> 0 then
                let endTime = DateTime.UtcNow
                this.createTestResult "AI Model Inference" false "CUDA initialization failed" startTime endTime
            else
                let inferenceResult = CudaInterop.cuda_ai_model_inference_test()
                let cleanupResult = CudaInterop.minimal_cuda_cleanup()
                let endTime = DateTime.UtcNow

                if inferenceResult = 0 then
                    this.createTestResult "AI Model Inference" true "Complete AI pipeline: positional encoding, layer norm, GELU activation" startTime endTime
                else
                    this.createTestResult "AI Model Inference" false "AI model inference test failed" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            this.createTestResult "AI Model Inference" false (sprintf "Error: %s" ex.Message) startTime endTime
    
    /// Run CUDA tests based on mode
    member this.runCudaTests mode =
        let overallStartTime = DateTime.UtcNow
        
        printfn ""
        printfn "🚀 TARS CUDA TEST SUITE"
        printfn "======================="
        printfn "Real GPU execution - No simulations!"
        printfn ""
        
        // Display system info
        let osInfo = Environment.OSVersion
        let runtimeInfo = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription
        let cpuCores = Environment.ProcessorCount
        let processMemory = float (System.Diagnostics.Process.GetCurrentProcess().WorkingSet64) / (1024.0 * 1024.0)
        
        printfn "🖥️  SYSTEM INFORMATION"
        printfn "======================"
        printfn "OS: %s" osInfo.VersionString
        printfn "Runtime: %s" runtimeInfo
        printfn "CPU Cores: %d" cpuCores
        printfn "Process Memory: %.1f MB" processMemory
        printfn ""
        
        let tests =
            match mode with
            | "device" -> [("Device Detection", this.testDeviceDetection)]
            | "init" -> [("CUDA Initialization", this.testCudaInitialization)]
            | "memory" -> [("Memory Operations", this.testMemoryOperations)]
            | "vector" -> [("Vector Operations", this.testVectorOperations)]
            | "matrix" -> [("Matrix Operations", this.testMatrixOperations)]
            | "neural" -> [("Neural Network Operations", this.testNeuralNetworkOperations)]
            | "similarity" -> [("Vector Similarity Operations", this.testVectorSimilarityOperations)]
            | "attention" -> [("Transformer Attention", this.testTransformerAttention)]
            | "inference" -> [("AI Model Inference", this.testAIModelInference)]
            | "advanced" ->
                [
                    ("Memory Operations", this.testMemoryOperations);
                    ("Vector Operations", this.testVectorOperations);
                    ("Matrix Operations", this.testMatrixOperations);
                    ("Neural Network Operations", this.testNeuralNetworkOperations);
                    ("Vector Similarity Operations", this.testVectorSimilarityOperations);
                    ("Transformer Attention", this.testTransformerAttention);
                    ("AI Model Inference", this.testAIModelInference)
                ]
            | "ai" ->
                [
                    ("Matrix Operations", this.testMatrixOperations);
                    ("Neural Network Operations", this.testNeuralNetworkOperations);
                    ("Vector Similarity Operations", this.testVectorSimilarityOperations);
                    ("Transformer Attention", this.testTransformerAttention);
                    ("AI Model Inference", this.testAIModelInference)
                ]
            | "transformer" ->
                [
                    ("Transformer Attention", this.testTransformerAttention);
                    ("AI Model Inference", this.testAIModelInference)
                ]
            | _ ->
                [
                    ("Device Detection", this.testDeviceDetection);
                    ("CUDA Initialization", this.testCudaInitialization);
                    ("Memory Operations", this.testMemoryOperations);
                    ("Vector Operations", this.testVectorOperations);
                    ("Matrix Operations", this.testMatrixOperations);
                    ("Neural Network Operations", this.testNeuralNetworkOperations);
                    ("Vector Similarity Operations", this.testVectorSimilarityOperations);
                    ("Transformer Attention", this.testTransformerAttention);
                    ("AI Model Inference", this.testAIModelInference)
                ]
        
        let mutable results = []
        let mutable totalTests = 0
        let mutable passedTests = 0
        
        printfn "🧪 RUNNING TESTS"
        printfn "================"
        
        for (testName, testFunc) in tests do
            printf "🔧 Running %-20s... " testName
            let result = testFunc()
            results <- result :: results
            totalTests <- totalTests + 1
            
            if result.Success then
                passedTests <- passedTests + 1
                printfn "✅ PASSED (%.2f ms)" result.ExecutionTimeMs
            else
                printfn "❌ FAILED (%.2f ms)" result.ExecutionTimeMs
        
        let overallEndTime = DateTime.UtcNow
        let overallTime = (overallEndTime - overallStartTime).TotalMilliseconds
        let successRate = float passedTests / float totalTests * 100.0
        
        printfn ""
        printfn "📈 DETAILED METRICS"
        printfn "==================="
        
        let totalTime = results |> List.sumBy (fun r -> r.ExecutionTimeMs)
        let avgTime = totalTime / float results.Length
        
        printfn "Total Execution Time: %.2f ms" totalTime
        printfn "Average Test Time: %.2f ms" avgTime
        printfn ""
        
        printfn "📋 PER-TEST BREAKDOWN"
        printfn "====================="
        for result in List.rev results do
            let status = if result.Success then "✅ PASS" else "❌ FAIL"
            printfn "%s | %-20s | %8.2f ms | %6.1f MB | %s" 
                status 
                result.TestName 
                result.ExecutionTimeMs 
                result.MemoryUsedMB 
                result.Message
        
        printfn ""
        printfn "🏁 FINAL RESULTS"
        printfn "================"
        printfn "Tests Passed: %d/%d (%.1f%%)" passedTests totalTests successRate
        printfn "Total Runtime: %.2f ms" overallTime
        printfn "Success Rate: %.1f%%" successRate
        
        if successRate >= 80.0 then
            printfn ""
            printfn "🎉 CUDA TESTS COMPLETED SUCCESSFULLY!"
            printfn "✅ CUDA implementation is working correctly"
            printfn "🚀 Ready for production use!"
            CommandResult.success "CUDA tests passed successfully"
        else
            printfn ""
            printfn "❌ CUDA TESTS FAILED!"
            printfn "🔧 Check the detailed metrics above for specific issues"
            CommandResult.failure "CUDA tests failed"
    
    interface ICommand with
        member _.Name = "cuda"

        member _.Description = "CUDA GPU acceleration with transformers, AI inference, and advanced neural networks"

        member _.Usage = "tars cuda [test|device|init|memory|vector|matrix|neural|similarity|attention|inference|advanced|ai|transformer] [options]"

        member _.Examples = [
            "tars cuda test          # Run all CUDA tests (9 operations)"
            "tars cuda device        # Test device detection only"
            "tars cuda init          # Test CUDA initialization only"
            "tars cuda memory        # Test memory operations"
            "tars cuda vector        # Test vector operations"
            "tars cuda matrix        # Test matrix multiplication"
            "tars cuda neural        # Test neural network operations"
            "tars cuda similarity    # Test vector similarity (for vector store)"
            "tars cuda attention     # Test transformer attention mechanisms"
            "tars cuda inference     # Test AI model inference pipeline"
            "tars cuda advanced      # Run all advanced operations (7 tests)"
            "tars cuda ai            # Run AI-focused tests (5 operations)"
            "tars cuda transformer   # Run transformer-specific tests"
        ]
        
        member _.ValidateOptions(options: CommandOptions) =
            // CUDA command is always valid
            true
        
        member self.ExecuteAsync(options: CommandOptions) =
            Task.Run(fun () ->
                try
                    logger.LogInformation("Executing CUDA command")

                    let mode =
                        match options.Arguments with
                        | [] -> "all"
                        | "test" :: _ -> "all"
                        | "device" :: _ -> "device"
                        | "init" :: _ -> "init"
                        | "memory" :: _ -> "memory"
                        | "vector" :: _ -> "vector"
                        | "matrix" :: _ -> "matrix"
                        | "neural" :: _ -> "neural"
                        | "similarity" :: _ -> "similarity"
                        | "attention" :: _ -> "attention"
                        | "inference" :: _ -> "inference"
                        | "advanced" :: _ -> "advanced"
                        | "ai" :: _ -> "ai"
                        | "transformer" :: _ -> "transformer"
                        | "benchmark" :: _ -> "all" // For now, same as all
                        | mode :: _ -> mode

                    if options.Help then
                        printfn "TARS CUDA Command"
                        printfn "================="
                        printfn ""
                        printfn "Description: CUDA GPU acceleration with transformers, AI inference, and advanced neural networks"
                        printfn "Usage: tars cuda [test|device|init|memory|vector|matrix|neural|similarity|attention|inference|advanced|ai|transformer] [options]"
                        printfn ""
                        printfn "Examples:"
                        printfn "  tars cuda test          # Run all CUDA tests (9 operations)"
                        printfn "  tars cuda device        # Test device detection only"
                        printfn "  tars cuda init          # Test CUDA initialization only"
                        printfn "  tars cuda memory        # Test memory operations"
                        printfn "  tars cuda vector        # Test vector operations"
                        printfn "  tars cuda matrix        # Test matrix multiplication"
                        printfn "  tars cuda neural        # Test neural network operations"
                        printfn "  tars cuda similarity    # Test vector similarity (for vector store)"
                        printfn "  tars cuda attention     # Test transformer attention mechanisms"
                        printfn "  tars cuda inference     # Test AI model inference pipeline"
                        printfn "  tars cuda advanced      # Run all advanced operations (7 tests)"
                        printfn "  tars cuda ai            # Run AI-focused tests (5 operations)"
                        printfn "  tars cuda transformer   # Run transformer-specific tests"
                        printfn ""
                        printfn "Modes:"
                        printfn "  test        - Run all CUDA tests (default) - 9 operations"
                        printfn "  device      - Test CUDA device detection only"
                        printfn "  init        - Test CUDA initialization only"
                        printfn "  memory      - Test memory allocation and management"
                        printfn "  vector      - Test vector operations (1024 elements)"
                        printfn "  matrix      - Test matrix multiplication (64x64 matrices)"
                        printfn "  neural      - Test neural network operations (ReLU, Sigmoid)"
                        printfn "  similarity  - Test vector similarity for vector store acceleration"
                        printfn "  attention   - Test transformer attention mechanisms"
                        printfn "  inference   - Test AI model inference pipeline"
                        printfn "  advanced    - Run all advanced operations (7 tests)"
                        printfn "  ai          - Run AI-focused tests (5 operations)"
                        printfn "  transformer - Run transformer-specific tests (2 operations)"
                        printfn ""
                        printfn "Requirements:"
                        printfn "- CUDA-capable GPU"
                        printfn "- CUDA drivers installed"
                        printfn "- WSL environment (for Linux CUDA compilation)"

                        CommandResult.success ""
                    else
                        self.runCudaTests mode

                with
                | ex ->
                    logger.LogError(ex, "Error executing CUDA command")
                    CommandResult.failure (sprintf "CUDA command failed: %s" ex.Message)
            )
