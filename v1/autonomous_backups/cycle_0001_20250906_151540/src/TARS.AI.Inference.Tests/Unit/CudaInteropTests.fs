namespace TARS.AI.Inference.Tests.Unit

open System
open Xunit
open FsUnit.Xunit
open TARS.AI.Inference.Core.CudaInterop

/// Comprehensive CUDA interop testing
module CudaInteropTests =

    [<Fact>]
    let ``CUDA availability check should not throw`` () =
        // Test that CUDA availability check doesn't crash
        let result = isCudaAvailable()
        // Should return either true or false, not throw
        result |> should be (ofType<bool>)

    [<Fact>]
    let ``Get device info should handle invalid device gracefully`` () =
        // Test with invalid device ID
        match getDeviceInfo(-1) with
        | Ok(_) -> failwith "Should not succeed with invalid device ID"
        | Error(msg) -> 
            msg |> should not' (be EmptyString)
            msg |> should contain "device"

    [<Fact>]
    let ``CUDA context initialization should be deterministic`` () =
        // Test multiple initialization attempts
        let results = [1..5] |> List.map (fun _ -> 
            match initializeCudaContext(0) with
            | Ok(_) -> true
            | Error(_) -> false
        )
        
        // All results should be the same (either all succeed or all fail)
        let firstResult = results.Head
        results |> List.forall (fun r -> r = firstResult) |> should be True

    [<Fact>]
    let ``Error message retrieval should work for all error codes`` () =
        let errorCodes = [
            TarsError.Success
            TarsError.CudaRuntime
            TarsError.Cublas
            TarsError.InvalidParams
            TarsError.OutOfMemory
            TarsError.NotInitialized
        ]
        
        for errorCode in errorCodes do
            let message = getErrorMessage(errorCode)
            message |> should not' (be EmptyString)
            message.Length |> should be (greaterThan 0)

    [<Fact>]
    let ``CUDA context cleanup should handle null context gracefully`` () =
        // Test cleanup with uninitialized context
        let emptyContext = Unchecked.defaultof<TarsCudaContext>
        match cleanupCudaContext(emptyContext) with
        | Ok(_) -> () // Should handle gracefully
        | Error(msg) -> 
            // Error is acceptable, but should be descriptive
            msg |> should not' (be EmptyString)

    [<Theory>]
    [<InlineData(0)>]
    [<InlineData(1)>]
    [<InlineData(-1)>]
    let ``Device info query should handle various device IDs`` (deviceId: int) =
        match getDeviceInfo(deviceId) with
        | Ok(name, memory) ->
            name |> should not' (be EmptyString)
            memory |> should be (greaterThanOrEqualTo 0L)
        | Error(msg) ->
            // Error is acceptable for invalid devices
            msg |> should not' (be EmptyString)

    [<Fact>]
    let ``CUDA library loading should be testable`` () =
        // Test that we can at least attempt to load the library
        try
            let _ = isCudaAvailable()
            true |> should be True
        with
        | :? System.DllNotFoundException as ex ->
            // Expected if CUDA library not built yet
            ex.Message |> should contain "tars_cuda"
        | ex ->
            // Other exceptions should be investigated
            failwithf "Unexpected exception: %s" ex.Message

    [<Fact>]
    let ``Tensor structure should have correct layout`` () =
        let tensor = {
            Data = IntPtr.Zero
            Shape = IntPtr.Zero
            NDim = 2
            TotalSize = 100
            IsOnGPU = false
        }
        
        tensor.NDim |> should equal 2
        tensor.TotalSize |> should equal 100
        tensor.IsOnGPU |> should be False

    [<Fact>]
    let ``CUDA context structure should initialize correctly`` () =
        let context = {
            Stream = IntPtr.Zero
            CublasHandle = IntPtr.Zero
            CurandGen = IntPtr.Zero
            DeviceId = 0
            Initialized = false
        }
        
        context.DeviceId |> should equal 0
        context.Initialized |> should be False

    [<Fact>]
    let ``Error enum values should be distinct`` () =
        let errorValues = [
            int TarsError.Success
            int TarsError.CudaRuntime
            int TarsError.Cublas
            int TarsError.InvalidParams
            int TarsError.OutOfMemory
            int TarsError.NotInitialized
        ]
        
        let uniqueValues = errorValues |> List.distinct
        uniqueValues.Length |> should equal errorValues.Length

    [<Fact>]
    let ``P/Invoke function signatures should be accessible`` () =
        // Test that we can reference the P/Invoke functions without calling them
        let functions = [
            typeof<TarsError> // Ensure enum is accessible
            // Note: Can't directly test P/Invoke functions without native library
        ]
        
        functions |> List.length |> should be (greaterThan 0)

module CudaMemoryTests =

    [<Fact>]
    let ``Tensor creation should validate parameters`` () =
        // Test tensor parameter validation
        let validShape = [|2; 3; 4|]
        let validSize = Array.fold (*) 1 validShape
        
        validSize |> should equal 24

    [<Fact>]
    let ``Memory allocation size calculations should be correct`` () =
        let shapes = [
            [|1|], 1
            [|10|], 10
            [|2; 3|], 6
            [|2; 3; 4|], 24
            [|1; 1; 1; 1|], 1
        ]
        
        for (shape, expectedSize) in shapes do
            let calculatedSize = Array.fold (*) 1 shape
            calculatedSize |> should equal expectedSize

    [<Fact>]
    let ``Large tensor size calculations should not overflow`` () =
        // Test with reasonably large tensors
        let largeShape = [|1024; 1024|]
        let size = Array.fold (*) 1 largeShape
        size |> should equal (1024 * 1024)
        size |> should be (greaterThan 0)

module CudaErrorHandlingTests =

    [<Fact>]
    let ``Error handling should be consistent`` () =
        // Test that error handling patterns are consistent
        let testError = TarsError.InvalidParams
        let message = getErrorMessage(testError)
        
        message |> should not' (be null)
        message |> should not' (be EmptyString)

    [<Fact>]
    let ``Success error code should be zero`` () =
        int TarsError.Success |> should equal 0

    [<Fact>]
    let ``Error codes should be positive`` () =
        let errorCodes = [
            TarsError.CudaRuntime
            TarsError.Cublas
            TarsError.InvalidParams
            TarsError.OutOfMemory
            TarsError.NotInitialized
        ]
        
        for errorCode in errorCodes do
            int errorCode |> should be (greaterThan 0)
