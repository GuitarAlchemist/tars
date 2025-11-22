namespace TarsEngine

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.CudaInterop

/// CUDA Memory Management Tests - Real GPU memory operations
module CudaMemoryTests =
    
    /// Memory test result
    type MemoryTestResult = {
        TestName: string
        Success: bool
        ExecutionTimeMs: float
        MemoryAllocatedMB: float
        ThroughputMBPerSec: float option
        ErrorMessage: string option
        PerformanceMetrics: Map<string, float>
    }
    
    /// CUDA Memory Test Suite
    type CudaMemoryTestSuite(logger: ILogger<CudaMemoryTestSuite>) =
        
        /// Test basic memory allocation and deallocation
        member _.TestBasicMemoryAllocation() = async {
            logger.LogInformation("🧪 Testing CUDA basic memory allocation/deallocation - REAL GPU MEMORY...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Basic Memory Allocation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        MemoryAllocatedMB = 0.0
                        ThroughputMBPerSec = None
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Basic Memory Allocation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        MemoryAllocatedMB = 0.0
                        ThroughputMBPerSec = None
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Test various memory sizes
                let testSizes = [
                    ("1MB", 1024UL * 1024UL)
                    ("10MB", 10UL * 1024UL * 1024UL)
                    ("100MB", 100UL * 1024UL * 1024UL)
                    ("1GB", 1024UL * 1024UL * 1024UL)
                ]
                
                let mutable totalAllocated = 0.0
                let mutable allocationTimes = []
                let startTime = DateTime.UtcNow
                
                for (sizeName, sizeBytes) in testSizes do
                    logger.LogInformation($"🔧 Allocating {sizeName} on GPU...")
                    
                    let allocStart = DateTime.UtcNow
                    let mutable ptr = nativeint 0
                    let allocResult = tars_cuda_malloc(&ptr, unativeint sizeBytes)
                    let allocEnd = DateTime.UtcNow
                    let allocTime = (allocEnd - allocStart).TotalMilliseconds
                    
                    if allocResult = TarsCudaError.Success then
                        logger.LogInformation($"✅ {sizeName} allocated successfully in {allocTime:F2}ms")
                        totalAllocated <- totalAllocated + float sizeBytes / 1e6
                        allocationTimes <- allocTime :: allocationTimes
                        
                        // Free the memory
                        let freeStart = DateTime.UtcNow
                        let freeResult = tars_cuda_free(ptr)
                        let freeEnd = DateTime.UtcNow
                        let freeTime = (freeEnd - freeStart).TotalMilliseconds
                        
                        if freeResult = TarsCudaError.Success then
                            logger.LogInformation($"✅ {sizeName} freed successfully in {freeTime:F2}ms")
                        else
                            logger.LogError($"❌ Failed to free {sizeName}: {freeResult}")
                            return {
                                TestName = "Basic Memory Allocation"
                                Success = false
                                ExecutionTimeMs = 0.0
                                MemoryAllocatedMB = totalAllocated
                                ThroughputMBPerSec = None
                                ErrorMessage = Some $"Memory deallocation failed for {sizeName}"
                                PerformanceMetrics = Map.empty
                            }
                    else
                        logger.LogError($"❌ Failed to allocate {sizeName}: {allocResult}")
                        return {
                            TestName = "Basic Memory Allocation"
                            Success = false
                            ExecutionTimeMs = 0.0
                            MemoryAllocatedMB = totalAllocated
                            ThroughputMBPerSec = None
                            ErrorMessage = Some $"Memory allocation failed for {sizeName}"
                            PerformanceMetrics = Map.empty
                        }
                
                let endTime = DateTime.UtcNow
                let totalTime = (endTime - startTime).TotalMilliseconds
                let avgAllocationTime = allocationTimes |> List.average
                let throughput = totalAllocated / (totalTime / 1000.0)
                
                // Cleanup CUDA
                let cleanupResult = tars_cuda_cleanup()
                if cleanupResult <> TarsCudaError.Success then
                    logger.LogWarning($"⚠️ CUDA cleanup warning: {cleanupResult}")
                
                logger.LogInformation($"✅ All memory tests completed: {totalTime:F2}ms")
                logger.LogInformation($"📊 Total memory tested: {totalAllocated:F1}MB")
                logger.LogInformation($"🚀 Average allocation time: {avgAllocationTime:F2}ms")
                
                return {
                    TestName = "Basic Memory Allocation"
                    Success = true
                    ExecutionTimeMs = totalTime
                    MemoryAllocatedMB = totalAllocated
                    ThroughputMBPerSec = Some throughput
                    ErrorMessage = None
                    PerformanceMetrics = Map [
                        ("total_memory_mb", totalAllocated)
                        ("total_time_ms", totalTime)
                        ("avg_allocation_time_ms", avgAllocationTime)
                        ("throughput_mb_per_sec", throughput)
                        ("num_allocations", float testSizes.Length)
                    ]
                }
            with
            | ex ->
                logger.LogError($"❌ Memory allocation test failed: {ex.Message}")
                return {
                    TestName = "Basic Memory Allocation"
                    Success = false
                    ExecutionTimeMs = 0.0
                    MemoryAllocatedMB = 0.0
                    ThroughputMBPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Test memory transfer operations (Host to Device and Device to Host)
        member _.TestMemoryTransfer() = async {
            logger.LogInformation("🧪 Testing CUDA memory transfer operations - REAL GPU TRANSFERS...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Memory Transfer"
                        Success = false
                        ExecutionTimeMs = 0.0
                        MemoryAllocatedMB = 0.0
                        ThroughputMBPerSec = None
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Memory Transfer"
                        Success = false
                        ExecutionTimeMs = 0.0
                        MemoryAllocatedMB = 0.0
                        ThroughputMBPerSec = None
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Test with 100MB of data
                let dataSize = 100 * 1024 * 1024 // 100MB
                let dataSizeMB = float dataSize / 1e6
                logger.LogInformation($"🔧 Testing transfers with {dataSizeMB:F1}MB of data")
                
                // Allocate host memory
                let hostData = Array.zeroCreate<byte> dataSize
                // Fill with test pattern
                for i in 0 .. dataSize - 1 do
                    hostData.[i] <- byte (i % 256)
                
                // Allocate GPU memory
                let mutable devicePtr = nativeint 0
                let allocResult = tars_cuda_malloc(&devicePtr, unativeint dataSize)
                
                if allocResult <> TarsCudaError.Success then
                    return {
                        TestName = "Memory Transfer"
                        Success = false
                        ExecutionTimeMs = 0.0
                        MemoryAllocatedMB = 0.0
                        ThroughputMBPerSec = None
                        ErrorMessage = Some $"GPU memory allocation failed: {allocResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                let startTime = DateTime.UtcNow
                
                // Pin host memory for the transfer
                use pinnedHandle = System.Runtime.InteropServices.GCHandle.Alloc(hostData, System.Runtime.InteropServices.GCHandleType.Pinned)
                let hostPtr = pinnedHandle.AddrOfPinnedObject()
                
                // Test Host to Device transfer
                logger.LogInformation("📤 Testing Host to Device transfer...")
                let h2dStart = DateTime.UtcNow
                let h2dResult = tars_cuda_memcpy_h2d(devicePtr, hostPtr, unativeint dataSize)
                let h2dEnd = DateTime.UtcNow
                let h2dTime = (h2dEnd - h2dStart).TotalMilliseconds
                
                if h2dResult <> TarsCudaError.Success then
                    tars_cuda_free(devicePtr) |> ignore
                    return {
                        TestName = "Memory Transfer"
                        Success = false
                        ExecutionTimeMs = h2dTime
                        MemoryAllocatedMB = dataSizeMB
                        ThroughputMBPerSec = None
                        ErrorMessage = Some $"Host to Device transfer failed: {h2dResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Test Device to Host transfer
                logger.LogInformation("📥 Testing Device to Host transfer...")
                let resultData = Array.zeroCreate<byte> dataSize
                use resultHandle = System.Runtime.InteropServices.GCHandle.Alloc(resultData, System.Runtime.InteropServices.GCHandleType.Pinned)
                let resultPtr = resultHandle.AddrOfPinnedObject()
                
                let d2hStart = DateTime.UtcNow
                let d2hResult = tars_cuda_memcpy_d2h(resultPtr, devicePtr, unativeint dataSize)
                let d2hEnd = DateTime.UtcNow
                let d2hTime = (d2hEnd - d2hStart).TotalMilliseconds
                
                // Cleanup GPU memory
                let freeResult = tars_cuda_free(devicePtr)
                let cleanupResult = tars_cuda_cleanup()
                
                let endTime = DateTime.UtcNow
                let totalTime = (endTime - startTime).TotalMilliseconds
                
                if d2hResult <> TarsCudaError.Success then
                    return {
                        TestName = "Memory Transfer"
                        Success = false
                        ExecutionTimeMs = totalTime
                        MemoryAllocatedMB = dataSizeMB
                        ThroughputMBPerSec = None
                        ErrorMessage = Some $"Device to Host transfer failed: {d2hResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Verify data integrity
                let dataMatches = hostData |> Array.zip resultData |> Array.forall (fun (a, b) -> a = b)
                
                if not dataMatches then
                    return {
                        TestName = "Memory Transfer"
                        Success = false
                        ExecutionTimeMs = totalTime
                        MemoryAllocatedMB = dataSizeMB
                        ThroughputMBPerSec = None
                        ErrorMessage = Some "Data integrity check failed - transferred data doesn't match"
                        PerformanceMetrics = Map.empty
                    }
                
                let h2dBandwidth = dataSizeMB / (h2dTime / 1000.0)
                let d2hBandwidth = dataSizeMB / (d2hTime / 1000.0)
                let totalBandwidth = (dataSizeMB * 2.0) / (totalTime / 1000.0)
                
                logger.LogInformation($"✅ Memory transfers completed successfully")
                logger.LogInformation($"📊 H2D: {h2dTime:F2}ms ({h2dBandwidth:F1} MB/s)")
                logger.LogInformation($"📊 D2H: {d2hTime:F2}ms ({d2hBandwidth:F1} MB/s)")
                logger.LogInformation($"✅ Data integrity verified")
                
                return {
                    TestName = "Memory Transfer"
                    Success = true
                    ExecutionTimeMs = totalTime
                    MemoryAllocatedMB = dataSizeMB
                    ThroughputMBPerSec = Some totalBandwidth
                    ErrorMessage = None
                    PerformanceMetrics = Map [
                        ("data_size_mb", dataSizeMB)
                        ("h2d_time_ms", h2dTime)
                        ("d2h_time_ms", d2hTime)
                        ("total_time_ms", totalTime)
                        ("h2d_bandwidth_mb_s", h2dBandwidth)
                        ("d2h_bandwidth_mb_s", d2hBandwidth)
                        ("total_bandwidth_mb_s", totalBandwidth)
                        ("data_integrity_verified", 1.0)
                    ]
                }
            with
            | ex ->
                logger.LogError($"❌ Memory transfer test failed: {ex.Message}")
                return {
                    TestName = "Memory Transfer"
                    Success = false
                    ExecutionTimeMs = 0.0
                    MemoryAllocatedMB = 0.0
                    ThroughputMBPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
