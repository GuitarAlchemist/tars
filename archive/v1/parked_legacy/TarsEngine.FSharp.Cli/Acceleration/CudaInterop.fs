namespace TarsEngine.FSharp.Cli.Acceleration

open System.Runtime.InteropServices
open TarsEngine.FSharp.Cli.Acceleration.CudaTypes

/// CUDA Interop - Native CUDA function bindings
module CudaInterop =
    
    /// Basic CUDA device management
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int tars_cuda_device_count()
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_cuda_init(int deviceId)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_cuda_cleanup()
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_cuda_get_device_info(
        int deviceId, 
        nativeint name, 
        int nameLen, 
        int64& totalMemory, 
        float& computeCapability)
    
    /// Memory management
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_cuda_malloc(nativeint& ptr, int64 size)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_cuda_free(nativeint ptr)
    
    /// Stream management
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_create_stream(int64& stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_destroy_stream(int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_synchronize_device()
    
    /// Basic CUDA operations
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_vector_similarity(
        nativeint vectors1, 
        nativeint vectors2, 
        nativeint results, 
        int numVectors, 
        int dimensions, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_matrix_multiply(
        nativeint A, 
        nativeint B, 
        nativeint C, 
        int M, 
        int N, 
        int K, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_reasoning_kernel(
        nativeint input, 
        nativeint output, 
        int complexity, 
        int64 stream)
    
    /// AI-specific CUDA kernels
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_gemm_fp16(
        nativeint A, 
        nativeint B, 
        nativeint C, 
        int M, 
        int N, 
        int K, 
        float alpha, 
        float beta, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_gemm_tensor_core(
        nativeint A, 
        nativeint B, 
        nativeint C, 
        int M, 
        int N, 
        int K, 
        float alpha, 
        float beta, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_gelu_forward(
        nativeint input, 
        nativeint output, 
        int size, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_relu_forward(
        nativeint input, 
        nativeint output, 
        int size, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_softmax_forward(
        nativeint input, 
        nativeint output, 
        int size, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_layer_norm(
        nativeint input, 
        nativeint gamma, 
        nativeint beta, 
        nativeint output, 
        int size, 
        float eps, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_scaled_dot_product_attention(
        nativeint Q, 
        nativeint K, 
        nativeint V, 
        nativeint output, 
        int seq_len, 
        int head_dim, 
        int64 stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern CudaError tars_embedding_lookup(
        nativeint embeddings, 
        nativeint indices, 
        nativeint output, 
        int vocab_size, 
        int embed_dim, 
        int seq_len, 
        int64 stream)
