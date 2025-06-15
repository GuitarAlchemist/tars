// cuda_kernels_hybrid_space.cu
// Phase 2: Custom CUDA Kernels for Hybrid Vector Geometry in TARS

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdio.h>

extern "C" {

// =============== Möbius Addition (Hyperbolic Space) =============== //

__device__ float mobius_denominator(float c, float xy, float x2, float y2) {
    return 1.0f + 2.0f * c * xy + c * c * x2 * y2;
}

__global__ void mobius_add_kernel(
    const float* x, const float* y,
    float* result,
    float c,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * dim;
    
    if (idx < total_elements) {
        int batch_idx = idx / dim;
        int dim_idx = idx % dim;
        
        // Calculate norms and dot product for this batch
        float x2 = 0.0f, y2 = 0.0f, xy = 0.0f;
        for (int i = 0; i < dim; i++) {
            float xi = x[batch_idx * dim + i];
            float yi = y[batch_idx * dim + i];
            x2 += xi * xi;
            y2 += yi * yi;
            xy += xi * yi;
        }
        
        float xi = x[idx];
        float yi = y[idx];
        
        float denom = mobius_denominator(c, xy, x2, y2);
        float num = (1.0f + 2.0f * c * xy + c * y2) * xi + (1.0f - c * x2) * yi;
        
        result[idx] = num / (denom + 1e-8f);
    }
}

// =============== Hyperbolic Distance (Poincaré Model) =============== //

__global__ void hyperbolic_distance_kernel(
    const float* u, const float* v,
    float* distances,
    float c,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float norm_u_sq = 0.0f, norm_v_sq = 0.0f, dot_uv = 0.0f;
        
        for (int i = 0; i < dim; i++) {
            float ui = u[batch_idx * dim + i];
            float vi = v[batch_idx * dim + i];
            norm_u_sq += ui * ui;
            norm_v_sq += vi * vi;
            dot_uv += ui * vi;
        }
        
        // Ensure points are in unit disk
        norm_u_sq = fminf(norm_u_sq, 0.999f);
        norm_v_sq = fminf(norm_v_sq, 0.999f);
        
        float diff_norm_sq = norm_u_sq + norm_v_sq - 2.0f * dot_uv;
        float denominator = (1.0f - norm_u_sq) * (1.0f - norm_v_sq);
        
        if (denominator > 1e-8f) {
            float ratio = 1.0f + 2.0f * diff_norm_sq / denominator;
            distances[batch_idx] = acoshf(fmaxf(ratio, 1.0f)) / sqrtf(c);
        } else {
            distances[batch_idx] = 0.0f;
        }
    }
}

// =============== Dual Quaternion Operations =============== //

__global__ void dual_quaternion_norm_kernel(
    const float* real, const float* dual,
    float* norms,
    int batch_size,
    int quat_dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float real_norm_sq = 0.0f;
        float dual_norm_sq = 0.0f;
        
        for (int i = 0; i < quat_dim; i++) {
            float r = real[batch_idx * quat_dim + i];
            float d = dual[batch_idx * quat_dim + i];
            real_norm_sq += r * r;
            dual_norm_sq += d * d;
        }
        
        norms[batch_idx] = sqrtf(real_norm_sq + dual_norm_sq);
    }
}

__global__ void dual_quaternion_multiply_kernel(
    const float* q1_real, const float* q1_dual,
    const float* q2_real, const float* q2_dual,
    float* result_real, float* result_dual,
    int batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Quaternion multiplication: q1 * q2
        // q = w + xi + yj + zk
        float q1w = q1_real[batch_idx * 4 + 0];
        float q1x = q1_real[batch_idx * 4 + 1];
        float q1y = q1_real[batch_idx * 4 + 2];
        float q1z = q1_real[batch_idx * 4 + 3];
        
        float q2w = q2_real[batch_idx * 4 + 0];
        float q2x = q2_real[batch_idx * 4 + 1];
        float q2y = q2_real[batch_idx * 4 + 2];
        float q2z = q2_real[batch_idx * 4 + 3];
        
        // Real part multiplication
        result_real[batch_idx * 4 + 0] = q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2z;
        result_real[batch_idx * 4 + 1] = q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y;
        result_real[batch_idx * 4 + 2] = q1w*q2y - q1x*q2z + q1y*q2w + q1z*q2x;
        result_real[batch_idx * 4 + 3] = q1w*q2z + q1x*q2y - q1y*q2x + q1z*q2w;
        
        // Dual part (simplified for now)
        for (int i = 0; i < 4; i++) {
            result_dual[batch_idx * 4 + i] = q1_dual[batch_idx * 4 + i] + q2_dual[batch_idx * 4 + i];
        }
    }
}

// =============== Projective Space Operations =============== //

__global__ void projective_normalize_kernel(
    float* vectors,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float norm = 0.0f;
        
        // Calculate norm
        for (int i = 0; i < dim; i++) {
            float val = vectors[batch_idx * dim + i];
            norm += val * val;
        }
        norm = sqrtf(norm + 1e-8f);
        
        // Normalize
        for (int i = 0; i < dim; i++) {
            vectors[batch_idx * dim + i] /= norm;
        }
    }
}

__global__ void projective_distance_kernel(
    const float* u, const float* v,
    float* distances,
    int batch_size,
    int dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float dot_product = 0.0f;
        
        for (int i = 0; i < dim; i++) {
            dot_product += u[batch_idx * dim + i] * v[batch_idx * dim + i];
        }
        
        // Projective distance is related to angle between normalized vectors
        dot_product = fmaxf(-1.0f, fminf(1.0f, dot_product));
        distances[batch_idx] = acosf(fabsf(dot_product));
    }
}

// =============== Host Interface Functions =============== //

void call_mobius_add(
    const float* h_x, const float* h_y, float* h_result,
    float curvature, int batch_size, int dim
) {
    size_t size = batch_size * dim * sizeof(float);
    
    float *d_x, *d_y, *d_result;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_result, size);
    
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    
    int total_elements = batch_size * dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    mobius_add_kernel<<<grid_size, block_size>>>(
        d_x, d_y, d_result, curvature, batch_size, dim
    );
    
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}

void call_hyperbolic_distance(
    const float* h_u, const float* h_v, float* h_distances,
    float curvature, int batch_size, int dim
) {
    size_t vector_size = batch_size * dim * sizeof(float);
    size_t distance_size = batch_size * sizeof(float);
    
    float *d_u, *d_v, *d_distances;
    cudaMalloc(&d_u, vector_size);
    cudaMalloc(&d_v, vector_size);
    cudaMalloc(&d_distances, distance_size);
    
    cudaMemcpy(d_u, h_u, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, vector_size, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    hyperbolic_distance_kernel<<<grid_size, block_size>>>(
        d_u, d_v, d_distances, curvature, batch_size, dim
    );
    
    cudaMemcpy(h_distances, d_distances, distance_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_distances);
}

void call_projective_normalize(
    float* h_vectors, int batch_size, int dim
) {
    size_t size = batch_size * dim * sizeof(float);
    
    float *d_vectors;
    cudaMalloc(&d_vectors, size);
    cudaMemcpy(d_vectors, h_vectors, size, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    projective_normalize_kernel<<<grid_size, block_size>>>(
        d_vectors, batch_size, dim
    );
    
    cudaMemcpy(h_vectors, d_vectors, size, cudaMemcpyDeviceToHost);
    cudaFree(d_vectors);
}

void call_dual_quaternion_norm(
    const float* h_real, const float* h_dual, float* h_norms,
    int batch_size, int quat_dim
) {
    size_t vector_size = batch_size * quat_dim * sizeof(float);
    size_t norm_size = batch_size * sizeof(float);
    
    float *d_real, *d_dual, *d_norms;
    cudaMalloc(&d_real, vector_size);
    cudaMalloc(&d_dual, vector_size);
    cudaMalloc(&d_norms, norm_size);
    
    cudaMemcpy(d_real, h_real, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dual, h_dual, vector_size, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    dual_quaternion_norm_kernel<<<grid_size, block_size>>>(
        d_real, d_dual, d_norms, batch_size, quat_dim
    );
    
    cudaMemcpy(h_norms, d_norms, norm_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_real);
    cudaFree(d_dual);
    cudaFree(d_norms);
}

} // extern "C"
