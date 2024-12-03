#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define MAX_BLOCK_M 64       // Adjust based on BLOCK_M
#define MAX_BLOCK_N 64       // Adjust based on BLOCK_N
#define MAX_BLOCK_DMODEL 64  // Adjust based on BLOCK_DMODEL

__global__ void mixed_sparse_attn_bwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float sm_scale,
    int context_size,
    const int* __restrict__ block_count,
    const int* __restrict__ block_offset,
    const int* __restrict__ column_count,
    const int* __restrict__ column_index,
    const float* __restrict__ Out,
    const float* __restrict__ DO,
    float* __restrict__ DQ,
    float* __restrict__ DK,
    float* __restrict__ DV,
    const float* __restrict__ L,
    const float* __restrict__ D,
    int stride_qz, int stride_qh, int stride_qm, int stride_qk,
    int stride_kz, int stride_kh, int stride_kn, int stride_kk,
    int stride_vz, int stride_vh, int stride_vn, int stride_vk,
    int stride_dqz, int stride_dqh, int stride_dqm, int stride_dqk,
    int stride_dkz, int stride_dkh, int stride_dkn, int stride_dkk,
    int stride_dvz, int stride_dvh, int stride_dvn, int stride_dvk,
    int Z, int H, int N_CTX, int P_SEQ,
    int num_block_q, int num_block_kv, int NNZ_S, int NNZ_V,
    int BLOCK_M,
    int BLOCK_DMODEL,
    int BLOCK_N,
    bool CAUSAL)
{
    int start_m = blockIdx.x;
    int off_hz = blockIdx.y;
    if (start_m >= num_block_q || off_hz >= H * Z) return;

    int off_z = off_hz / H;
    int off_h = off_hz % H;
    float qk_scale = sm_scale * 1.44269504f;  // ln2 reciprocal

    // Offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh;
    K += off_z * stride_kz + off_h * stride_kh;
    V += off_z * stride_vz + off_h * stride_vh;
    DO += off_z * stride_qz + off_h * stride_qh;

    DQ += off_z * stride_dqz + off_h * stride_dqh;
    DK += off_z * stride_dkz + off_h * stride_dkh;
    DV += off_z * stride_dvz + off_h * stride_dvh;

    // Thread indices within the block
    int thread_m = threadIdx.y;  // [0, BLOCK_M)
    int thread_d = threadIdx.x;  // [0, BLOCK_DMODEL)

    // Initialize offsets
    int offs_m = start_m * BLOCK_M + thread_m;  // Offset in M dimension
    int offs_k = thread_d;                      // Offset in D_MODEL dimension

    // Shared memory for Q, K, V, DO
    __shared__ float q_shared[MAX_BLOCK_M][MAX_BLOCK_DMODEL];
    __shared__ float do_shared[MAX_BLOCK_M][MAX_BLOCK_DMODEL];
    __shared__ float dq_shared[MAX_BLOCK_M][MAX_BLOCK_DMODEL];

    // Load Q and DO into shared memory
    if (thread_m < BLOCK_M && offs_m < context_size && thread_d < BLOCK_DMODEL) {
        int q_index = offs_m * stride_qm + offs_k * stride_qk;
        q_shared[thread_m][thread_d] = Q[q_index];

        int do_index = offs_m * stride_qm + offs_k * stride_qk;
        do_shared[thread_m][thread_d] = DO[do_index];
    } else {
        q_shared[thread_m][thread_d] = 0.0f;
        do_shared[thread_m][thread_d] = 0.0f;
    }

    // Initialize dq_shared
    dq_shared[thread_m][thread_d] = 0.0f;

    // Load L and D
    float l_i = 0.0f;
    float D_i = 0.0f;
    if (thread_d == 0 && offs_m < context_size) {
        int l_index = off_hz * N_CTX + offs_m;
        l_i = L[l_index];
        D_i = D[l_index];
    }
    __syncthreads();

    // Number of non-sparse blocks
    int num_blks = block_count[off_hz * num_block_q + start_m];
    const int* blks_ptr = block_offset + (off_hz * num_block_q + start_m) * NNZ_S;

    // Number of non-sparse columns
    int num_cols = column_count[off_hz * num_block_q + start_m];
    const int* cols_ptr = column_index + (off_hz * num_block_q + start_m) * NNZ_V;

    // Shared memory for K, V
    __shared__ float k_shared[MAX_BLOCK_N][MAX_BLOCK_DMODEL];
    __shared__ float v_shared[MAX_BLOCK_N][MAX_BLOCK_DMODEL];

    // Loop over non-sparse blocks
    for (int block_index = 0; block_index < num_blks; ++block_index) {
        int start_n = blks_ptr[block_index];

        // Compute cols
        int offs_n = threadIdx.z;  // [0, BLOCK_N)
        int cols = start_n + offs_n;

        // Load K and V into shared memory
        if (offs_n < BLOCK_N && cols < context_size && thread_d < BLOCK_DMODEL) {
            int k_index = cols * stride_kn + offs_k * stride_kk;
            int v_index = cols * stride_vn + offs_k * stride_vk;
            k_shared[offs_n][thread_d] = K[k_index];
            v_shared[offs_n][thread_d] = V[v_index];
        } else {
            k_shared[offs_n][thread_d] = 0.0f;
            v_shared[offs_n][thread_d] = 0.0f;
        }
        __syncthreads();

        // Compute qk and p
        float qk = 0.0f;
        if (offs_m < context_size && offs_n < BLOCK_N) {
            for (int d = 0; d < BLOCK_DMODEL; ++d) {
                qk += q_shared[thread_m][d] * k_shared[offs_n][d];
            }
            qk *= qk_scale;

            // Apply causal mask if needed
            if (CAUSAL) {
                if ((P_SEQ + offs_m) < cols) {
                    qk = -INFINITY;
                }
            }

            float p = exp2f(qk - l_i);

            // Compute dv
            float dv_val = p * do_shared[thread_m][thread_d];

            // Atomic add to DV
            int dv_index = cols * stride_dvn + offs_k * stride_dvk;
            atomicAdd(&DV[dv_index], dv_val);

            // Compute dp
            float dp = -D_i + 0.0f;  // Assuming D_i is scalar per row
            for (int d = 0; d < BLOCK_DMODEL; ++d) {
                dp += do_shared[thread_m][d] * v_shared[offs_n][d];
            }

            // Compute ds
            float ds = p * dp * sm_scale;

            // Compute dk
            float dk_val = ds * q_shared[thread_m][thread_d];

            // Atomic add to DK
            int dk_index = cols * stride_dkn + offs_k * stride_dkk;
            atomicAdd(&DK[dk_index], dk_val);

            // Compute dq
            dq_shared[thread_m][thread_d] += ds * k_shared[offs_n][thread_d];
        }
        __syncthreads();
    }

    // Loop over column indices
    int num_block_ns = (num_cols + BLOCK_N - 1) / BLOCK_N;
    for (int block_n_idx = 0; block_n_idx < num_block_ns; ++block_n_idx) {
        int offs_n = threadIdx.z;  // [0, BLOCK_N)
        int col_idx = block_n_idx * BLOCK_N + offs_n;

        // Load column indices
        __shared__ int cols_shared[MAX_BLOCK_N];
        if (offs_n < BLOCK_N && col_idx < num_cols) {
            cols_shared[offs_n] = cols_ptr[col_idx];
        } else {
            cols_shared[offs_n] = -1;
        }
        __syncthreads();

        int cols = cols_shared[offs_n];

        // Load K and V into shared memory
        if (offs_n < BLOCK_N && cols >= 0 && cols < context_size && thread_d < BLOCK_DMODEL) {
            int k_index = cols * stride_kn + offs_k * stride_kk;
            int v_index = cols * stride_vn + offs_k * stride_vk;
            k_shared[offs_n][thread_d] = K[k_index];
            v_shared[offs_n][thread_d] = V[v_index];
        } else {
            k_shared[offs_n][thread_d] = 0.0f;
            v_shared[offs_n][thread_d] = 0.0f;
        }
        __syncthreads();

        // Compute qk and p
        float qk = 0.0f;
        if (offs_m < context_size && offs_n < BLOCK_N && cols >= 0) {
            for (int d = 0; d < BLOCK_DMODEL; ++d) {
                qk += q_shared[thread_m][d] * k_shared[offs_n][d];
            }
            qk *= qk_scale;

            // Apply causal mask if needed
            if (CAUSAL) {
                if ((P_SEQ + offs_m) < cols) {
                    qk = -INFINITY;
                }
            }

            float p = exp2f(qk - l_i);

            // Compute dv
            float dv_val = p * do_shared[thread_m][thread_d];

            // Atomic add to DV
            int dv_index = cols * stride_dvn + offs_k * stride_dvk;
            atomicAdd(&DV[dv_index], dv_val);

            // Compute dp
            float dp = -D_i + 0.0f;  // Assuming D_i is scalar per row
            for (int d = 0; d < BLOCK_DMODEL; ++d) {
                dp += do_shared[thread_m][d] * v_shared[offs_n][d];
            }

            // Compute ds
            float ds = p * dp * sm_scale;

            // Compute dk
            float dk_val = ds * q_shared[thread_m][thread_d];

            // Atomic add to DK
            int dk_index = cols * stride_dkn + offs_k * stride_dkk;
            atomicAdd(&DK[dk_index], dk_val);

            // Compute dq
            dq_shared[thread_m][thread_d] += ds * k_shared[offs_n][thread_d];
        }
        __syncthreads();
    }

    // Write back DQ
    if (offs_m < context_size && thread_d < BLOCK_DMODEL) {
        int dq_index = offs_m * stride_dqm + offs_k * stride_dqk;
        DQ[dq_index] = dq_shared[thread_m][thread_d];
    }
}
