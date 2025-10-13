#include <mma.h>
#include <cuda.h>
// #include <cuda_runtime.h>
// #include "../utils.h"
#include <iostream>
#include "2d_utils.h"
#include <chrono>

using namespace nvcuda;

#define BLOCK_SIZE_ROW 8
#define BLOCK_SIZE_COL 256
#define HALO 7
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + HALO * 2)
#define PAD 4
#define SM_SIZE_COL (15 * D_BLOCK_SIZE_ROW + PAD)
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 16)
#define UNIT_LENGTH 15
#define TENSOR_CORE_M 16 // 应该是N，但是M=N=16
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define WARP_PER_BLOCK 4
// #define ACCS_PER_WARP (BLOCK_SIZE_COL * BLOCK_SIZE_ROW / 64 / WARP_PER_BLOCK)
#define MMA_NUM 29
#define ceild(n,d)	(((n)-1)/(d) + 1)

__constant__ float param_matrix_d[2 * 232 * TENSOR_CORE_M];


__global__ void kernel2d (const float * __restrict__ in, float * __restrict__ out, const int ldm, const int * __restrict__ lookup_table1, const int * __restrict__ lookup_table2) {
    __shared__ float sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];
    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;
    int totalThreads = blockDim.x;
#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += totalThreads) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
        sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
    }
    __syncthreads();

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
    //     printf("Shared memory 1:\n");
    //     for (int i = 0; i < SM_SIZE_ROW; i++) {
    //         printf("Row %d: ", i);
    //         for (int j = 0; j < SM_SIZE_COL; j++) {
    //             printf("%.0f ", sharedmem[0][i * SM_SIZE_COL + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("Shared memory 2:\n");
    //     for (int i = 0; i < SM_SIZE_ROW; i++) {
    //         printf("Row %d: ", i);
    //         for (int j = 0; j < SM_SIZE_COL; j++) {
    //             printf("%.0f ", sharedmem[1][i * SM_SIZE_COL + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    int warp_id = threadIdx.x / 32;

    nvcuda::wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> param_frag[2][MMA_NUM];
#pragma unroll
    for (int i = 0; i < MMA_NUM; i++) {
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 8 * 16, 16);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 232 * 16 + i * 8 * 16, 16);
    }

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> in_frag;

    for (int col = warp_id * 2 * UNIT_LENGTH; col < warp_id * 2 * UNIT_LENGTH + 2 * UNIT_LENGTH; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 8, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 8, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }

        wmma::store_matrix_sync(out + begin + IDX(HALO + col / UNIT_LENGTH, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}

/**
 * @param in input array pointer
 * @param out output array pointer
 * @param params parameter array pointer (length 225)
 * 
*/
void gpu_box_2d1r_float(const float * __restrict__ in, float * __restrict__ out, const float * __restrict__ params, const int times, const int input_m, const int input_n) {
    float param_matrix_h[2][232 * 16] = {0.0};

    // Initialize parameter matrix
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 16 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 16 + col] = params[i * UNIT_LENGTH + j - col + 15];
                }
            }
        }
    }

    // printf("Parameter matrix 1:\n");
    // for (int i = 0; i < 232; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         printf("%.0f ", param_matrix_h[0][i * 16 + j]);
    //     }
    //     printf("\n");
    // }

    // printf("Parameter matrix 2:\n");
    // for (int i = 0; i < 232; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         printf("%.0f ", param_matrix_h[1][i * 16 + j]);
    //     }
    //     printf("\n");
    // }
    
    CUDA_CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 16 * 232 * sizeof(float)));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO + 4;
    const size_t array_size = rows * cols * sizeof(float);
    float *  array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    
    const int BLOCK_M = (input_m + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW; 
    const int BLOCK_N = (input_n + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL; 
    dim3 grid_config(BLOCK_M, BLOCK_N);
    // dim3 grid_config(1, 1);
    dim3 block_config(32 * WARP_PER_BLOCK);

    // Lookup table
    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 16 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 16 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }

    // printf("Lookup table 1:\n");
    // for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
    //     printf("Row %d: ", i);
    //     for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
    //         printf("%d ", lookup_table1_h[i][j]);
    //     }
    //     printf("\n");
    // }

    // printf("Lookup table 2:\n");
    // for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
    //     printf("Row %d: ", i);
    //     for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
    //         printf("%d ", lookup_table2_h[i][j]);
    //     }
    //     printf("\n");
    // }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CUDA_CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    

    // timing
    int i = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(; i < times; i++) {
        CUDAKERNELCHECK((kernel2d<<<grid_config, block_config>>>(array_d[i % 2], array_d[(i + 1) % 2], cols, lookup_table1_d, lookup_table2_d)));
        cudaDeviceSynchronize();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "ConvStencil(2D): " << std::endl;
    // std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    double secs = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6;

    std::cout <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << ", " << ((double)input_m * input_n) / secs / 1e9 * times * 7 << std::endl;
    
    CUDA_CHECK(cudaMemcpy(out, array_d[i % 2], array_size, cudaMemcpyDeviceToHost));

    return;
}
