#include <iostream>
#include "2d_utils.h"
#include <cstring>

// #define CHECK_ERROR

#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define ABS(x, y) (((x) > (y)) ? ((x) - (y)) : ((y) - (x)))

void cpu_2d_7r_float(const float *in, float *out, const float *param, const int m, const int n){
    int halo = 7;
    int cols = n + 2 * halo + 4;

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            float result = 0.0;
            for(int ii = -halo; ii <= halo; ii++) {
                for(int jj = -halo; jj <= halo; jj++) {
                    result += in[IDX(i+ii+halo, j+jj+halo+1, cols)] * param[IDX(ii+halo, jj+halo, 15)];
                }
            }
            out[IDX(i, j, n)] = result;
        }
    }
}

void printHelp()
{
    const char *helpMessage =
        "Program name: convstencil_2d_float\n"
        "Usage: convstencil_2d_float input_size_of_first_dimension input_size_of_second_dimension time_iteration_size\n";
    printf("%s\n", helpMessage);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printHelp();
        return 1;
    }

    int m = 0;
    int n = 0;
    int times = 0;
    try
    {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
        times = std::stoi(argv[3]);
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e)
    {
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }

    float param[225] = {0.0};
    for (int i = 0; i < 225; i++) {
        // param[i] = 1.0;
        param[i] = i % 17 + 1.0;
    }

    // std::cout << "Params: \n";
    // for(int i = 0; i < 15; i++) {
    //     for(int j = 0; j < 15; j++) {
    //       std::cout << param[IDX(i, j, 15)] << " ";
    //     }
    //     std::cout << std::endl;
    // } 

    // std::cout << "Input matrix size: " << m << " x " << n << std::endl;
    // std::cout << "Time iteration size: " << times << std::endl;

    int halo = 7;

    int rows = m + 2 * halo;
    int cols = n + 2 * halo + 4;

    size_t matrix_size = (unsigned long)rows * cols * sizeof(float);

    float *in = (float *)malloc(matrix_size);
    float *out = (float *)malloc(matrix_size);

    memset(in, 0, matrix_size);
    memset(out, 0, matrix_size);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            // in[IDX(i+halo, j+halo+1, cols)] = 1.0;
            in[IDX(i+halo, j+halo+1, cols)] = IDX(i, j, n) % 17 + 1;
        }
    }

    // std::cout << "Input:" << std::endl;
    // for(int i = 0; i < rows; i++) {
    //     for(int j = 0; j < cols && j < 32; j++) {
    //         std::cout << in[IDX(i, j, cols)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
#ifdef CHECK_ERROR
    float *cpu_out = (float *)malloc(m * n * sizeof(float));
    memset(cpu_out, 0, m * n * sizeof(float));

    cpu_2d_7r_float(in, cpu_out, param, m, n);

    // std::cout << "CPU output:" << std::endl;
    // for(int i = 0; i < m; i++) {
    //     for(int j = 0; j < n && j < 32; j++) {
    //         std::cout << cpu_out[IDX(i, j, n)] << " ";
    //     }
    //     std::cout << std::endl;
    // }
#endif

    gpu_box_2d1r_float(in, out, param, times, m, n);

    // std::cout << "GPU output:" << std::endl;
    // for(int i = 0; i < m; i++) {
    //     for(int j = 0; j < n && j < 32; j++) {
    //         std::cout << out[IDX(i+halo, j+halo, cols)] << " ";
    //     }
    //     std::cout << std::endl;
    // }

#ifdef CHECK_ERROR
    bool correct = true;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if (ABS(cpu_out[IDX(i, j, n)], out[IDX(i+halo, j+halo+1, cols)]) > 1e-5) {
                correct = false;
                std::cout << "Error at (" << i << "," << j << "): CPU = " << cpu_out[IDX(i, j, n)] << ", GPU = " << out[IDX(i+halo, j+halo, cols)] << std::endl;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        std::cout << "Result = PASS" << std::endl;
    } else {
        std::cout << "Result = FAIL" << std::endl;
    }
#endif

    free(in);
    free(out);

    return 0;
}