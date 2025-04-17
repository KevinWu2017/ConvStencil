cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(spmm_coo VERSION 0.01 LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
enable_language(CXX CUDA)

# Define the include DIRs
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

# Define the link libraries
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/usr/local/cuda/lib64)
add_executable(convstencil_1d src/1d/main.cu src/1d/gpu_1r.cu src/1d/gpu_2r.cu src/1d/1d_utils.h)
# target_link_libraries( gpu_box2d1r ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
set_target_properties(
    convstencil_1d
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(convstencil_1d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math>)

# 定义一个函数来添加cuDNN可执行文件，统一处理共同配置
function(add_cudnn_executable target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} ${CUDA_cudnn_LIBRARY})
  set_target_properties(
    ${target_name}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -lcudnn --use_fast_math>)
  target_link_libraries(${target_name} ${CUDA_LIBRARIES} cudnn)
endfunction()

# 使用函数添加所有cuDNN可执行文件
add_cudnn_executable(cudnn_1d3p src/cudnn/conv_1d3p.cu)

add_cudnn_executable(cudnn_1d5p src/cudnn/conv_1d5p.cu)

add_cudnn_executable(cudnn_box2d49p src/cudnn/conv_box2d49p.cu)
add_cudnn_executable(cudnn_box2d49p_float src/cudnn/conv_box2d49p_float.cu)
add_cudnn_executable(cudnn_box2d49p_half src/cudnn/conv_box2d49p_half.cu)

add_cudnn_executable(cudnn_box2d9p src/cudnn/conv_box2d9p.cu)
add_cudnn_executable(cudnn_box2d9p_float src/cudnn/conv_box2d9p_float.cu)
add_cudnn_executable(cudnn_box2d9p_half src/cudnn/conv_box2d9p_half.cu)

add_cudnn_executable(cudnn_box3d27p src/cudnn/conv_box3d27p.cu)


add_executable(convstencil_2d src/2d/main.cu src/2d/gpu.cu src/2d/2d_utils.h)
# target_link_libraries( gpu_box2d1r ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
set_target_properties(
    convstencil_2d
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(convstencil_2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math>)

add_executable(convstencil_3d src/3d/main.cu src/3d/gpu_box.cu src/3d/gpu_star.cu  src/3d/3d_utils.h)
# target_link_libraries( gpu_box2d1r ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} )
set_target_properties(
    convstencil_3d
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(convstencil_3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math>)

# set(CMAKE_CUDA_ARCHITECTURES 80)  
# add_subdirectory(breakdown)
